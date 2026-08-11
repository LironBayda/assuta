"""
core/kan_layers.py

Kolmogorov-Arnold Network (KAN) layer, "efficient-kan" style: each layer is
    base_activation(x) @ base_weight   +   B-spline_basis(x) @ spline_weight
i.e. nn.Linear + a learnable per-edge nonlinearity (the spline term), which
is what gives KAN its expressiveness advantage over a plain MLP for sharp /
non-smooth functions (see pet/pikan_pet.py's hard-gated k3 head for why that
matters here).

`base_activation` mirrors core/model.py's f_x activation convention ("sine"
or "tanh"), plus "silu" (the standard efficient-kan default) so this is a
genuine drop-in alternative trunk for PhysicsInformedNN, not a separate
one-off architecture.

INIT, per activation (see docstring on reset_parameters for the reasoning
and the literature this is based on):
    - "sine": SIREN init (Sitzmann et al. 2020) -- uniform(-1/fan_in, 1/fan_in)
      for the first layer, uniform(-sqrt(6/fan_in)/omega_0, ...) elsewhere.
      This is copied from this repo's own SineLayer (core/model.py) --
      confirmed against current SIREN literature to be the exact standard
      scheme (searched this session; e.g. arXiv:2405.18084, arXiv:2509.12980
      both state the identical bound). See core/pinn.py for a real bug this
      session found and fixed: PhysicsInformedNN was blanket-reinitializing
      SineLayer's weights AFTER construction, silently destroying this init
      every run with activation="sine". The same discipline is applied here
      -- reset_parameters() is the ONLY place base_weight gets initialized,
      and callers must not re-apply a generic init on top of it.
    - "tanh": Xavier/Glorot uniform -- the standard recommendation for
      bounded, zero-centered activations (tanh, sine-without-omega-scaling),
      designed to keep activation variance roughly constant across layers
      (Glorot & Bengio 2010). Matches core/pinn.py's existing init_weights
      for the tanh branch of f_x.
    - "silu" (default): kaiming_uniform_, the standard choice for
      ReLU-family activations, unchanged from the original efficient-kan
      default.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    A single KAN layer: in_features -> out_features.

    Parameters
    ----------
    grid_size, spline_order : spline resolution (see B-spline basis below).
    base_activation : "silu" (default), "tanh", or "sine".
    omega_0 : frequency scale, only used when base_activation="sine" (see
        core/model.py's SineLayer / f_x docstring for what this controls).
    is_first : only meaningful for base_activation="sine" -- whether this is
        the network's first layer (different init bound, see reset_parameters).
    grid_range : input domain the SPLINE grid covers. Normalize your inputs
        to roughly this range; outside it the spline falls back to a linear
        tail. Independent of the base-path activation/init above.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        omega_0: float = 1.0,
        is_first: bool = False,
        grid_range=(-1.0, 1.0),
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ):
        super().__init__()
        if base_activation not in ("silu", "tanh", "sine"):
            raise ValueError(f"base_activation must be 'silu', 'tanh', or 'sine', got {base_activation!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation_name = base_activation
        self.omega_0 = omega_0
        self.is_first = is_first

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise

        self.reset_parameters()

    def _base_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_activation_name == "silu":
            return F.silu(x)
        elif self.base_activation_name == "tanh":
            return torch.tanh(x)
        else:  # "sine"
            return torch.sin(self.omega_0 * x)

    def reset_parameters(self):
        """
        Initializes base_weight per the activation-specific scheme described
        in this file's module docstring, and spline_weight via the standard
        efficient-kan noise-fit (activation-independent -- the spline term
        is a separate learnable nonlinearity, not coupled to the base path's
        activation choice).
        """
        if self.base_activation_name == "sine":
            # SIREN init (Sitzmann et al. 2020), identical scheme to this
            # repo's core/model.py SineLayer.init_weights.
            with torch.no_grad():
                if self.is_first:
                    bound = 1.0 / self.in_features
                else:
                    bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.base_weight.uniform_(-bound, bound)
        elif self.base_activation_name == "tanh":
            nn.init.xavier_uniform_(self.base_weight)
        else:  # "silu"
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        nn.init.zeros_(self.base_bias)

        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                self.scale_spline
                * self._curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                )
            )

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (
                grid[:, k:-1] - grid[:, : -(k + 1)]
            ) * bases[:, :, :-1]
            right = (grid[:, k + 1 :] - x) / (
                grid[:, k + 1 :] - grid[:, 1:-k]
            ) * bases[:, :, 1:]
            bases = left + right
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self._b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_out = F.linear(self._base_activation(x), self.base_weight, self.base_bias)
        spline_basis = self._b_splines(x)
        spline_out = F.linear(
            spline_basis.view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        out = base_out + spline_out
        return out.reshape(*orig_shape[:-1], self.out_features)


class KAN(nn.Module):
    """
    Stack of KANLinear layers -- drop-in replacement for an MLP.

    base_activation / omega_0 apply to every layer. `is_first` is set
    automatically for layer 0 only (matches SineLayer's convention: only the
    network's very first layer uses the 1/fan_in bound).
    """

    def __init__(
        self,
        layers_hidden,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        omega_0: float = 1.0,
        grid_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                KANLinear(
                    in_f,
                    out_f,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    omega_0=omega_0,
                    is_first=(i == 0),
                    grid_range=grid_range,
                )
                for i, (in_f, out_f) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:]))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

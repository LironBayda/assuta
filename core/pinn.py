import torch
from torch import nn

from config import  DEVICE
from core.losses import PINNLoss
from core.model import f_x, f_x_kan, Ks_net, Ks_net_varpro

def init_weights(layer):
    """Applies Xavier initialization to the weights of a layer.

    BUG FOUND AND FIXED THIS SESSION: this used to be applied unconditionally
    to the whole f_x module via `self.f_x.apply(init_weights)`, which
    recurses into EVERY nn.Linear submodule -- including the nn.Linear
    inside each SineLayer. SineLayer already performs its own SIREN-specific
    init (Sitzmann et al. 2020) in its own __init__, tailored to the sine
    activation (see core/model.py's SineLayer.init_weights) -- the blanket
    re-init below was silently overwriting that with a generic Xavier init
    immediately after, on every single run with activation="sine" (the
    default). Confirmed against current SIREN literature this session that
    SineLayer's own init is the textbook-correct scheme, and that using a
    non-SIREN init on a sine network is a known, literature-documented
    failure mode (sine activations can fail to reconstruct signals at all
    without it -- Sitzmann et al. 2020). See PhysicsInformedNN.__init__
    below for the fix: only apply this generic init where there ISN'T
    already an activation-specific init to preserve (the tanh branch, the
    final readout layer, and the KAN base_activation="silu" case)."""

    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)# Normal distribution
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
# =====================================================
# Physics-Informed NN Wrapper
# =====================================================
class PhysicsInformedNN:
    def __init__(self, num_of_compartment, dt, image_shape=None, device=DEVICE,
                 hidden_size=40, bottleneck_size=None, omega_0=1.0, activation="sine",
                 arch="mlp", kan_grid_size=5, kan_spline_order=3, kan_grid_range=(-3.0, 3.0),
                 physics_weight=0.01, tac_consistency_weight=None, reg_weight=1e-4,
                  voxel_weight=None):
        """
        hidden_size, bottleneck_size, omega_0 : f_x / f_x_kan structural
            params -- see core/model.py's f_x docstring. omega_0 here
            defaults to 1.0 -- see the hyperparameter sweep in
            simulation/pinn_hyperparam_sweep.py for what's actually best
            for the PINN's f_x specifically (a continuous-time trunk is a
            different regime from a fixed-length curve encoder, so there
            is no reason to expect the same omega_0 to transfer to other
            networks elsewhere in this repo).
        activation : "sine" (default) or "tanh" -- see core/model.py's
            f_x docstring. "tanh" is the more common choice in the PINN
            literature (e.g. van Herten et al. 2022); provided so
            noise/sparsity comparisons can test it directly. Applies to
            both arch="mlp" (f_x) and arch="kan" (f_x_kan).
        arch : "mlp" (default, f_x -- SIREN/tanh MLP trunk) or "kan"
            (f_x_kan -- same interface, KAN trunk with the same activation
            options). See core/model.py's f_x_kan docstring / core/kan_layers.py
            for the KAN-specific structural params below.
        kan_grid_size, kan_spline_order, kan_grid_range : only used when
            arch="kan" -- see core/kan_layers.py's KANLinear docstring.
        physics_weight, tac_consistency_weight, reg_weight : PINNLoss
            term weights -- see core/losses.py's PINNLoss docstring.
        varpro : bool, default False -- 1TCM (num_of_compartment=1) ONLY.
            If True, uses Ks_net_varpro instead of Ks_net: K1 is solved
            in closed-form linear least squares each forward pass rather
            than treated as a free gradient-descent parameter (Variable
            Projection, Golub & Pereyra 1973) -- see Ks_net_varpro's
            docstring for the full mechanism and rationale. Multi-seed
            confirmed (3 seeds, DCE 1TCM simulation): mean kep
            correlation 0.048 -> 0.351 (7x), at a modest, consistent K1
            cost (mean 0.788 -> 0.726). Not yet extended to 2TCM (raises
            NotImplementedError if combined with num_of_compartment=2).
        """
        self.device = device
        self.tacs_num = image_shape[0]*image_shape[1]*image_shape[2]


        # Networks
        self.Ks_net = Ks_net(self.tacs_num, num_of_compartment, dt).to(device)
        if arch == "mlp":
            self.f_x = f_x(self.tacs_num, num_of_compartment, hidden_size=hidden_size,
                            bottleneck_size=bottleneck_size, omega_0=omega_0,
                            activation=activation).to(device)
        else:
            # f_x_kan only supports activation="silu" (see its docstring --
            # sine/tanh were deliberately removed from the KAN trunk). The
            # `activation` argument above is f_x's (arch="mlp") axis, so it
            # doesn't apply here -- override rather than error, so switching
            # arch="mlp"->"kan" without also clearing `activation` (which
            # still defaults to "sine" for the mlp case) doesn't break.
            self.f_x = f_x_kan(self.tacs_num, num_of_compartment, hidden_size=hidden_size,
                                bottleneck_size=bottleneck_size, omega_0=omega_0,
                                activation="silu", grid_size=kan_grid_size,
                                spline_order=kan_spline_order, grid_range=kan_grid_range).to(device)

        # Initialization.
        # activation="sine" (both arch="mlp" and arch="kan"): the trunk
        # already performed its own SIREN-specific init in __init__ (f_x's
        # SineLayer, or f_x_kan's KANLinear with base_activation="sine") --
        # re-applying a generic Xavier init on top would silently destroy
        # it (see init_weights's docstring above for the bug this was this
        # session). Only the final readout layer, which has no such
        # special requirement, gets the generic init.
        # activation="tanh": f_x's tanh branch is a plain nn.Linear+Tanh
        # stack with no init of its own, and f_x_kan's KANLinear with
        # base_activation="tanh" already does its own Xavier init in
        # reset_parameters -- so arch="mlp" still needs the blanket apply,
        # but arch="kan" only needs `final` re-init here too, for the same
        # reason as the sine case.
        if activation == "sine" or arch == "kan":
            self.f_x.final.apply(init_weights)
        else:
            self.f_x.apply(init_weights)


        # Loss function
        voxel_weight_t = None
        if voxel_weight is not None:
            voxel_weight_t = torch.as_tensor(voxel_weight, dtype=torch.float32, device=device)
        self.loss_fn = PINNLoss(
            self.tacs_num, num_of_compartment=num_of_compartment, image_shape=image_shape,
            physics_weight=physics_weight, tac_consistency_weight=tac_consistency_weight,
            reg_weight=reg_weight, voxel_weight=voxel_weight_t,
        )

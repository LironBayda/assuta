import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PINNLoss(nn.Module):
    """
    Physics-Informed Neural Network (PINN) Loss.

    Combines:
    - Data fidelity (TAC and c_p)
    - L2 regularization
    - Physics (ODE residual + TAC consistency) constraints, optionally
      weighted causally over time so early dynamics are fit before the
      network is allowed to "cheat" on later time points (Wang et al.,
      "Respecting causality is all you need for training physics-informed
      neural networks", 2022).

    Note: the leading dimension of `output1`/`output2` here is the time
    axis (this loss is called on `f_x(t)` output, not an i.i.d. batch),
    which is what makes causal time-weighting meaningful.
    """

    def __init__(
        self,
        tacs_num,
        num_of_compartment=1,
        physics_weight=0.01,  # DCE 0.0001
        tac_consistency_weight=10,
        reg_weight=1e-4,
        causality_eps=1,
        image_shape=None,
        voxel_weight=None,
    ):
        """
        Parameters
        ----------
        tacs_num : int
            Number of TAC time points predicted by the network.
        physics_weight : float
            Weight for the ODE-residual terms in the physics loss.
        tac_consistency_weight : float or None
            Weight for the physics-integrated-TAC vs. network-TAC
            consistency term. Defaults to `physics_weight` if not given
            (previously this term was implicitly weighted at 1.0,
            inconsistent with the residual terms it's added alongside).
        reg_weight : float
            Weight for the L2 regularization term. Previously this term
            only regularized the first time point (`tac_pred[0]`); it now
            covers the full prediction, so the weight defaults small to
            avoid swamping the other terms — retune if you rely on the
            old magnitude.
        causality_eps : float
            Causal weighting strength for the physics loss over time.
            0.0 (default) reproduces the original uniform-weighting
            behavior exactly. Increase (e.g. via annealing across epochs)
            to enforce that early-time residuals are learned before later
            ones are allowed to dominate the loss.
        image_shape : tuple or None
            Reserved for a future spatial-prior loss term. Not used
            internally by this class currently.
        voxel_weight : torch.Tensor, shape (tacs_num,), or None
            Per-voxel multiplicative weight applied to every loss term
            before summing over voxels. Fixes a real failure mode found
            via simulation: all loss terms here SUM (not average) over
            voxels, so a small lesion (e.g. 66 voxels among 26000
            background voxels) contributes a negligible fraction of the
            total loss/gradient regardless of how wrong its fit is — the
            shared f_x trunk's gradient ends up dominated by fitting the
            majority class well, and the minority class's Ks_raw entries
            get comparatively little useful gradient signal even though
            they are technically independent parameters. Pass e.g.
            inverse-class-frequency weights (from a coarse segmentation
            or prior voxelwise pass) to counteract this. None (default)
            reproduces the original uniform-weighting behavior exactly.
        """
        super().__init__()
        self.tacs_num = tacs_num
        self.physics_weight = physics_weight
        self.tac_consistency_weight = (
            tac_consistency_weight if tac_consistency_weight is not None else physics_weight
        )
        self.reg_weight = reg_weight
        self.causality_eps = causality_eps
        self.image_shape = image_shape
        self.num_of_compartment = num_of_compartment
        self.voxel_weight = voxel_weight  # (tacs_num,) or None


        # Populated after each forward() call: dict of unweighted and
        # weighted component losses, for logging or external adaptive
        # weighting schemes (e.g. NTK-based balancing).
        self.last_components = {}

    # ---------------------------
    # Internal loss functions
    # ---------------------------
    def _squared_error(self, output, target):
        if self.voxel_weight is not None:
            return torch.sum(self.voxel_weight * (output - target) ** 2)
        return torch.sum((output - target) ** 2)

    def _squared_sum(self, x):
        if self.voxel_weight is not None:
            return torch.sum(self.voxel_weight * x ** 2)
        return torch.sum(x ** 2)

    def _negative_penalty(self, x):
        return torch.sum(x[x < 0] ** 2)

    def _per_time_squared_error(self, output, target):
        """Squared error summed over all dims except the leading time dim,
        weighted per-voxel if self.voxel_weight is set."""
        diff2 = (output - target) ** 2
        if self.voxel_weight is not None:
            diff2 = diff2 * self.voxel_weight
        return diff2.reshape(diff2.shape[0], -1).sum(dim=1)

    def _per_time_squared_sum(self, x):
        x2 = x ** 2
        if self.voxel_weight is not None:
            x2 = x2 * self.voxel_weight
        return x2.reshape(x.shape[0], -1).sum(dim=1)

    def _causal_weights(self, per_time_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized causal weights w_i = exp(-eps * sum_{k<i} L_k),
        detached from the graph (weights are treated as constants w.r.t.
        optimization, per Wang et al.). eps=0 -> uniform weights.
        """
        T = per_time_loss.shape[0]
        with torch.no_grad():
            exclusive_cumsum = torch.cumsum(per_time_loss, dim=0) - per_time_loss
            weights = torch.exp(-self.causality_eps * exclusive_cumsum)
            weights = weights / (weights.sum() + 1e-12) * T
        return weights

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, outputs, targets):
        """
        Compute total loss.

        Parameters
        ----------
        outputs : tuple
            (output1, output2)
            - output1: [T, tacs_num + c_p_dim] network predictions (TAC + c_p)
              over T time points.
            - output2: [T, ...] physics outputs.
        targets : tuple
            (tac_real, c_p_real)
            - tac_real: [T, tacs_num]
            - c_p_real: [T, c_p_dim] or [T]
        """
        output1, output2 = outputs
        tac_real, c_p_real = targets
        T = output1.shape[0]

        # -------------------
        # Split TAC and c_p
        # -------------------
        if self.num_of_compartment == 1:
            tac_pred = output1[:, : self.tacs_num]
        else:
            tac_pred = output1[:, : self.tacs_num] + output1[:, self.tacs_num : -1]
        c_p_pred = output1[:, -1].view(T, -1)
        c_p_real = c_p_real.view(T, -1)
        tac_real = tac_real.view(T, -1)

        # -------------------
        # Data Loss
        # -------------------
        loss_tac = self._squared_error(tac_pred, tac_real)
        loss_cp = self._squared_error(c_p_pred, c_p_real)

        # -------------------
        # Regularization (now over the full prediction, not just t=0)
        # -------------------
        loss_reg = self.reg_weight * (
            self._negative_penalty(tac_pred)
            + self._negative_penalty(c_p_pred)
            + self._squared_sum(tac_pred[0])
            + self._squared_sum(c_p_pred[0])
        )

        # -------------------
        # Physics Loss (causally weighted over time)
        # -------------------
        # Per-time-point ODE residual magnitude, summed across all
        # intermediate physics outputs.
        residual_per_time = sum(
            self._per_time_squared_sum(x) for x in output2[:-1]
        )

        # Per-time-point TAC-consistency residual.
        tac_phys = output2[-1].view(T, -1)
        consistency_per_time = self._per_time_squared_error(tac_phys, tac_pred)

        combined_per_time = (
            self.physics_weight * residual_per_time
            + self.tac_consistency_weight * consistency_per_time
        )
        causal_weights = self._causal_weights(combined_per_time)
        loss_phys = (causal_weights * combined_per_time).sum()

        # -------------------
        # Total Loss
        # -------------------
        total_loss = loss_tac + loss_cp + loss_reg + loss_phys

        self.last_components = {
            "loss_tac": loss_tac.detach(),
            "loss_cp": loss_cp.detach(),
            "loss_reg": loss_reg.detach(),
            "loss_phys": loss_phys.detach(),
            "total_loss": total_loss.detach(),
            "causal_weights_min": causal_weights.min().detach(),
        }

        return total_loss
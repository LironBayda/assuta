import torch
from torch import nn

from config import  DEVICE
from core.losses import PINNLoss
from core.model import f_x, f_x_bpinn, Ks_net

def init_weights(layer):
    """Applies Normal initialization to the weights of a layer."""

    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)# Normal distribution
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
# =====================================================
# Physics-Informed NN Wrapper
# =====================================================
class PhysicsInformedNN:
    def __init__(self, num_of_compartment, dt, image_shape=None, device=DEVICE,
                 bayesian=False, dropout_p=0.1,
                 hidden_size=40, bottleneck_size=None, omega_0=1.0, activation="sine",
                 physics_weight=0.01, tac_consistency_weight=None, reg_weight=1e-4,
                 ks_init=None, voxel_weight=None):
        """
        bayesian : bool
            If True, use f_x_bpinn (MC-Dropout "Sine B-PINN") instead of
            the deterministic f_x -- see core/model.py for details. The
            dropout stays active at inference too, so repeated forward
            passes give a distribution rather than a point estimate.
        hidden_size, bottleneck_size, omega_0 : f_x/f_x_bpinn structural
            params -- see core/model.py's f_x docstring. omega_0 here
            defaults to 0.1 (f_x's own default), NOT the 10.0 found best
            for the unrelated MVE encoder elsewhere in this
            repo -- see the hyperparameter sweep in
            simulation/pinn_hyperparam_sweep.py for what's actually best
            for the PINN's f_x specifically (a continuous-time trunk is a
            different regime from a fixed-length curve encoder, so there
            is no reason to expect the same omega_0 to transfer).
        activation : "sine" (default) or "tanh" -- see core/model.py's
            f_x docstring. "tanh" is the more common choice in the PINN
            literature (e.g. van Herten et al. 2022); provided so
            noise/sparsity comparisons can test it directly.
        physics_weight, tac_consistency_weight, reg_weight : PINNLoss
            term weights -- see core/losses.py's PINNLoss docstring.
        """
        self.device = device
        self.bayesian = bayesian
        self.tacs_num = image_shape[0]*image_shape[1]*image_shape[2]

        # Networks
        if bayesian:
            self.f_x = f_x_bpinn(self.tacs_num, num_of_compartment, hidden_size=hidden_size,
                                  bottleneck_size=bottleneck_size, omega_0=omega_0,
                                  dropout_p=dropout_p, activation=activation).to(device)
        else:
            self.f_x = f_x(self.tacs_num, num_of_compartment, hidden_size=hidden_size,
                            bottleneck_size=bottleneck_size, omega_0=omega_0,
                            activation=activation).to(device)
        self.Ks_net = Ks_net(self.tacs_num, num_of_compartment, dt, ks_init=ks_init).to(device)
        #initialization
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




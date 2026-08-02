"""
Global configuration file for the Physiological Maps Pipeline
(PET + DCE + Dictionary Learning + PINN)

Author: You
"""

from pathlib import Path

import numpy as np
import torch

# ============================================================
# -------------------- SYSTEM CONFIG -------------------------
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# -------------------- DCE / MRI CONFIG ----------------------
# ============================================================

# SPGR Parameters
SPGR = {
    "alpha_static_deg": [2.0, 15.0],   # Flip angles for T1 mapping
    "alpha_dynamic_deg": 15.0,         # Dynamic sequence flip angle
    "TR_sec": 0.00518,
    "r1_relaxivity": 4.5               # 1/(mM·s)
}

# ============================================================
# -------------------- DICTIONARY CONFIG ---------------------
# ============================================================

BLOOD_DICTIONARY = {
    "num_atoms": 3,         # Blood, Tissue, Background
    "dt_seconds": 9.5,
    "K1": 0.1,
    "k2": 0.55,
    "blood_threshold": 0.9
}

# ============================================================
# -------------------- AIF CONFIG ----------------------------
# ============================================================

AIF = {
    "type": "parker",
    "units": "minutes"
}

# ============================================================
# -------------------- PINN CONFIG ---------------------------
# ============================================================

PINN = {
    "num_compartments": 2,
    "epochs":1000, #DCE 100
    "learning_rate": 0.001,
    "num_batches": 1,
    "dt": 0.152244
}
# ============================================================
# -------------------- DCE CONFIG ---------------------------
# ================================= ===========================

DCE = {
    "dt": np.asarray([0.1522]*35),
    "num_point": 35,
    "blood_num_frames": 35

}


PET = {
    "dt": np.asarray([10,20,30,40,50,60,90,120,150,180,210,240,270,300,330,360,410,460,510,560,610,660,960,1200]),
    "num_point": 24,
    "cvae_path": "/home/liron/Downloads/CVAE(25).pth",
    "blood_num_frames":10

}

# ============================================================
# -------------------- TRAINING CONFIG -----------------------
# ============================================================

TRAINING = {
    "use_amp": False,
    "save_k_maps": True,
    "verbose": True
}
# ============================================================
# -------------------- INPUT FILE NAMES ---------------------
# ============================================================

CVAE = {
    "latent_dim" : 1

}

# ============================================================
# -------------------- OUTPUT FILE NAMES ----1----------------
# ============================================================

OUTPUTS = {
    "blood_mask": "blood.nii",
    "k_param_prefix": "K_",
}

# ============================================================
# -------------------- SAFETY CHECK --------------------------
# ============================================================

def print_config():
    """Prints the full configuration (for experiment reproducibility)."""
    import pprint
    cfg = {
        "DEVICE": DEVICE,
        "SPGR": SPGR,
        "BLOOD_DICTIONARY": BLOOD_DICTIONARY,
        "AIF": AIF,
        "PINN": PINN,
        "TRAINING": TRAINING,
        "OUTPUTS": OUTPUTS,
        "CVAE":CVAE
    }
    pprint.pprint(cfg)

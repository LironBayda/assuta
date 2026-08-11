import os

import numpy as np

from config import PINN, DEVICE, PET
from core.train import Trainer
from pet.preprocessing import preprocessing
from simulation.voxelwise_pet import calculate_pet_voxelwise, calculate_pet_voxelwise_1tcm
from dce.analysis import save_maps_as_nifti


def pipeline(
        path,
        num_of_compartment=2,
        epochs=PINN["epochs"],
        device=DEVICE,
        method="pinn",
        windowed=True,
        axis="xy",
        stride=None,
        activation="sine",
        arch="kan",
):
    """
    Full PET kinetic pipeline -- mirrors dce.analysis.pipeline's
    structure/method dispatch ("pinn" / "voxelwise") rather than being a
    PINN-only hardcoded function, for consistency across modalities.

    activation, arch : trunk configuration for method="pinn", forwarded to
        core.train.Trainer -- see Trainer's docstring for the three
        supported configurations (arch="mlp"+activation="sine"/"tanh", or
        arch="kan" which is silu-only and ignores `activation`). Untested
        on PET specifically as of this wiring -- the mlp+sine recipe this
        Trainer's other defaults were tuned against was a 1TCM/DCE
        simulation (see Trainer's docstring), not PET/2TCM.

    method:
        "pinn"       -> PINN kinetic fitting (1TCM if num_of_compartment=1,
                        2TCM/irreversible if num_of_compartment=2)
        "voxelwise" -> classical voxelwise NLLS fitting -- 1TCM
                        (calculate_pet_voxelwise_1tcm) if
                        num_of_compartment=1, else the full irreversible
                        2TCM fit (calculate_pet_voxelwise)

    windowed, axis, slice_window, window_size : see
        dce.analysis.pipeline's docstring / core.train.Trainer -- only
        used when method="pinn".

    CAVEAT (num_of_compartment=2, the default here): the recipe these
    defaults reflect (Trainer's own defaults -- hidden_size=10,
    physics_weight annealed 0.01->100, windowed=True/axis=xy/
    window_size=16) was found and multi-seed-tested on 1TCM/DCE
    specifically. The Ve-unified parameterization fix itself
    (core/model.py's convolve_1cm_for_minimize/get_x_1tcm) only applies
    to the 1TCM code path -- 2TCM's get_x_2tcm was already internally
    consistent and untouched by that fix. But the OTHER new defaults
    (hidden_size=10, the physics_weight anneal, windowed/axis/
    window_size) flow through to 2TCM via the shared Trainer regardless,
    without having been independently verified there. If PET/2TCM
    results look worse than before, try overriding hidden_size=40,
    physics_weight=0.01, physics_weight_start=None, windowed=False
    explicitly (the old defaults) until this gets its own sweep.
    """

    # -----------------------------
    # 1. Preprocessing
    # -----------------------------
    img, aif, affine = preprocessing(path)
    # img: (X, Y, Z, T)

    x, y, z, _ = np.asarray(img.shape) //  3
    cropped = img[x:-x, y:-y, :, :].transpose(3, 0, 1, 2)
    # (T, X, Y, Z)

    new_affine = affine.copy()
    new_affine[:3, 3] += new_affine[:3, :3] @ np.array([x, y, z])

    # BUGFIX (found via simulation-based validation this session):
    # PET["dt"] holds per-frame *durations*, not cumulative timestamps --
    # passing it straight through (as this pipeline previously did) gives
    # the PINN/voxelwise fitters a non-monotonic-in-general, physically
    # wrong time axis. dce.analysis.pipeline already cumsum()s DCE["dt"]
    # for exactly this reason; do the same here for consistency and
    # correctness.
    # BUGFIX #2: a PRIOR bugfix here (see the note above) added cumsum()
    # but never converted seconds to minutes. config.PET["dt"] holds frame
    # durations in SECONDS (e.g. [10, 20, 30, ..., 1200] -- 10s, 20s, 30s,
    # ... frames); np.cumsum(...) alone gives cumulative SECONDS. Every
    # other part of this codebase assumes MINUTES: kinetics_literature.py's
    # K1/k2/k3 ranges are explicitly documented "units: 1/min",
    # feng_input_function and every simulation test script this session
    # divided by 60 before using cumsum'd PET timestamps. Without the
    # /60.0 below, this pipeline was feeding a ~60x-too-large time axis to
    # every downstream PINN/voxelwise fit -- e.g. a literature k3~0.11/min
    # value would appear to decay ~60x slower than intended against
    # seconds-scale t, making the tracer look essentially non-trapping
    # within any real scan window. This affects EVERY prior PET pipeline
    # run through this function.
    t = np.cumsum(PET["dt"]) / 60.0

    # -----------------------------
    # 2. PINN
    # -----------------------------
    if method == "pinn":

        trainer = Trainer(
            c_p=aif,
            num_of_compartment=num_of_compartment,
            t=t,
            device=device,
            affine=new_affine,
            save_path=path,
            epochs=epochs,
            activation=activation,
            arch=arch,
            windowed=windowed,
            axis=axis,
            stride=stride,
        )

        return trainer.train(cropped)

    # -----------------------------
    # 3. Classical voxelwise fitting
    # -----------------------------
    elif method == "voxelwise":

        if num_of_compartment == 1:
            maps = calculate_pet_voxelwise_1tcm(cropped, t, aif)
        elif num_of_compartment == 2:
            maps = calculate_pet_voxelwise(cropped, t, aif)
        else:
            raise ValueError("voxelwise PET fitting supports num_of_compartment in {1, 2}")

        save_maps_as_nifti(maps, new_affine, out_dir=path)
        return maps

    else:
        raise ValueError(f"Unknown method {method}")


def run_all_pet(root_path, epochs=1000, device="cpu", method="pinn",
                 windowed=True, axis="xy", slice_window=1, window_size=16, stride=None,
                 activation="sine", arch="mlp"):
    """
    Batch executor for PET pipelines -- mirrors dce.analysis.run_all_dce.
    Processes all subjects matching sub*/pet within the root directory.
    windowed, axis, slice_window, window_size, stride are only used when method="pinn".
    activation, arch : forwarded to pipeline() (method="pinn" only) -- see
    pipeline()'s docstring / Trainer's.
    """
    import glob

    print(f"[INFO] Searching for subjects in: {root_path}")
    subject_paths = sorted(glob.glob(os.path.join(root_path, "sub*")))

    if len(subject_paths) == 0:
        print("[WARNING] No subject folders found matching sub*/")
        return

    print(f"[INFO] Found {len(subject_paths)} subject(s).")

    for i, sub_path in enumerate(subject_paths, start=1):
        subject_id = os.path.basename(sub_path)
        pet_path = os.path.join(sub_path, "pet")
        if not os.path.isdir(pet_path):
            print(f"[SKIP] {subject_id}: no pet/ folder")
            continue
        print(f"[{i}/{len(subject_paths)}] Processing {subject_id}")
        try:
            pipeline(pet_path, epochs=epochs, device=device, method=method,
                     windowed=windowed, axis=axis,
                     stride=stride, activation=activation, arch=arch)
        except Exception as e:
            print(f"[ERROR] {subject_id}: {e}")

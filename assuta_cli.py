"""
assuta CLI -- run kinetic parameter estimation on either simulated data
(a literature-grounded phantom, no real data needed) or real DCE-MRI /
PET subject data (single subject, or a whole batch of subjects).

Usage
-----
Simulation, quick check (small phantom, fast):
    python assuta_cli.py --source simulation --modality dce --method voxelwise
    python assuta_cli.py --source simulation --modality pet --method pinn

Real data, single subject (expects a preprocessed subject directory,
same layout dce.preprocessing / pet.preprocessing already expect):
    python assuta_cli.py --source real --path /data/sub01/pet --modality pet --method pinn
    python assuta_cli.py --source real --path /data/sub01/dce --modality dce --method voxelwise

Real data, a whole folder of subjects at once -- expects the layout
root/sub*/dce and/or root/sub*/pet (any subject folder missing the
relevant modality subfolder is skipped, not an error):
    python assuta_cli.py --source real --batch-root /data --modality dce --method pinn
    python assuta_cli.py --source real --batch-root /data --modality pet --method voxelwise
    python assuta_cli.py --source real --batch-root /data --modality both --method pinn

Windowed PINN training (default for --method pinn -- independent
per-window fits; pass --no-windowed for the older whole-image behavior):
    python assuta_cli.py --source simulation --modality dce --method pinn
    python assuta_cli.py --source simulation --modality dce --method pinn --no-windowed

True 3D box windowing (--axis xyz, requires --window-size as X Y Z):
    python assuta_cli.py --source simulation --modality dce --method pinn --axis xyz --window-size 16 16 8

Install as a console command via `pip install -e .` (see pyproject.toml),
then just run `assuta ...` instead of `python assuta_cli.py ...`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIM_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_data")


def run_simulation(args):
    """Build a phantom, simulate DCE or PET data, run the chosen
    estimation method, print/save a scored comparison against known
    ground truth -- no real data needed."""
    import numpy as np

    from simulation.phantom import build_prostate_phantom
    from simulation.kinetics_literature import (
        DCE_1TCM_PARAMS, PET_2TCM_PARAMS, sample_param_maps,
    )
    from simulation.forward_models import (
        parker_aif, feng_input_function, simulate_1tcm_volume, simulate_2tcm_volume,
    )
    from scipy.stats import pearsonr

    out_dir = os.path.join(SIM_DATA_DIR, "validation_runs", f"cli_{args.modality}_{args.method}")
    os.makedirs(out_dir, exist_ok=True)

    info = build_prostate_phantom(tuple(args.shape), seed=args.seed)
    label = info.label

    if args.modality == "dce":
        gt = sample_param_maps(label, DCE_1TCM_PARAMS, ["K1", "k2"], seed=args.seed)
        t = np.cumsum(np.asarray([0.1522] * 35))
        aif = parker_aif(t)
        img, _ = simulate_1tcm_volume(t, aif, gt["K1"], gt["k2"], noise_std=args.noise_std, rng=args.seed)
        num_of_compartment = 1
        param_names = ["K1", "k2"]
    elif args.modality == "pet":
        gt = sample_param_maps(label, PET_2TCM_PARAMS, ["K1", "k2", "k3"], seed=args.seed)
        pet_dt = np.asarray([10, 20, 30, 40, 50, 60, 90, 120, 150, 180, 210, 240,
                              270, 300, 330, 360, 410, 460, 510, 560, 610, 660, 960, 1200])
        t = np.cumsum(pet_dt) / 60.0
        aif = feng_input_function(t)
        img, _ = simulate_2tcm_volume(t, aif, gt["K1"], gt["k2"], gt["k3"], noise_std=args.noise_std, rng=args.seed)
        num_of_compartment = 2
        param_names = ["K1", "k2", "k3"]
    else:
        raise SystemExit(f"--modality {args.modality!r} not supported for --source simulation "
                          f"(use 'dce' or 'pet'; 'both' is only valid with --source real --batch-root)")

    tissue = label > 0
    print(f"[simulation] modality={args.modality} shape={img.shape} tissue_voxels={tissue.sum()}")

    if args.method == "voxelwise":
        if args.modality == "dce":
            from dce.analysis import calculate_dce_voxelwise
            maps = calculate_dce_voxelwise(img, t, aif, mask=tissue)
        elif num_of_compartment == 1:
            from simulation.voxelwise_pet import calculate_pet_voxelwise_1tcm
            maps = calculate_pet_voxelwise_1tcm(img, t, aif, mask=tissue, verbose=False)
        else:
            from simulation.voxelwise_pet import calculate_pet_voxelwise
            maps = calculate_pet_voxelwise(img, t, aif, mask=tissue, verbose=False)

        for name in param_names:
            r = pearsonr(maps[name][tissue], gt[name][tissue])[0]
            print(f"  {name}: corr={r:.3f}")

    elif args.method == "pinn":
        from core.train import Trainer
        trainer = Trainer(c_p=aif, num_of_compartment=num_of_compartment, t=t,
                           device=args.device, affine=np.eye(4), save_path=out_dir, epochs=args.epochs,
                           windowed=args.windowed, axis=args.axis, slice_window=args.slice_window,
                           window_size=args.window_size, stride=args.stride)
        ks_out, hist = trainer.train(img, z_slices=[0])

        for i, name in enumerate(param_names):
            r = pearsonr(ks_out[i][tissue], gt[name][tissue])[0]
            print(f"  {name}: corr={r:.3f}")

    else:
        raise ValueError(f"--method {args.method!r} not supported for --source simulation "
                          f"(use 'voxelwise' or 'pinn')")

    print(f"[simulation] outputs in {out_dir}")


def run_real_data(args):
    """Run the DCE-MRI or PET pipeline on a real (preprocessed) subject
    directory."""
    if args.path is None:
        raise SystemExit("--path is required for --source real, unless using --batch-root")
    if args.modality == "both":
        raise SystemExit("--modality both is only valid together with --batch-root, not a single --path")

    if args.modality == "dce":
        from dce.analysis import pipeline
    else:
        from pet.analysis import pipeline

    result = pipeline(args.path, method=args.method, epochs=args.epochs, device=args.device,
                       windowed=args.windowed, axis=args.axis, slice_window=args.slice_window,
                       window_size=args.window_size, stride=args.stride)
    print(f"[{args.modality}] pipeline done, method={args.method}, device={args.device}")
    return result


def run_batch(args):
    """
    Batch-process a whole folder of subjects: root/sub*/dce and/or
    root/sub*/pet, using the existing run_all_dce (dce.analysis) /
    run_all_pet (pet.analysis) batch executors -- these already implement
    exactly this folder-structure convention (glob root/sub*, skip any
    subject missing the relevant modality subfolder).
    """
    if args.modality in ("dce", "both"):
        from dce.analysis import run_all_dce
        print(f"[batch] processing root/sub*/dce under {args.batch_root}")
        run_all_dce(args.batch_root, epochs=args.epochs, device=args.device, method=args.method,
                    windowed=args.windowed, axis=args.axis, slice_window=args.slice_window,
                    window_size=args.window_size, stride=args.stride)

    if args.modality in ("pet", "both"):
        from pet.analysis import run_all_pet
        print(f"[batch] processing root/sub*/pet under {args.batch_root}")
        run_all_pet(args.batch_root, epochs=args.epochs, device=args.device, method=args.method,
                    windowed=args.windowed, axis=args.axis, slice_window=args.slice_window,
                    window_size=args.window_size, stride=args.stride)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    essential = parser.add_argument_group(
        "essential (this is really all most runs need)")
    essential.add_argument("--source", choices=["simulation", "real"], required=True,
                            help="'simulation': run on a built-in phantom (no real data needed). "
                                 "'real': run on real data (single subject via --path, or a whole "
                                 "folder of subjects via --batch-root). Which modality (DCE vs PET) "
                                 "is chosen separately via --modality in both cases.")
    essential.add_argument("--modality", choices=["dce", "pet", "both"], default="dce",
                            help="Which kinetic model family to use (1TCM/DCE-style vs 2TCM/PET-style). "
                                 "'both' is only valid with --source real --batch-root.")
    essential.add_argument("--method", choices=["pinn", "voxelwise"], default="pinn")
    essential.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    essential.add_argument("--epochs", type=int, default=1000)

    real_data = parser.add_argument_group("real data (--source real)")
    real_data.add_argument("--path", default=None,
                            help="Real subject directory, single subject (e.g. /data/sub01/dce). "
                                 "Mutually exclusive with --batch-root.")
    real_data.add_argument("--batch-root", default=None,
                            help="Root directory containing sub*/dce and/or sub*/pet subfolders -- "
                                 "processes every subject found, skipping any missing the relevant "
                                 "modality. Mutually exclusive with --path.")

    windowing = parser.add_argument_group(
        "windowed PINN training (advanced -- defaults are already sensible, "
        "only touch these if you know you need to)")
    windowing.add_argument("--windowed", action="store_true", default=True,
                            help="(--method pinn only, default True) train independent per-window "
                                 "PINNs instead of one whole-image fit. Pass --no-windowed for the "
                                 "older whole-image behavior.")
    windowing.add_argument("--no-windowed", dest="windowed", action="store_false")

    windowing.add_argument("--axis", choices=["xy", "z", "xyz"], default="xy",
                            help="'xy' (default): small in-plane patches, Z kept whole. 'z': "
                                 "slice-wise, full X,Y per window, split along Z instead. 'xyz': "
                                 "a genuine 3D box window, splitting all three spatial dimensions "
                                 "(requires --window-size as 3 values).")
    windowing.add_argument("--slice-window", type=int, default=1,
                            help="(--axis z only) Z-slices per window.")
    windowing.add_argument("--window-size", type=int, nargs="+", default=[16],
                            help="spatial window size: one value for a cube/square (--axis xy), "
                                 "two values X Y (--axis xy), or three values X Y Z (--axis xyz, "
                                 "required).")
    windowing.add_argument("--stride", type=int, nargs="+", default=None,
                            help="window stride (--axis xy/xyz only); same value-count rules as "
                                 "--window-size. Defaults to non-overlapping windows (stride=window-size).")

    sim_only = parser.add_argument_group("simulation only (--source simulation)")
    sim_only.add_argument("--seed", type=int, default=0)
    sim_only.add_argument("--shape", type=int, nargs=3, default=[32, 32, 8],
                           help="phantom shape, e.g. --shape 64 64 20")
    sim_only.add_argument("--noise-std", type=float, default=0.03)
    args = parser.parse_args()

    # --window-size / --stride come in as lists of ints (nargs="+");
    # Trainer expects a bare int for a square/cube, or a tuple for an
    # explicit (X,Y) / (X,Y,Z) size -- axis="xyz" requires the 3-tuple form.
    args.window_size = args.window_size[0] if len(args.window_size) == 1 else tuple(args.window_size)
    if args.stride is not None:
        args.stride = args.stride[0] if len(args.stride) == 1 else tuple(args.stride)

    if args.source == "simulation":
        if args.path is not None or args.batch_root is not None:
            raise SystemExit("--path/--batch-root only apply to --source real, not --source simulation")
        run_simulation(args)
        return

    # --source real
    if args.path is not None and args.batch_root is not None:
        raise SystemExit("--path and --batch-root are mutually exclusive -- use one or the other")
    if args.path is None and args.batch_root is None:
        raise SystemExit("--source real requires either --path (single subject) or --batch-root (a folder of subjects)")

    if args.batch_root is not None:
        run_batch(args)
    else:
        run_real_data(args)


if __name__ == "__main__":
    main()

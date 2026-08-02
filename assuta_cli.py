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
        affine = np.eye(4)
        trainer = Trainer(c_p=aif, num_of_compartment=num_of_compartment, t=t,
                           device=args.device, affine=affine, save_path=out_dir, epochs=args.epochs)
        ks_out, hist = trainer.train(img, z_slices=[0])
        for i, name in enumerate(param_names):
            r = pearsonr(ks_out[i][tissue], gt[name][tissue])[0]
            print(f"  {name}: corr={r:.3f}")
        print(f"  final loss: {hist['loss'][-1]:.3f}")

    elif args.method == "bayesian":
        from core.uncertainty import estimate_with_uncertainty
        result = estimate_with_uncertainty(
            img, aif, t, num_of_compartment=num_of_compartment,
            save_path=out_dir, affine=np.eye(4), device=args.device,
            n_ensemble=args.n_ensemble, epochs=args.epochs, dropout_p=args.dropout_p,
            tissue_mask=tissue,
        )
        for i, name in enumerate(param_names):
            r = pearsonr(result["K_mean"][i][tissue], gt[name][tissue])[0]
            unc_mean = result["K_uncertainty_demeaned"][i][tissue].mean()
            print(f"  {name}: corr={r:.3f}  mean per-voxel uncertainty={unc_mean:.4g}")

    else:
        raise ValueError(f"--method {args.method!r} not supported for --source simulation "
                          f"(use 'voxelwise', 'pinn', or 'bayesian')")

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
                       n_ensemble=args.n_ensemble, dropout_p=args.dropout_p)
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
                    n_ensemble=args.n_ensemble, dropout_p=args.dropout_p)

    if args.modality in ("pet", "both"):
        from pet.analysis import run_all_pet
        print(f"[batch] processing root/sub*/pet under {args.batch_root}")
        run_all_pet(args.batch_root, epochs=args.epochs, device=args.device, method=args.method,
                    n_ensemble=args.n_ensemble, dropout_p=args.dropout_p)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["simulation", "real"], required=True,
                         help="'simulation': run on a built-in phantom (no real data needed). "
                              "'real': run on real data (single subject via --path, or a whole "
                              "folder of subjects via --batch-root). Which modality (DCE vs PET) "
                              "is chosen separately via --modality in both cases.")
    parser.add_argument("--modality", choices=["dce", "pet", "both"], default="dce",
                         help="Which kinetic model family to use (1TCM/DCE-style vs 2TCM/PET-style). "
                              "'both' is only valid with --source real --batch-root.")
    parser.add_argument("--method", choices=["pinn", "voxelwise", "bayesian"], default="pinn",
                         help="'bayesian' additionally computes a per-voxel uncertainty map "
                              "(Sine B-PINN deep ensemble) -- see --n-ensemble/--dropout-p.")
    parser.add_argument("--path", default=None,
                         help="Real subject directory, single subject (e.g. /data/sub01/dce). "
                              "Mutually exclusive with --batch-root. (--source real only)")
    parser.add_argument("--batch-root", default=None,
                         help="Root directory containing sub*/dce and/or sub*/pet subfolders -- "
                              "processes every subject found, skipping any missing the relevant "
                              "modality. Mutually exclusive with --path. (--source real only)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                         help="Used consistently across simulation, single-subject, and batch modes "
                              "(default cpu).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-ensemble", type=int, default=5,
                         help="(--method bayesian only) number of B-PINN ensemble members. "
                              "Runtime scales ~linearly with this.")
    parser.add_argument("--dropout-p", type=float, default=0.1,
                         help="(--method bayesian only) MC-dropout probability in the B-PINN.")
    parser.add_argument("--seed", type=int, default=0, help="(--source simulation only)")
    parser.add_argument("--shape", type=int, nargs=3, default=[32, 32, 8],
                         help="(--source simulation only) phantom shape, e.g. --shape 64 64 20")
    parser.add_argument("--noise-std", type=float, default=0.03, help="(--source simulation only)")
    args = parser.parse_args()

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

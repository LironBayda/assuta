import os
import SimpleITK as sitk
import numpy as np
from typing import Tuple


def _preprocess_clip(img: sitk.Image, upper_percentile: float = 98) -> sitk.Image:
    """Clip high-intensity outliers and normalize image to [0,1]."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    hi = np.percentile(arr, upper_percentile)
    arr = np.clip(arr, 0, hi) / (hi + 1e-6)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def _extract_n_frame(img4d: sitk.Image, i: int) -> sitk.Image:
    """Extract 3D frame i from a 4D image."""
    size = img4d.GetSize()  # (X,Y,Z,T)
    extract = sitk.ExtractImageFilter()
    extract.SetSize([size[0], size[1], size[2], 0])
    extract.SetIndex([0, 0, 0, i])
    frame3d = extract.Execute(img4d)
    return sitk.Cast(frame3d, sitk.sitkFloat32)


def _find_file(folder: str, pattern: str) -> str :
    """Find first file in folder containing the pattern."""
    files = [f for f in os.listdir(folder) if pattern.lower() in f.lower()]
    return os.path.join(folder, files[0]) if files else None


def _save_image(img: sitk.Image, out_path: str) -> None:
    """Write image to disk and print confirmation."""
    sitk.WriteImage(img, out_path)
    print(f"Saved image → {out_path}")


# ---------------- Registration Functions ----------------
def _rigid_registration(moving: sitk.Image, reference: sitk.Image) -> sitk.Transform:
    """Rigid 3D registration of moving → reference."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.3)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=0.0001, minStep=1e-4,
                                                 numberOfIterations=10, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInterpolator(sitk.sitkLinear)

    initial_rigid = sitk.CenteredTransformInitializer(reference, moving,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg.SetInitialTransform(initial_rigid, inPlace=False)

    final_rigid = reg.Execute(reference, moving)
    print("Rigid registration done.")
    return final_rigid


def _affine_registration(moving: sitk.Image, reference: sitk.Image, rigid_tx: sitk.Transform) -> Tuple[sitk.Transform, sitk.Image, sitk.Image]:
    """Affine registration after rigid alignment. Returns composite transform and resampled images."""
    # Resample moving image with rigid transform first
    moving_rigid = sitk.Resample(moving, reference, rigid_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    moving_rigid = sitk.Cast(moving_rigid, sitk.sitkFloat32)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.6)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=0.0001, minStep=1e-4, numberOfIterations=10, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()

    affine_init = sitk.AffineTransform(3)
    try:
        base_rigid = rigid_tx.GetTransform(0) if hasattr(rigid_tx, "GetTransform") else rigid_tx
        affine_init.SetMatrix(base_rigid.GetMatrix())
        affine_init.SetTranslation(base_rigid.GetTranslation())
        if hasattr(base_rigid, "GetCenter"):
            affine_init.SetCenter(base_rigid.GetCenter())
    except Exception:
        pass

    reg.SetInitialTransform(affine_init, inPlace=False)
    affine_opt = reg.Execute(reference, moving_rigid)
    print("Affine registration done.")

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(rigid_tx)
    composite.AddTransform(affine_opt)

    moving_affine = sitk.Resample(moving_rigid, reference, affine_opt, sitk.sitkLinear, 0.0, moving.GetPixelID())
    return composite, moving_rigid, moving_affine


def _bspline_registration(moving_affine: sitk.Image, reference: sitk.Image, affine_tx: sitk.Transform) -> sitk.Transform:
    """Gentle BSpline deformable registration after affine alignment."""
    grid_spacing = [60.0, 60.0, 30.0]
    img_size = reference.GetSize()
    img_spacing = reference.GetSpacing()
    mesh_size = [max(1, round((img_size[i] * img_spacing[i]) / grid_spacing[i])) for i in range(3)]

    bspline_init = sitk.BSplineTransformInitializer(reference, mesh_size, order=1)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=4)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-4, numberOfIterations=1)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([3, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(bspline_init, inPlace=False)

    deform_tx = reg.Execute(reference, moving_affine)
    print("BSpline registration done.")

    return sitk.Transform(deform_tx)


# ---------------- Transform application ----------------
def _apply_transform_3d_or_4d(moving_img: sitk.Image, final_tx: sitk.Transform, reference_img: sitk.Image, crop_margin_ratio=1/3) -> sitk.Image:
    """Apply transform to 3D or 4D moving image."""
    if moving_img.GetDimension() == 3:
        return sitk.Resample(moving_img, reference_img, final_tx, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
    else:
        registered_frames = []
        for t in range(moving_img.GetSize()[3]):  # T = number of time frames
            frame = _extract_n_frame(moving_img, t)  # 3D
            reg_frame = sitk.Resample(frame, reference_img, final_tx)  # same 3D transform
            registered_frames.append(reg_frame)

        registered_4d = sitk.JoinSeries(registered_frames)

        return registered_4d




def _compute_transform(moving: sitk.Image, reference: sitk.Image, run_affine=True, run_bspline=True) -> sitk.Transform:
    rigid_tx = _rigid_registration(moving, reference)
    if run_affine:
        composite, _, moving_affine = _affine_registration(moving, reference, rigid_tx)
        current_tx = composite
    else:
        moving_affine = sitk.Resample(moving, reference, rigid_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
        current_tx = rigid_tx
    if run_bspline:
        final_tx = _bspline_registration(moving_affine, reference, current_tx)
    else:
        final_tx = current_tx
    return final_tx

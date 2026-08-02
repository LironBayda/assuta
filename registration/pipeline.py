# ---------------- High-level registration pipeline ----------------
import numpy as np

from registration.utils import _extract_n_frame, _compute_transform, _apply_transform_3d_or_4d
import SimpleITK as sitk

def _get_3d_reg_img(moving_img: sitk.Image, reference_last: sitk.Image,
                              run_affine, run_bspline, crop_margin_ratio):
    moving_img_3d = sitk.Cast(moving_img, sitk.sitkFloat32)
    reference_last_3d = sitk.Cast(reference_last, sitk.sitkFloat32)

    final_tx = _compute_transform(moving_img_3d, reference_last_3d, run_affine, run_bspline)
    img_reg = _apply_transform_3d_or_4d(moving_img_3d, final_tx, reference_last_3d, crop_margin_ratio)
    return img_reg
def _reg_all_frame_to_reference(moving_img: sitk.Image, reference_last: sitk.Image,
                              run_affine, run_bspline, crop_margin_ratio):
    n_frames_mov = moving_img.GetSize()[3]

    reg_frames = []
    for i in range(n_frames_mov):
        frame = _extract_n_frame(moving_img, i)
        reg_frame= _get_3d_reg_img(frame, reference_last,run_affine, run_bspline, crop_margin_ratio)
        reg_frames.append(reg_frame)
    img_reg = sitk.JoinSeries(reg_frames)
    return img_reg



def moving_to_reference_full(moving_path: str, reference_path: str,
                             reg_all=True, run_affine=True, run_bspline=True, crop_margin_ratio=1/3) -> np.ndarray:
    """Register moving image (3D or 4D) to reference image last frame."""
    moving_img = sitk.ReadImage(moving_path)
    reference_img = sitk.ReadImage(reference_path)

    n_frames = reference_img.GetSize()[3] if reference_img.GetDimension() == 4 else 1
    reference_last = _extract_n_frame(reference_img, n_frames-1) if n_frames > 1 else reference_img
    reference_last = sitk.Cast(reference_last, sitk.sitkFloat32)

    if moving_img.GetDimension() == 3:
        # Ensure float32
        img_reg=_get_3d_reg_img(moving_img, reference_last,run_affine , run_bspline, crop_margin_ratio)


    else:
        if reg_all:
            img_reg=_reg_all_frame_to_reference(moving_img, reference_last,run_affine, run_bspline, crop_margin_ratio)
            img_reg.CopyInformation(reference_img)

        else:
            moving_last = _extract_n_frame(moving_img, moving_img.GetSize()[3] - 1)
            print("moving_last.GetSize()")

            print(moving_last.GetSize())

            final_tx = _compute_transform(moving_last, reference_last, run_affine, run_bspline)

            img_reg=_apply_transform_3d_or_4d(moving_img, final_tx, reference_last, crop_margin_ratio)
            print(img_reg.GetSize())

    img_reg=sitk.GetArrayFromImage(img_reg)
    img_reg = np.transpose(img_reg, range(len(img_reg.shape)-1,-1,-1))

    return img_reg

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

import numpy as np
import SimpleITK as sitk


def mask_to_reference(
    moving_path: str,
    reference_path: str,
    mask_path: str,
    run_affine: bool = True,
    run_bspline: bool = True
) -> np.ndarray:
    """
    Register a mask from moving-image space to reference-image space.

    Registration image:
        Sum of moving frames 4, 5, 6, and 23.

    Reference image:
        Last frame of the reference image if it is 4D.

    The final transform is applied to the mask using nearest-neighbor
    interpolation.
    """

    # ---------------------------------------------------------
    # 1. Read images
    # ---------------------------------------------------------
    moving_img = sitk.ReadImage(moving_path)
    reference_img = sitk.ReadImage(reference_path)
    mask_img = sitk.ReadImage(mask_path)

    # ---------------------------------------------------------
    # 2. Get the last reference frame
    # ---------------------------------------------------------
    if reference_img.GetDimension() == 4:
        n_frames = reference_img.GetSize()[3]
        reference_last = _extract_n_frame(
            reference_img,
            n_frames - 1
        )
    else:
        reference_last = reference_img

    reference_last = sitk.Cast(
        reference_last,
        sitk.sitkFloat32
    )

    # ---------------------------------------------------------
    # 3. Create registration image from moving frames
    # ---------------------------------------------------------
    if moving_img.GetDimension() != 4:
        raise ValueError(
            "moving_img must be a 4D image because frames "
            "4, 5, 6 and 23 are used."
        )

    n_moving_frames = moving_img.GetSize()[3]

    required_frames = [4, 5, 6, 23]

    if max(required_frames) >= n_moving_frames:
        raise ValueError(
            f"Moving image has only {n_moving_frames} frames, "
            f"but frame 23 is required."
        )

    moving_last = (
        _extract_n_frame(moving_img, 4)
        + _extract_n_frame(moving_img, 5)
        + _extract_n_frame(moving_img, 6)
        + _extract_n_frame(moving_img, 23)
    )

    moving_last = sitk.Cast(
        moving_last,
        sitk.sitkFloat32
    )

    print("Moving registration image size:",
          moving_last.GetSize())

    print("Reference image size:",
          reference_last.GetSize())

    # ---------------------------------------------------------
    # 4. Compute registration transform
    # ---------------------------------------------------------
    final_tx = _compute_transform(
        moving_last,
        reference_last,
        run_affine,
        run_bspline
    )

    # ---------------------------------------------------------
    # 5. Apply transform to mask
    # ---------------------------------------------------------
    mask_img = sitk.Cast(
        mask_img,
        sitk.sitkInt32
    )

    # If the mask is 4D, this should normally be handled
    # separately. For a 3D mask, Resample directly.
    if mask_img.GetDimension() != 3:
        raise ValueError(
            "mask_img must be a 3D mask."
        )

    mask_registered = sitk.Resample(
        mask_img,
        reference_last,
        final_tx,
        sitk.sitkNearestNeighbor,
        0,
        mask_img.GetPixelID()
    )

    # ---------------------------------------------------------
    # 6. Convert to NumPy
    # ---------------------------------------------------------
    mask_registered = sitk.GetArrayFromImage(
        mask_registered
    )

    # SimpleITK returns:
    #     (z, y, x)
    #
    # Convert to:
    #     (x, y, z)
    mask_registered = np.transpose(
        mask_registered,
        (2, 1, 0)
    )

    return mask_registered



def moving_to_reference_full(moving_path: str, reference_path: str,
                             reg_all=True, run_affine=True, run_bspline=True, crop_margin_ratio=1/3) -> np.ndarray:
    """Register moving image (3D or 4D) to reference image last frame."""
    moving_img = sitk.ReadImage(moving_path)
    reference_img = sitk.ReadImage(reference_path)

    n_frames = reference_img.GetSize()[3] if reference_img.GetDimension() == 4 else 1
    reference_last = _extract_n_frame(reference_img, 0) if n_frames > 1 else reference_img
    reference_last = sitk.Cast(reference_last, sitk.sitkFloat32)

    if moving_img.GetDimension() == 3:
        # Ensure float32
        img_reg=_get_3d_reg_img(moving_img, reference_last,run_affine , run_bspline, crop_margin_ratio)


    else:
        if reg_all:
            img_reg=_reg_all_frame_to_reference(moving_img, reference_last,run_affine, run_bspline, crop_margin_ratio)
            img_reg.CopyInformation(reference_img)

        else:
            moving_last =_extract_n_frame(moving_img, 4)+_extract_n_frame(moving_img, 5)+_extract_n_frame(moving_img, 6)+_extract_n_frame(moving_img, 23)
            print("moving_last.GetSize()")

            print(moving_last.GetSize())

            final_tx = _compute_transform(moving_last, reference_last, run_affine, run_bspline)

            img_reg=_apply_transform_3d_or_4d(moving_img, final_tx, reference_last, crop_margin_ratio)
            print(img_reg.GetSize())

    img_reg=sitk.GetArrayFromImage(img_reg)
    img_reg = np.transpose(img_reg, range(len(img_reg.shape)-1,-1,-1))

    return img_reg

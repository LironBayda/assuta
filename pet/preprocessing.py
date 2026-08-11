import os.path

import numpy as np
import nibabel as nib
from scipy.signal import fftconvolve
from typing import Tuple, Optional

from config import PET
from core.create_blood_mask import make_blood_mask
from core.utils import find_files, get_tac_from_masked_region
from io_utils.jepg_utils import save_single_series_plot
from io_utils.nifti_utils import load_image_series
from registration import moving_to_reference_full


class STCPVC:
    """
    Geometric Single-Target Correction (STC) for Dynamic PET
    ---------------------------------------------------------
    Implements:
      a^{k+1}(x) = ( b(x) - g_k(x) ) / R(x)

    where:
      g_k(x) = I(x) * conv(a_out, h) + (1 - I(x)) * conv(a_in, h)
      R(x)   = I(x) * conv(I, h) + (1 - I(x)) * (1 - conv(I, h))
    """

    # -----------------------------
    # Init
    # -----------------------------
    def __init__(
        self,
        mask: np.ndarray,
        voxel_size: Tuple[float, float, float],
        psf_kernel: Optional[np.ndarray] = None,
        fwhm_mm: Optional[float] = None,
        eps: float = 1e-6,
    ):
        assert mask.ndim == 3, "mask must be 3D"
        self.mask = mask.astype(np.float32)
        self.I = self.mask
        self.one_minus_I = 1.0 - self.I
        self.voxel_size = voxel_size
        self.eps = eps

        if psf_kernel is None:
            assert fwhm_mm is not None, "Provide psf_kernel or fwhm_mm"
            self.psf_kernel = self._build_gaussian_psf(fwhm_mm)
        else:
            self.psf_kernel = psf_kernel

        self.R_map, self.conv_I = self._compute_R_map()

    # -----------------------------
    # PSF
    # -----------------------------
    def _build_gaussian_psf(self, fwhm_mm: float) -> np.ndarray:
        sx = fwhm_mm / (2.3548 * self.voxel_size[0])
        sy = fwhm_mm / (2.3548 * self.voxel_size[1])
        sz = fwhm_mm / (2.3548 * self.voxel_size[2])
        sigma = np.array([sx, sy, sz])

        radius = int(np.ceil(3 * sigma.max()))
        gx = np.arange(-radius, radius + 1)
        X, Y, Z = np.meshgrid(gx, gx, gx, indexing="ij")

        kern = np.exp(
            -0.5
            * (
                (X / sigma[0]) ** 2
                + (Y / sigma[1]) ** 2
                + (Z / sigma[2]) ** 2
            )
        )
        kern /= kern.sum()
        return kern

    # -----------------------------
    # Convolution
    # -----------------------------
    def _convolve(self, image: np.ndarray) -> np.ndarray:
        return fftconvolve(image, self.psf_kernel, mode="same")

    # -----------------------------
    # R-map
    # -----------------------------
    def _compute_R_map(self):
        conv_I = self._convolve(self.I)
        R = self.I * conv_I + self.one_minus_I * (1.0 - conv_I) # (1.0 - conv_I) IS LOGICAL NOT
        R = np.maximum(R, self.eps)
        return R, conv_I

    # -----------------------------
    # Single Frame STC
    # -----------------------------
    def _stc_single_frame(self, b_frame: np.ndarray, n_iter: int) -> np.ndarray:
        a = b_frame.copy()

        for _ in range(n_iter):
            a_in = a * self.I
            a_out = a * self.one_minus_I

            conv_in = self._convolve(a_in)
            conv_out = self._convolve(a_out)

            gk = self.I * conv_out + self.one_minus_I * conv_in
            a = (b_frame - gk) / self.R_map
            a = np.maximum(a, 0.0)

        return a

    # -----------------------------
    # Dynamic STC
    # -----------------------------
    def apply_dynamic(
        self,
        pet_dyn: np.ndarray,
        n_iter: int = 8,
        save_path: Optional[str] = None,
        affine: Optional[np.ndarray] = None,
    ):
        """
        pet_dyn: (T, X, Y, Z)
        """
        assert pet_dyn.ndim == 4, "pet_dyn must be (T, X, Y, Z)"

        T = pet_dyn.shape[0]
        corrected = np.zeros_like(pet_dyn)

        for t in range(T):
            corrected[t] = self._stc_single_frame(pet_dyn[t], n_iter)

        if save_path is not None:
            assert affine is not None, "Affine required to save NIfTI"
            img = nib.Nifti1Image(corrected.transpose(1,2,3,0).astype(np.float32), affine)
            nib.save(img, save_path)

        return corrected


def load_dynamic_pet(dyn_path):
    """
    Load dynamic PET NIfTI and extract:
    - image data as float32
    - affine
    - voxel size (dx, dy, dz) in mm
    - enforce (T, X, Y, Z) layout
    """

    img_nii = nib.load(dyn_path)
    affine = img_nii.affine
    hdr = img_nii.header
    img = img_nii.get_fdata().astype(np.float32)

    # ----------------------------
    # Voxel size in mm
    # ----------------------------
    voxel_size = hdr.get_zooms()[:3]   # (dx, dy, dz)

    return img, affine, voxel_size




def preprocessing(path: str, dyn_pattern: str = "*_pet*",dce_pattern: str = "*dyns*") -> tuple:
    """
    This function performs:
    1. Loading of the dynamic PET image series.
    2. Blood mask extraction from the PET image.
    3. Partial volume correction using Single-Timepoint Correction (STC) PVC.
    4. Extraction of the arterial input function (AIF) from the corrected PET image.
    5. Plotting of the AIF.

    Parameters
    ----------
    path : str
        Path to the patient/session folder containing the dynamic PET images.
    dyn_pattern : str, optional
        Filename pattern to identify dynamic PET NIfTI files (default "*dyns*").

    Returns
    -------
    pet_corr : np.ndarray
        4D PET image after STC partial volume correction. Shape: (T, X, Y, Z)
    aif : np.ndarray
        Extracted arterial input function (1D array, length = number of frames)
    affine : np.ndarray
        4x4 affine matrix for spatial registration of the PET images.

    Raises
    ------
    FileNotFoundError
        If no dynamic PET image matches the specified pattern.
    ValueError
        If loaded images have unexpected dimensions.
    """

    # ----------------------------
    # 1. Load dynamic PET image
    # ----------------------------
    print(f"[INFO] Preprocessing PET data in folder: {path}")
    dyn_files = find_files(path, dyn_pattern)
    if not dyn_files:
        raise FileNotFoundError(f"No dynamic PET images found using pattern '{dyn_pattern}' in {path}")
    dyn_path = dyn_files[0]
    print(f"[INFO] Found dynamic PET file: {dyn_path}")

    # Load PET data and extract voxel size

    # ----------------------------
    # 2. Blood mask extraction
    # ----------------------------
    print("[INFO] Extracting blood mask from PET image...")
    blood_mask = nib.load(os.path.join(path.replace('/pet','/dce'),'blood.nii'))
    blood_mask=blood_mask.get_fdata()

    print(f"[INFO] Blood mask created with shape {blood_mask.shape}")
    dyn_path   = find_files(path, dyn_pattern)[0]
    dce_path   = find_files(os.path.join(path.replace('/pet','/dce')), dce_pattern)[0]

    img=moving_to_reference_full(dyn_path, dce_path,
    reg_all = False, run_affine = True, run_bspline = True, crop_margin_ratio = 5/ 11)
    _, affine, voxel_size = load_dynamic_pet(dce_path)

    # ----------------------------
    # 3. STC Partial Volume Correction
    # ----------------------------
    print("[INFO] Applying STC partial volume correction...")
    stc = STCPVC(
        mask=blood_mask,
        voxel_size=voxel_size,
        fwhm_mm=4.8  # Siemens Biograph PET/MR OSEM FWHM
    )
    pet_corr = stc.apply_dynamic(
        img.transpose(3,0,1,2),
        n_iter=10,
        save_path=f"{path}/pet_stc_corrected.nii.gz",
        affine=affine
    ).transpose(1,2,3,0)
    print("[INFO] STC PVC completed and saved as 'pet_stc_corrected.nii.gz'")

    # ----------------------------
    # 4. AIF extraction
    # ----------------------------
    print("[INFO] Extracting arterial input function (AIF)...")
    aif = get_tac_from_masked_region(pet_corr, blood_mask)*1.62
    print(f"[INFO] Extracted AIF with {len(aif)} time points")

    # Plotting for verification
    save_single_series_plot(
        path,
        PET["dt"],

        aif,
        "DCE AIF",
        "Time (min)",
        "Concentration (mM)"
    )
    print("[INFO] AIF plot saved.")

    img, affine, voxel_size = load_dynamic_pet(dce_path)
    # ----------------------------
    # 5. Return results
    # ----------------------------
    return pet_corr, aif, affine

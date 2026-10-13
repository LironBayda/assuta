from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi


class HyprLr:
    """
    Dynamic PET denoising via HYPR-LR (HighlYconstrained backPRojection,
    Local Restriction).

    Reference: Improved kinetic analysis of dynamic PET data with
    optimized HYPR-LR.
    https://jnm.snmjournals.org/content/jnumed/51/7/1147.full.pdf
    """

    def __init__(self, images, affine, frame_durations, fwhm_mm=16.0, voxel_sizes=None):
        """
        images : ndarray (n_frames, X, Y, Z) -- dynamic PET series, any
            number of frames and any spatial size.
        affine : ndarray (4, 4) -- NIfTI affine shared by every frame.
        frame_durations : sequence of length n_frames -- acquisition
            duration of each frame (any consistent unit), used to
            time-weight frames into the composite image i_c. Required --
            no protocol-specific default is assumed.
        fwhm_mm : float -- full width at half maximum (mm) of the boxcar
            low-pass filter. Default 9mm is a common PET reconstruction
            FWHM; override for your scanner/reconstruction.
        voxel_sizes : optional ndarray (3,) voxel size in mm per axis.
            Derived from `affine` if omitted.
        """
        images = np.asarray(images, dtype=float)
        if images.ndim != 4:
            raise ValueError(f"images must be (n_frames, X, Y, Z); got shape {images.shape}")

        n_frames = images.shape[0]
        frame_durations = np.asarray(frame_durations, dtype=float)
        if frame_durations.shape != (n_frames,):
            raise ValueError(
                f"frame_durations has {frame_durations.shape[0]} entries but "
                f"images has {n_frames} frames"
            )

        images = np.nan_to_num(images, nan=0.0)
        images[images < 0] = 0

        self.images = images
        self.affine = np.asarray(affine, dtype=float)
        self.frame_durations = frame_durations
        self.fwhm_mm = fwhm_mm
        # mm/voxel along each spatial axis, from the affine's rotation/scale
        # block -- replaces the old fixed 256-vs-other kernel-size lookup so
        # the filter adapts to whatever resolution the image actually has.
        self.voxel_sizes = (
            np.asarray(voxel_sizes, dtype=float)
            if voxel_sizes is not None
            else np.linalg.norm(self.affine[:3, :3], axis=0)
        )
        self.n_frames = n_frames
        self.spatial_shape = images.shape[1:]
        self.results = np.empty_like(images)

    def _kernel_size_voxels(self):
        """FWHM (mm) -> odd kernel size (voxels) per spatial axis, from this
        image's actual voxel size. No hardcoded image-size branch."""
        size = np.round(self.fwhm_mm / self.voxel_sizes).astype(int)
        size = np.maximum(size, 1)
        size += size % 2 == 0  # force odd so the kernel has a center voxel
        return tuple(int(s) for s in size)

    def _boxcar_kernel(self):
        size = self._kernel_size_voxels()
        kernel = np.ones(size, dtype=float)
        return kernel / kernel.sum()

    @staticmethod
    def _composite_image(images, durations):
        """Time-weighted sum of a block of frames (i_c in the HYPR-LR
        paper). Works for any block length or spatial shape."""
        weights = durations.reshape(-1, *([1] * (images.ndim - 1)))
        return np.sum(images * weights, axis=0)

    def _process_block(self, images, durations):
        """
        HYPR-LR for one contiguous block of frames sharing a composite
        image i_c: i_h = i_c * (F*I) / (F*i_c) for every frame in the
        block. F is convolution with the boxcar kernel.
        """
        i_c = self._composite_image(images, durations)
        kernel = self._boxcar_kernel()
        denominator = ndi.convolve(i_c, kernel, mode="reflect")
        denominator[denominator == 0] = np.finfo(float).eps  # avoid /0 outside the object

        block_result = np.empty_like(images)
        for i in range(images.shape[0]):
            numerator = ndi.convolve(images[i], kernel, mode="reflect")
            i_h = i_c * (numerator / denominator)
            block_result[i] = np.nan_to_num(i_h, nan=0.0)
        return block_result

    def create_hypr_lr_images(self, block_boundaries=None):
        """
        Run HYPR-LR over the whole dynamic series.

        block_boundaries : optional list of (start, end) frame-index pairs
            (end exclusive) -- e.g. an uptake phase, a retention phase, an
            equilibrium phase, each with its own composite image. If
            omitted, every frame shares one composite image built from the
            whole series -- a safe default for any protocol/frame count.
        """
        if block_boundaries is None:
            block_boundaries = [(0, self.n_frames)]

        for start, end in block_boundaries:
            if not (0 <= start < end <= self.n_frames):
                raise ValueError(f"invalid block boundary ({start}, {end}) for {self.n_frames} frames")
            self.results[start:end] = self._process_block(
                self.images[start:end], self.frame_durations[start:end]
            )

        return self.results

    def save(self, out_path):
        """Save the denoised series as a single 4D NIfTI (X, Y, Z, T)."""
        img_nii = nib.Nifti1Image(np.moveaxis(self.results, 0, -1).astype(np.float64), self.affine)
        nib.save(img_nii, str(out_path))
        print(f"[SAVED] {out_path}")

    @classmethod
    def from_nifti_dir(cls, directory, nii_prefix, frame_durations, **kwargs):
        """
        Load every NIfTI file in `directory` whose name starts with
        `nii_prefix` (sorted by filename) as one dynamic-PET frame each.
        Any number of files/frame size is supported -- no hardcoded frame
        count or fixed index into the file list.
        """
        directory = Path(directory)
        paths = sorted(p for p in directory.iterdir() if p.name.startswith(nii_prefix))
        if not paths:
            raise FileNotFoundError(f"no files starting with '{nii_prefix}' found in {directory}")

        volumes = [nib.load(str(p)) for p in paths]
        images = np.stack([v.get_fdata() for v in volumes], axis=0)
        affine = volumes[-1].affine  # every frame shares one grid; any one's affine is fine
        return cls(images, affine, frame_durations, **kwargs)
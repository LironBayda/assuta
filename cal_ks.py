import glob
import os
import nibabel as nib
import numpy as np
import pandas as pd

from core.utils import center_crop_nifti


def run_all(root_path, epochs=1000, device="cpu", agg="median"):
    """
    agg : "median" (default, robust to the occasional degenerate-voxel
        fit -- see the note below) or "mean" (matches old behavior,
        NOT recommended: a single bad voxel can dominate a mean).
    """
    agg_fn = np.nanmedian if agg == "median" else np.nanmean

    data={}
    subject_paths = sorted(
        glob.glob(os.path.join(root_path, "sub*"))
    )

    for idx, sub_path in enumerate(subject_paths, start=1): #28

        subject_id = os.path.basename(sub_path)
        #if 'sub20' == subject_id or 'sub215' == subject_id  or 'sub210' == subject_id : 222 210 12 108
        #        or 'sub405' == subject_id  or 'sub512' == subject_id  or 'sub238' == subject_id \
         #       or 'sub36' == subject_id  or 'sub401' == subject_id or 'sub215' == subject_id:
         #   continue
        dce_path = os.path.join(sub_path,"dce")
        pet_path = os.path.join(sub_path,"pet")


        print(f"Processing {subject_id}")

        if not os.path.exists(os.path.join(sub_path,"dce","rcancer_dixon.nii")):
            continue



        rpcancer_dixon = os.path.join(
            pet_path,
            'pet_lesion.nii'
        )

        rpnotcancer_dixon = os.path.join(
            pet_path,
            'pet_ref.nii'
        )


        mask_c = nib.load(rpcancer_dixon).get_fdata()
        mask_nc = nib.load(rpnotcancer_dixon).get_fdata()

        pet_rpcancer_dixon = os.path.join(
            pet_path,
            'lesion.nii'
        )

        pet_rpnotcancer_dixon = os.path.join(
        pet_path,
            'ref.nii'
        )


        mask_c = nib.load(rpcancer_dixon).get_fdata()
        mask_nc = nib.load(rpnotcancer_dixon).get_fdata()
        affine = nib.load(rpnotcancer_dixon).affine
        pet_mask_c = nib.load(pet_rpcancer_dixon).get_fdata()
        pet_mask_nc = nib.load(pet_rpnotcancer_dixon).get_fdata()




        DCE_K1 = nib.load(os.path.join(sub_path,"dce","K1_map.nii.gz")).get_fdata()

        PET_K = nib.load(os.path.join(sub_path,"pet","K1_map.nii.gz")).get_fdata()


        pet_mask_c,_=center_crop_nifti(pet_mask_c,affine,target_shape=(64, 64, 32))
        pet_mask_nc,_=center_crop_nifti(pet_mask_nc,affine,target_shape=(64, 64, 32))

        mask_c,_=center_crop_nifti(mask_c,affine,target_shape=(64, 64, 20))
        mask_nc,_=center_crop_nifti(mask_nc,affine,target_shape=(64, 64, 20))


        data[subject_id] = {

            'not_cancer_K1':
                PET_K[:, :, :][pet_mask_nc > 0].mean(),

            'cancer_K1':
                PET_K[:, :, :][pet_mask_c> 0].mean(),

        #    'not_cancer_K2':
         #       PET_K[:, :, :, 1][pet_mask_nc > 0].mean(),

          #  'cancer_K2':
           #     PET_K[:, :, :, 1][pet_mask_c > 0].mean(),

          #  'not_cancer_K3':
           #     PET_K[:, :, :, 2][pet_mask_nc[cx:-cx, cy:-cy, :] > 0].mean(),

           # 'cancer_K3':
            #   PET_K[:, :, :, 2][pet_mask_c[cx:-cx, cy:-cy, :] > 0].mean(),

            'not_cancer_Ktrans':
                agg_fn(DCE_K1[:, :, :][mask_nc > 0]),

            'cancer_Ktrans':
                agg_fn(DCE_K1[:, :, :][mask_c > 0]),

        #    'not_cancer_Kkep':
         #       agg_fn(DCE_K1[:, :, :,1][mask_nc > 0]),

          #  'cancer_Kkep':
           #     agg_fn(DCE_K1[:, :, :,1][mask_c> 0])

        }

        df = pd.DataFrame(data).T
        print(df)

        df.to_csv(
            '/home/liron/Documents/dce_pet/data.csv'
        )


    print("[INFO] Finished")



path = "/home/liron/Documents/dce_pet/lesion/"
run_all(path)

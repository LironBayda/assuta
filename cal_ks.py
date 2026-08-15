import glob
import os
import nibabel as nib
import numpy as np
import pandas as pd


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

    for idx, sub_path in enumerate(subject_paths[:14], start=1):

        subject_id = os.path.basename(sub_path)

        dce_path = os.path.join(sub_path,"dce")
        pet_path = os.path.join(sub_path,"pet")


        print(f"Processing {subject_id}")

        if not os.path.exists(os.path.join(sub_path,"dce")):
            continue



        rpcancer_dixon = os.path.join(
            dce_path,
            'cancer.nii'
        )

        rpnotcancer_dixon = os.path.join(
            dce_path,
            'ref.nii'
        )


        mask_c = nib.load(rpcancer_dixon).get_fdata()
        mask_nc = nib.load(rpnotcancer_dixon).get_fdata()

        pet_rpcancer_dixon = os.path.join(
            dce_path,
            'cancer.nii'
        )

        pet_rpnotcancer_dixon = os.path.join(
            dce_path,
            'ref.nii'
        )


        mask_c = nib.load(rpcancer_dixon).get_fdata()
        mask_nc = nib.load(rpnotcancer_dixon).get_fdata()

        pet_mask_c = nib.load(pet_rpcancer_dixon).get_fdata()
        pet_mask_nc = nib.load(pet_rpnotcancer_dixon).get_fdata()




        DCE_K1 = nib.load(os.path.join(sub_path,"dce","K_mean_windowed.nii")).get_fdata()

        PET_K = nib.load(os.path.join(sub_path,"pet","K_mean_windowed.nii")).get_fdata()

        # Compute crop margins as one-third of each dimension
        cx = pet_mask_c.shape[0] // 3
        cy = pet_mask_c.shape[1] // 3
        cz = pet_mask_c.shape[2] // 3

        pet_mask_c=pet_mask_c#[cx: -cx, cy: -cy,:]
        pet_mask_nc=pet_mask_nc#[cx: -cx, cy: -cy,:]

        cxd = mask_c.shape[0] // 3
        cyd = mask_c.shape[1] // 3
        mask_c=mask_c#[cxd: -cxd, cyd: -cyd,:]
        mask_nc=mask_nc#[cxd: -cxd, cyd: -cyd,:]


        data[subject_id] = {

            'not_cancer_K1':
                PET_K[:, :, :, 0][pet_mask_nc > 0].mean(),

            'cancer_K1':
                PET_K[:, :, :, 0][pet_mask_c> 0].mean(),

            'not_cancer_K2':
                PET_K[:, :, :, 1][pet_mask_nc > 0].mean(),

            'cancer_K2':
                PET_K[:, :, :, 1][pet_mask_c > 0].mean(),

          #  'not_cancer_K3':
           #     PET_K[:, :, :, 2][pet_mask_nc[cx:-cx, cy:-cy, :] > 0].mean(),

           # 'cancer_K3':
            #   PET_K[:, :, :, 2][pet_mask_c[cx:-cx, cy:-cy, :] > 0].mean(),

            'not_cancer_Ktrans':
                agg_fn(DCE_K1[:, :, :,0][mask_nc > 0]),

            'cancer_Ktrans':
                agg_fn(DCE_K1[:, :, :,0][mask_c > 0]),

            'not_cancer_Kkep':
                agg_fn(DCE_K1[:, :, :,1][mask_nc > 0]),

            'cancer_Kkep':
                agg_fn(DCE_K1[:, :, :,1][mask_c> 0])

        }

        df = pd.DataFrame(data).T
        print(df)

        df.to_csv(
            '/home/liron/Documents/dce_pet/data.csv'
        )


    print("[INFO] Finished")



path = "/home/liron/Documents/dce_pet/lesion/"
run_all(path)

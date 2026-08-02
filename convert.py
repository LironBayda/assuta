import shutil
from pathlib import Path

src_root = Path("/media/liron/Transcend/copy")
dst_root = Path("/home/liron/Documents/dce_pet/lesion")

dst_root.mkdir(parents=True, exist_ok=True)

for subject_dir in src_root.iterdir():

    if not subject_dir.is_dir():
        continue

    # find dynPET folder regardless of capitalization
    pet_dir = None
    for d in subject_dir.iterdir():
        if d.is_dir() and d.name.lower() == "dynpet":
            pet_dir = d
            break

    if pet_dir is None:
        print(f"No dynPET found in {subject_dir.name}")
        continue

    # create output subject/dynPET folder
    dst_pet_dir = dst_root / subject_dir.name / "pet"
    dst_pet_dir.mkdir(parents=True, exist_ok=True)

    # copy files
    for file in pet_dir.iterdir():
        if 'cancer_dixon.nii' in file.name:
            if dst_pet_dir.exists():

                shutil.copy2(
                    file,
                    dst_pet_dir / file.name
                )

            print(f"Copied {file.name} from {subject_dir.name}")
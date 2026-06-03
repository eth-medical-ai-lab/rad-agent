"""
This script converts the RadChestCT dataset from .npz format to NIfTI (.nii.gz) format.
It reads the metadata from a CSV file to get the list of scan IDs, loads each corresponding .npz file, and saves it as a NIfTI file in the specified output directory.
Also ensure XY orientation is the same as CT Rate NIFTI files.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm


df = pd.read_csv(
    "<RADCHEST_CT_DATASET_DIR>/CT_Scan_Metadata_Complete_35747.csv",
    header=0,
    index_col=0,
)

all_ids = df.index.tolist()

output_dir = Path(
    "<RADCHEST_CT_DATASET_DIR>/NIFTI"
)
output_dir.mkdir(parents=True, exist_ok=True)

n_files = 0
for i, id in tqdm(enumerate(all_ids)):
    if (output_dir / f"{id}.nii.gz").exists():
        n_files += 1
        continue
    input_path = (
        f"<RADCHEST_CT_DATASET_DIR>/{id}.npz"
    )
    if not Path(input_path).exists():
        continue
    data_array = np.load(
        f"<RADCHEST_CT_DATASET_DIR>/{id}.npz"
    )["ct"]
    out_spacing = [0.8, 0.8, 0.8]  # Example spacing in mm for (z, x, y)
    data_ready = np.transpose(data_array, (1, 2, 0))  # Reorder to [X, Y, Z]
    data_ready = np.transpose(data_ready, (1, 0, 2))
    affine = np.diag([0.8, 0.8, 0.8, 1.0])  # Create affine matrix
    nifti_img = nib.Nifti1Image(data_ready, affine)  # Create NIfTI image
    nib.save(nifti_img, output_dir / f"{id}.nii.gz")  # Save the NIfTI file
    n_files += 1

print(f"Processed {n_files} files and saved to {output_dir}")

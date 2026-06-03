import argparse
import ast
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

from pathlib import Path
from tqdm import tqdm

# import multiprocessing as mp


def process_row(row):

    # Set up directory parameters
    VolumeName = row["VolumeName"]
    dir1 = VolumeName.rsplit("_", 1)[0]
    dir2 = VolumeName.rsplit("_", 2)[0]
    filepath = os.path.join(data_root, "dataset/train", dir2, dir1, VolumeName)

    # Read Image
    image = sitk.ReadImage(filepath)

    # Set Spacing
    (x, y), z = map(float, ast.literal_eval(row["XYSpacing"])), row["ZSpacing"]
    image.SetSpacing((x, y, z))

    # Set Origin
    image.SetOrigin(ast.literal_eval(row["ImagePositionPatient"]))

    # Set Direction
    orientation = ast.literal_eval(row["ImageOrientationPatient"])
    row_cosine, col_cosine = orientation[:3], orientation[3:6]
    z_cosine = np.cross(row_cosine, col_cosine).tolist()
    image.SetDirection(row_cosine + col_cosine + z_cosine)

    # Fix Rescale
    RescaleIntercept = row["RescaleIntercept"]
    RescaleSlope = row["RescaleSlope"]
    adjusted_hu = image * RescaleSlope + RescaleIntercept

    # Convert the image to int16
    adjusted_hu = sitk.Cast(adjusted_hu, sitk.sitkInt16)

    # Write Image
    dirpath = os.path.dirname(filepath)
    dirpath = dirpath.replace("/train/", "/train_fixed/")
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(adjusted_hu, os.path.join(dirpath, os.path.basename(filepath)))


def main(metadata):

    # Convert DataFrame rows to a list of Series, each representing a row
    rows = [row[1] for row in metadata.iterrows()]
    for row in tqdm(rows):
        process_row(row)


if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a part of a DataFrame.")
    parser.add_argument(
        "part_num", type=int, default=1, help="The part number to process (1-indexed)."
    )
    parser.add_argument(
        "total_parts",
        type=int,
        default=12,
        help="The total number of parts to divide the DataFrame into.",
    )
    args = parser.parse_args()

    data_root = "<CT_RATE_DATASET_DIR>"
    metadata = pd.read_csv(
        os.path.join(data_root, "dataset/metadata/train_metadata.csv")
    )

    # Calculate the number of rows in each part
    total_rows = len(metadata)
    part_size = total_rows // args.total_parts
    remainder = total_rows % args.total_parts

    # Calculate the start and end indices for the slice
    start = (args.part_num - 1) * part_size + min(args.part_num - 1, remainder)
    end = start + part_size + (1 if args.part_num <= remainder else 0)

    main(metadata.iloc[start:end])

from typing import Tuple, Optional
import nibabel as nib
import numpy as np


def load_nifti(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load CT image in nii.gz format

    Args:
        file_path: nii.gz file path

    Returns:
        image_data: Image data array
        header_info: Header information dictionary
    """
    # Load image
    img = nib.load(file_path)
    image_data = img.get_fdata()
    affine = img.affine

    # Get header information
    header = img.header
    header_info = {
        "shape": image_data.shape,
        "affine": affine,
        "data_type": header.get_data_dtype(),
        "voxel_size": header.get_zooms(),
    }

    return image_data, header_info


def save_nifti(
    image_data: np.ndarray,
    affine: np.ndarray,
    output_path: str,
    data_type: Optional[str] = None,
):
    """
    Save as nii.gz format

    Args:
        image_data: Image data
        affine: Affine matrix
        output_path: Output file path
        data_type: Data type, if None automatically inferred
    """
    if data_type is None:
        data_type = image_data.dtype

    # Create Nifti image object
    img = nib.Nifti1Image(image_data.astype(data_type), affine)

    # Save image
    nib.save(img, output_path)

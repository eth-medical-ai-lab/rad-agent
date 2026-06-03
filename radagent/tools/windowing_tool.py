from pathlib import Path
from typing import List, Union
import numpy as np
from constants_and_path_utils import WINDOWED_IMAGES
from tools.ct_utils import load_nifti, save_nifti
from PIL import Image


def ct_windowing(
    img: np.ndarray,
    window_center: float,
    window_width: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Adjust CT image window width and window level

    Args:
        image: Input CT image array (HU values)
        window_center: Window Level
        window_width: Window Width

    Returns:
        windowed_image: Image after window width/level adjustment, value range [0, 1]
    """
    # Calculate window range
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2

    img = np.clip(img, window_min, window_max)

    if normalize:
        # Normalize to [0, 1]
        img = (img - window_min) / (window_max - window_min)
    return img


def windowing_fn(
    input_files: Union[str, List[str]],
    window: str,
):
    """
    Apply CT window width/level adjustment to nii.gz file

    Args:
        input_file: Input nii.gz file path
        output_file: Output nii.gz file path
        window_center: Window level
        window_width: Window width
        preset: Preset window type, optional 'brain', 'lung', 'bone', 'abdomen'
    """
    if isinstance(input_files, str):
        input_files = [input_files]
    # Preset window parameters
    presets = {
        "lung": (-600, 1500),  # Lung window
        "bone": (300, 1500),  # Bone window
        "abdomen": (60, 350),  # Abdomen window
        "mediastinum": (50, 350),  # Mediastinum window
    }
    assert window in presets, f"Unsupported window preset: {window}"
    output_files = []
    for input_file in input_files:
        if input_file.endswith(".nii") or input_file.endswith(".nii.gz"):
            out_dir = WINDOWED_IMAGES / Path(input_file).stem
        else:
            file = str(input_file).replace("'", "").replace(".npy", "").split("/")[-2:]
            out_dir = WINDOWED_IMAGES / file[0] / file[1]
        out_dir.mkdir(parents=True, exist_ok=True)
        input_file = str(input_file).replace("'", "")
        if input_file.endswith(".nii") or input_file.endswith(".nii.gz"):
            output_file = out_dir / f"{window}_window.nii.gz"
        elif input_file.endswith(".npy"):
            output_file = out_dir / f"{window}_window.png"
        else:
            return f"ERROR: Unsupported file format for input file: {input_file} needs .npy or .nii"
        if output_file.exists():
            output_files.append(str(output_file))
            continue

        window_center, window_width = presets[window]

        # Load image
        if input_file.endswith(".nii") or input_file.endswith(".nii.gz"):
            image_data, header_info = load_nifti(input_file)
        elif input_file.endswith(".npy"):
            image_data = np.load(input_file)
            header_info = {
                "shape": image_data.shape,
                "voxel_size": (1.0, 1.0, 1.0),  # Placeholder voxel size
                "affine": np.eye(4),  # Placeholder affine
            }

        # Apply window width/level
        windowed_image = ct_windowing(
            image_data,
            window_center,
            window_width,
            normalize=input_file.endswith(".npy"),
        )

        # Save result
        if input_file.endswith(".nii") or input_file.endswith(".nii.gz"):
            save_nifti(windowed_image, header_info["affine"], output_file)

        elif input_file.endswith(".npy"):
            windowed_image = (windowed_image * 255).astype(np.uint8)
            im = Image.fromarray(windowed_image)
            im.save(output_file)
        output_files.append(str(output_file))
    return f"Saved {window.capitalize()} images at: {output_files}"


if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools

    args = args_tools()
    mcp = FastMCP("see", stateless_http=False)

    @mcp.tool()
    async def windowing_tool(input_file: Union[str, List[str]], window: str) -> dict:
        result = windowing_fn(input_file, window)
        return {"meta": input_file, "outputs": result}

    mcp.run(transport="http", host=args.host, port=args.port)

from pathlib import Path
import numpy as np
from tools.ct_utils import load_nifti
from constants_and_path_utils import PRECOMPUTED_SEG


def slice_equidistant_select(ct_path, n_slices, direction="axial"):
    out_dir = PRECOMPUTED_SEG / Path(ct_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ct_data, _ = load_nifti(ct_path)
    except FileNotFoundError:
        return f"ERROR: CT file not found at {ct_path}"
    if direction == "axial":
        n_total = ct_data.shape[-1]
        indices_to_select = [
            int((i / (n_slices + 1)) * n_total) for i in range(1, n_slices + 1)
        ]
        print("Selected z indices:", indices_to_select)
        exported_arrays = []
        for z in indices_to_select:
            slice_img = ct_data[:, :, z]
            out_array = out_dir / f"axial_slice_{z:03d}.npy"
            exported_arrays.append(out_array)
            np.save(out_array, slice_img)

    elif direction == "sagittal":
        n_total = ct_data.shape[-3]
        indices_to_select = [
            int((i / (n_slices + 1)) * n_total) for i in range(1, n_slices + 1)
        ]
        print("Selected x indices:", indices_to_select)
        exported_arrays = []
        for x in indices_to_select:
            slice_img = ct_data[x, :, :]
            out_array = out_dir / f"sagittal_slice_{x:03d}.npy"
            exported_arrays.append(out_array)
            np.save(out_array, slice_img)
    elif direction == "coronal":
        n_total = ct_data.shape[-2]
        indices_to_select = [
            int((i / (n_slices + 1)) * n_total) for i in range(1, n_slices + 1)
        ]
        print("Selected y indices:", indices_to_select)
        exported_arrays = []
        for y in indices_to_select:
            slice_img = ct_data[:, y, :]
            out_array = out_dir / f"coronal_slice_{y:03d}.npy"
            exported_arrays.append(out_array)
            np.save(out_array, slice_img)
    return f"Relevant CT slices files: {[str(p) for p in exported_arrays]}"


if __name__ == "__main__":
    # image_path = "/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/train_fixed/dataset/train_fixed/train_18/train_18_a/train_18_a_2.nii.gz"
    # segmentation_path = "/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/train_fixed/dataset/train_fixed/train_18/train_18_a/train_18_a_2_segmentation.nii.gz"
    # result = slice_equidistant_select_tool(Path(image_path), Path(segmentation_path))
    # print(result)
    from fastmcp import FastMCP
    from tool_configs import args_tools

    args = args_tools()
    mcp = FastMCP("see", stateless_http=False)

    @mcp.tool()
    async def extract_slices_from_ct(
        image_path: str,
        n_slices: int = 5,
        direction: str = "axial",
    ) -> dict:
        result = slice_equidistant_select(Path(image_path), n_slices, direction)
        return {"meta": image_path, "outputs": result}

    mcp.run(transport="http", host=args.host, port=args.port)

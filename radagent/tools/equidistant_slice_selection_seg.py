from pathlib import Path
import numpy as np
from tools.ct_utils import load_nifti


def slice_equidistant_select(ct_path, seg_path, n_slices, min_voxels_per_slice=5):
    if ct_path == seg_path:
        return "ERROR: Invalid segmentation path, please generate a segmentation map first."
    try:
        ct_data, _ = load_nifti(ct_path)
    except FileNotFoundError:
        return f"ERROR: CT file not found at {ct_path}"
    # print("CT shape:", ct_data.shape)
    try:
        seg_data, _ = load_nifti(seg_path)
        if seg_data.max() > 1 or seg_data.min() < 0:
            return "ERROR: Invalid segmentation path, please generate a segmentation map first."
    except FileNotFoundError:
        return f"ERROR: Segmentation file not found at {seg_path}"
    seg_mask = seg_data > 0.5
    # This seems fishy - needs checking
    lesion_per_slice = seg_mask.sum(axis=(0, 1))
    # print(lesion_per_slice.shape)

    all_slice_info = []
    for z, n_lesion in enumerate(lesion_per_slice):
        if n_lesion > min_voxels_per_slice:
            all_slice_info.append((z, n_lesion))
    if not all_slice_info:
        return "No relevant slices found in the segmentation map."

    # check if there is only one leision/abnormality
    z_values = sorted(list(set([info[0] for info in all_slice_info])))
    # print(all_slice_info)
    discontinuous_groups = []
    current_group = [z_values[0]]

    for z in z_values:
        if z - current_group[-1] > 1:
            discontinuous_groups.append(current_group)
            current_group = [z]
        else:
            current_group.append(z)
    discontinuous_groups.append(current_group)

    # print(f"detected {len(discontinuous_groups)} abnormalities。")

    selected_z_indices = []
    for group in discontinuous_groups:
        all_slices_in_group = sorted(
            [info[0] for info in all_slice_info if info[0] in group]
        )
        indices_to_select = [
            int((i / (n_slices + 1)) * len(all_slices_in_group))
            for i in range(1, n_slices + 1)
        ]
        selected_z_indices.extend(
            [
                all_slices_in_group[idx]
                for idx in indices_to_select
                if idx < len(all_slices_in_group)
            ]
        )

    selected_z_indices = sorted(selected_z_indices)
    print("Selected z indices:", selected_z_indices)
    exported_arrays = []
    out_dir = seg_path.parent / seg_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for z in selected_z_indices:
        slice_img = ct_data[:, :, z]
        out_array = out_dir / f"slice_{z:03d}.npy"
        exported_arrays.append(out_array)
        out_array.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_array, slice_img)

    return f"Relevant CT slices files: {[str(p) for p in exported_arrays]}"


if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools

    args = args_tools()
    mcp = FastMCP("see", stateless_http=False)

    @mcp.tool()
    async def get_several_slices_from_segmentation(
        image_path: str, segmentation_path: str, n_slices: int = 3
    ) -> dict:
        result = slice_equidistant_select(
            Path(image_path), Path(segmentation_path), n_slices
        )
        return {"meta": image_path, "outputs": result}

    mcp.run(transport="http", host=args.host, port=args.port)

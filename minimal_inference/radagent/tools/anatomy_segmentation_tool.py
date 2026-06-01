from pathlib import Path
from typing import List
from constants_and_path_utils import PRECOMPUTED_SEG
from tools.effusion_segmentation_tool import run_totalsegmentator

available_organs = [
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    # "adrenal_gland_right",
    # "adrenal_gland_left",
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "esophagus",
    "trachea",
    "thyroid_gland",
    # "small_bowel",
    # "colon",
    "kidney_cyst_left",
    "kidney_cyst_right",
    "heart",
    "aorta",
    "pulmonary_vein",
    "brachiocephalic_trunk",
    # "subclavian_artery_right",
    # "subclavian_artery_left",
    # "common_carotid_artery_right",
    # "common_carotid_artery_left",
    # "brachiocephalic_vein_left",
    # "brachiocephalic_vein_right",
    # "atrial_appendage_left",
    # "superior_vena_cava",
    # "inferior_vena_cava",
    # "portal_vein_and_splenic_vein",
    # "iliac_artery_left",
    # "iliac_artery_right",
    # "iliac_vena_left",
    # "iliac_vena_right",
]

def anatomy_segmentation(ct_path, out_dir, organs):
    args = ["-ta", "total"]
    error = None
    out_dir = Path(out_dir)
    for o in organs:
        if o not in available_organs:
            if error is None:
                error = "ERROR:"
            error += f"Organ {o} not in available organs list."
    if error is not None:
        return error
    target_paths = [out_dir / f"{o}.nii.gz" for o in organs]
    args += ["--roi_subset"] + organs
    if not all([p.exists() for p in target_paths]):
        run_totalsegmentator(ct_path, out_dir, args)
    return f"List of saved segmentation files: {[str(p) for p in target_paths]}"


if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools

    args = args_tools()
    mcp = FastMCP("see", stateless_http=False)

    @mcp.tool()
    async def anatomy_segmentation_tool(image_path: str, organs: List[str]) -> dict:
        """Generates organ segmentation maps with TotalSegmentator."""
        out_dir = PRECOMPUTED_SEG / Path(image_path).stem
        result = anatomy_segmentation(image_path, out_dir, organs)
        return {"meta": image_path, "outputs": result}

    mcp.run(transport="http", host=args.host, port=args.port)

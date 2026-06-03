# -*- coding: utf-8 -*-
import os
import glob
from pathlib import Path
import subprocess


def run_totalsegmentator(ct_path, out_dir, args):

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "TotalSegmentator",
        "-i",
        str(ct_path),
        "-o",
        str(out_dir),
        "--device",
        "gpu",
    ] + args
    print(">>> running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(">>> TotalSegmentator finished, result stored in:", out_dir)


def effusion_segmentation(ct_path, out_dir):
    args = ["-ta", "pleural_pericard_effusion"]
    out_dir = Path(out_dir)
    target_paths = [
        out_dir / "pleural_effusion.nii.gz",
        out_dir / "pericardial_effusion.nii.gz",
    ]
    if not all([p.exists() for p in target_paths]):
        run_totalsegmentator(ct_path, out_dir, args)
        # list files in out_dir
        print(glob.glob(str(out_dir / "*.nii.gz")))
    return f"List of saved segmentation files: {[str(p) for p in target_paths]}"


if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools
    from constants_and_path_utils import PRECOMPUTED_SEG

    args = args_tools()

    mcp = FastMCP("see", stateless_http=False)

    @mcp.tool()
    async def effusion_segmentation_tool(image_path: str) -> dict:
        """Generates pleural and pericardial effusion segmentations maps."""
        out_dir = PRECOMPUTED_SEG / Path(image_path).stem
        result = effusion_segmentation(image_path, out_dir)
        return {"meta": image_path, "outputs": result}

    mcp.run(transport="http", host=args.host, port=args.port)

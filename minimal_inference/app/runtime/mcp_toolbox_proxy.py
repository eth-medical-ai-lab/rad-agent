import argparse
from typing import List, Literal, Union

from fastmcp import Client, FastMCP

from app.runtime.tool_server_registry import get_servers


mcp = FastMCP("minimal-inference-toolbox", stateless_http=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


async def list_backend_tools(server_name: str):
    async with Client(get_servers()[server_name]["addr"], timeout=3600) as client:
        return await client.list_tools()


async def call_backend(server_name: str, tool_name: str, arguments: dict) -> dict:
    async with Client(get_servers()[server_name]["addr"], timeout=3600) as client:
        result = await client.call_tool(tool_name, arguments)
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    return {"meta": None, "outputs": result}


@mcp.tool()
async def toolbox_status() -> dict:
    """Show connectivity to each backend tool server."""
    outputs = {}
    for server_name, config in get_servers().items():
        item = {
            "addr": config["addr"],
            "port": config.get("port"),
            "env": config["env"],
            "device": config["device"],
            "ready": False,
        }
        try:
            tools = await list_backend_tools(server_name)
            item["ready"] = True
            item["tools"] = [getattr(tool, "name", "unknown") for tool in tools]
        except Exception as exc:  # noqa: BLE001
            item["error"] = str(exc)
        outputs[server_name] = item
    return {"meta": None, "outputs": outputs}


@mcp.tool()
async def windowing_tool(
    input_file: Union[str, List[str]],
    window: Literal["lung", "bone", "abdomen", "mediastinum"],
) -> dict:
    """Apply a CT viewing window to a volume or one or more slices."""
    return await call_backend(
        "windowing_tool",
        "windowing_tool",
        {"input_file": input_file, "window": window},
    )


@mcp.tool()
async def extract_slices_from_ct(
    image_path: str,
    n_slices: int = 5,
    direction: Literal["axial", "sagittal", "coronal"] = "axial",
) -> dict:
    """Extract evenly spaced CT slices as .npy files for downstream slice tools such as slice_vqa_tool."""
    return await call_backend(
        "extract_slices_from_ct",
        "extract_slices_from_ct",
        {
            "image_path": image_path,
            "n_slices": n_slices,
            "direction": direction,
        },
    )


@mcp.tool()
async def biggest_slice_selection_tool(
    image_path: str,
    segmentation_path: str,
) -> dict:
    """Pick the largest segmented slice for each segmented region."""
    return await call_backend(
        "biggest_slice_selection_tool",
        "biggest_slice_selection_tool",
        {
            "image_path": image_path,
            "segmentation_path": segmentation_path,
        },
    )


@mcp.tool()
async def get_several_slices_from_segmentation(
    image_path: str,
    segmentation_path: str,
    n_slices: int = 3,
) -> dict:
    """Extract several informative slices across a segmented region."""
    return await call_backend(
        "get_several_slices_from_segmentation",
        "get_several_slices_from_segmentation",
        {
            "image_path": image_path,
            "segmentation_path": segmentation_path,
            "n_slices": n_slices,
        },
    )


@mcp.tool()
async def anatomy_segmentation_tool(image_path: str, organs: List[str]) -> dict:
    """Generate organ segmentation masks with TotalSegmentator. Supported organ names are exact strings such as lung_upper_lobe_left, lung_lower_lobe_left, lung_upper_lobe_right, lung_middle_lobe_right, lung_lower_lobe_right, trachea, heart, aorta, esophagus, thyroid_gland, liver, spleen, stomach, pancreas, kidney_left, kidney_right, kidney_cyst_left, kidney_cyst_right, pulmonary_vein, and brachiocephalic_trunk. Use effusion_segmentation_tool for pleural/pericardial effusion rather than requesting pleura here."""
    return await call_backend(
        "anatomy_segmentation_tool",
        "anatomy_segmentation_tool",
        {
            "image_path": image_path,
            "organs": organs,
        },
    )


@mcp.tool()
async def effusion_segmentation_tool(image_path: str) -> dict:
    """Generate pleural and pericardial effusion masks with TotalSegmentator."""
    return await call_backend(
        "effusion_segmentation_tool",
        "effusion_segmentation_tool",
        {
            "image_path": image_path,
        },
    )


@mcp.tool()
async def disease_classifier_tool(image_path: str) -> dict:
    """Classify a CT volume across the pathology label set."""
    return await call_backend(
        "disease_classifier_tool",
        "disease_classifier_tool",
        {
            "image_path": image_path,
        },
    )


@mcp.tool()
async def slice_vqa_tool(image_paths: List[str], question: str) -> dict:
    """Ask a question about one or more extracted CT slices. Accepts PNG or .npy slice files, not NIfTI volumes; use extract_slices_from_ct first when starting from a CT volume."""
    return await call_backend(
        "slice_vqa_tool",
        "slice_vqa_tool",
        {
            "image_paths": image_paths,
            "question": question,
        },
    )


@mcp.tool()
async def ct_vqa_tool(query: str, image_path: str) -> dict:
    """Ask a question about a CT volume with the CT VQA backend."""
    return await call_backend(
        "ct_vqa_tool",
        "ct_vqa_tool",
        {
            "query": query,
            "image_path": image_path,
        },
    )


@mcp.tool()
async def report_generation_tool(query: str, image_path: str) -> dict:
    """Generate a draft CT report from a volume and prompt."""
    return await call_backend(
        "report_generation_tool",
        "report_generation_tool",
        {
            "query": query,
            "image_path": image_path,
        },
    )


if __name__ == "__main__":
    args = parse_args()
    mcp.run(transport="http", host=args.host, port=args.port)

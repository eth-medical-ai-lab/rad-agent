from constants_and_path_utils import CT_CHAT_MODELS, PRECOMPUTED_CLASSIFIER

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "toolbox_src" / "CT_CLIP_main"))
from scripts.zero_shot_clip_inference import SingleFileInference_clip


class DiseaseClassifierTool:
    def __init__(self):
        self.model = SingleFileInference_clip(
            model_path=CT_CHAT_MODELS / "models/CT-CLIP-Related/CT_VocabFine_v2.pt"
        )

    def run(self, image_path):
        filepath = PRECOMPUTED_CLASSIFIER / f"{Path(image_path).stem}.txt"
        if filepath.exists():
            with open(filepath, "r") as f:
                outputs = f.read()
            return {
                "meta": None,
                "outputs": outputs,
            }

        outputs = self.model.predict_and_generate_report(
            nii_path=image_path,
        )
        with open(filepath, "w") as f:
            f.write(outputs)
        return {
            "meta": None,
            "outputs": outputs,
        }


if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools
    import asyncio

    request_semaphore = asyncio.Semaphore(8)

    args = args_tools()

    mcp = FastMCP("see", stateless_http=False)
    disease_classifier_tool_instance = DiseaseClassifierTool()

    @mcp.tool()
    async def disease_classifier_tool(image_path: str) -> dict:
        """Analyzes CT scans and classifies them for 18 different pathologies."""
        async with request_semaphore:
            # Run the synchronous tool in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, disease_classifier_tool_instance.run, image_path
            )

    print(f"Starting disease_classifier_tool on {args.host}:{args.port}")
    mcp.run(transport="http", host=args.host, port=args.port)

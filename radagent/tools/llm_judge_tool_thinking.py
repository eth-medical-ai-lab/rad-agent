from typing import List
from tools.llm_judge_tool import ReportJudgeTool


if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools
    from vllm import SamplingParams

    args = args_tools()
    mcp = FastMCP("see", stateless_http=False)
    report_judge_tool_instance = ReportJudgeTool(
        model_name="Qwen/Qwen3-30B-A3B-Thinking-2507",
        sampling_params=SamplingParams(temperature=0.1, top_p=0.95, max_tokens=25000),
    )

    @mcp.tool()
    async def report_judge_tool(
        ground_truth_report: str, candidate_report: str
    ) -> dict:
        return await report_judge_tool_instance.run_report_judge(
            ground_truth_report=ground_truth_report,
            candidate_report=candidate_report,
        )

    @mcp.tool()
    async def trajectory_judge_tool(
        conversation_trajectory: List[dict],
    ) -> dict:
        return await report_judge_tool_instance.run_tool_sequence_judge(
            conversation_trajectory
        )

    @mcp.tool()
    async def summary_judge_tool(
        conversation_trajectory: List[dict],
    ) -> dict:
        return await report_judge_tool_instance.run_summary_judge(
            conversation_trajectory
        )

    mcp.run(transport="http", host=args.host, port=args.port)

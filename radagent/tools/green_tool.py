import asyncio
import concurrent.futures
from typing import List
from evaluation.green_score.green import GREEN


# --- 1. Modified Tool with Internal Queue Logic ---


class GreenTool:
    def __init__(self):
        model_name = "StanfordAIMI/GREEN-radllama2-7b"
        # Initializing the model (RadLlama-2-7b)
        self.green_scorer = GREEN(model_name, output_dir=".", cpu=False)

        # Batching Config
        self.queue = asyncio.Queue()
        self.batch_size = 24
        self.batch_timeout = 2  # 400ms window
        self._worker_task = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def ensure_worker_started(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self.worker())

    async def worker(self):
        loop = asyncio.get_running_loop()
        try:
            while True:
                # Wait for the first request
                first_item = await self.queue.get()
                batch = [first_item]

                # Start the timer for the batch
                start_time = loop.time()
                while len(batch) < self.batch_size:
                    time_left = self.batch_timeout - (loop.time() - start_time)
                    if time_left <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(), timeout=time_left
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # Extract data for processing
                gt_reports = [b[0] for b in batch]
                cand_reports = [b[1] for b in batch]
                futures = [b[2] for b in batch]

                try:
                    # Run the heavy LLM scoring in the thread pool
                    results = await loop.run_in_executor(
                        self.executor, self.run_batch, gt_reports, cand_reports
                    )

                    for res, fut in zip(results, futures):
                        if not fut.done():
                            fut.set_result(res)
                except Exception as e:
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(e)
                finally:
                    for _ in range(len(batch)):
                        self.queue.task_done()

        except asyncio.CancelledError:
            self.executor.shutdown(wait=False)
            raise

    def run_batch(self, gt_reports: List[str], cand_reports: List[str]) -> List[float]:
        """Synchronous batch execution on the GPU."""
        # GREEN's __call__ or similar method usually takes lists for batching
        return self.green_scorer(gt_reports, cand_reports)[2]


# --- 2. MCP Server Setup ---

if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools

    args = args_tools()
    mcp = FastMCP("green_server", stateless_http=False)
    green_tool_instance = GreenTool()

    @mcp.tool()
    async def green_tool(ground_truth_report: str, candidate_report: str) -> dict:
        """Calculate GREEN score for radiology reports using batching."""
        await green_tool_instance.ensure_worker_started()

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Queue the work
        await green_tool_instance.queue.put(
            (ground_truth_report, candidate_report, future)
        )

        try:
            score = await future
            return {
                "meta": None,
                "outputs": str(score),
            }
        except Exception as e:
            return {"meta": "Error", "outputs": str(e)}

    mcp.run(transport="http", host=args.host, port=args.port)

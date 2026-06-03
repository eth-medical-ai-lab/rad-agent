import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "toolbox_src" / "CT_CHAT_main"))
from ct_chat_full_model import CTChat_full_model


class ReportGenerationTool:
    def __init__(self):
        self.model = CTChat_full_model()
        self.queue = asyncio.Queue()
        self.batch_size = 4
        self.batch_timeout = 2  # 500ms window to collect queries
        self._worker_task = None

    async def ensure_worker_started(self):
        """Ensure the worker is running (idempotent)."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self.worker())

    async def worker(self):
        """Background loop that monitors the queue and batches queries."""
        try:
            while True:
                # Wait for at least one item to start a batch
                first_item = await self.queue.get()
                batch = [first_item]
                print(f"Batch started with query: {first_item[0][:50]}...", flush=True)

                # Try to fill the batch until batch_size or timeout
                start_time = asyncio.get_event_loop().time()
                while len(batch) < self.batch_size:
                    time_left = self.batch_timeout - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    if time_left <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(), timeout=time_left
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # Separate inputs and the futures associated with them
                # batch item structure: (query, image_path, future)
                queries = [b[0] for b in batch]
                image_paths = [b[1] for b in batch]
                futures = [b[2] for b in batch]

                print(f"Processing ct-chat batch of {len(queries)} queries", flush=True)

                try:
                    results = self.run_batch(queries, image_paths)

                    for res, fut in zip(results, futures):
                        if not fut.done():
                            fut.set_result(res)
                except Exception as e:
                    print(f"Error processing batch: {e}", flush=True)
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(e)

        except asyncio.CancelledError:
            print("Worker shutting down gracefully...", flush=True)
            raise  # Re-raise to properly cancel the task

    def run_batch(self, queries, image_paths):
        queries = [
            "<image>\n<provided>" + query + "<report_generation>" for query in queries
        ]
        return self.model.run_batch(queries, image_paths)


if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools

    args = args_tools()

    mcp = FastMCP("see", stateless_http=False)
    report_generation_tool_instance = ReportGenerationTool()

    @mcp.tool()
    async def report_generation_tool(query: str, image_path: str) -> dict:
        """Analyze CT images with automatic background batching."""
        # Ensure worker is started on first call
        await report_generation_tool_instance.ensure_worker_started()

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Put the request into the tool's internal queue
        await report_generation_tool_instance.queue.put((query, image_path, future))

        # Wait for the background worker to resolve this specific call
        try:
            result = await future
            return {
                "meta": None,
                "outputs": result,
            }
        except Exception as e:
            return {"meta": "Error", "outputs": str(e)}

    mcp.run(transport="http", host=args.host, port=args.port)

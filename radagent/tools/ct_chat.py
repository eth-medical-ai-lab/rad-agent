import sys
import asyncio
from pathlib import Path
from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent / "toolbox_src" / "CT_CHAT_main"))
from ct_chat_full_model import CTChat_full_model
from tool_configs import args_tools


class GPUWorker:
    """Manages a single GPU and its batching logic."""

    def __init__(self, device_id, shared_queue):
        self.device_id = device_id
        self.queue = shared_queue
        self.batch_size = 7
        self.batch_timeout = 2

        print(f"Initializing model on GPU {device_id}...", flush=True)
        self.model = CTChat_full_model(
            device=f"cuda:{device_id}"
        )  # Pass device_id if your model supports it

    async def run(self):
        """The main loop for this specific GPU worker."""
        loop = asyncio.get_running_loop()

        # specific executor for this worker to ensure we don't block the other GPU
        # We generally just need one global pool, but let's keep it simple.

        print(f"Worker {self.device_id} started.", flush=True)

        while True:
            # 1. Wait for the first item (Competing for the shared queue)
            first_item = await self.queue.get()
            batch = [first_item]
            print(f"[GPU {self.device_id}] Batch started...", flush=True)

            # 2. Try to fill the rest of the batch
            start_time = loop.time()
            while len(batch) < self.batch_size:
                time_left = self.batch_timeout - (loop.time() - start_time)
                if time_left <= 0:
                    break
                try:
                    # We wait for more items to arrive in the shared queue
                    item = await asyncio.wait_for(self.queue.get(), timeout=time_left)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # 3. Process the batch
            queries = [b[0] for b in batch]
            image_paths = [b[1] for b in batch]
            futures = [b[2] for b in batch]

            try:
                # CRITICAL: Run in executor so this GPU doesn't block the event loop
                # This allows the OTHER worker to keep picking up items while this one computes.
                results = await loop.run_in_executor(
                    None, self.run_batch_sync, queries, image_paths
                )

                for res, fut in zip(results, futures):
                    if not fut.done():
                        fut.set_result(res)

            except Exception as e:
                print(f"[GPU {self.device_id}] Error: {e}", flush=True)
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)

    def run_batch_sync(self, queries, image_paths):
        """The blocking synchronous code that runs on the GPU."""
        formatted_queries = [
            "<image>\n<provided>" + q + "<short_answe>" for q in queries
        ]
        # Ensure the model uses the correct device logic internally if needed
        return self.model.run_batch(formatted_queries, image_paths)


class MultiGPUToolManager:
    def __init__(self, num_gpus=2):
        self.queue = asyncio.Queue(maxsize=100)  # Added backpressure cap
        self.workers = []
        self.num_gpus = num_gpus
        self._started = False

    async def ensure_workers_started(self):
        if self._started:
            return

        # Create a worker for each GPU
        for i in range(self.num_gpus):
            worker = GPUWorker(device_id=i, shared_queue=self.queue)
            asyncio.create_task(worker.run())
            self.workers.append(worker)

        self._started = True


# --- MCP Server Setup ---
if __name__ == "__main__":
    args = args_tools()
    mcp = FastMCP("see", stateless_http=False)

    # Initialize the Manager (not the model yet)
    tool_manager = MultiGPUToolManager(num_gpus=2)

    @mcp.tool()
    async def ct_vqa_tool(query: str, image_path: str) -> dict:
        """Analyze CT images using multi-gpu batching."""
        await tool_manager.ensure_workers_started()

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Add to the shared queue. The first available GPU worker will pick it up.
        await tool_manager.queue.put((query, image_path, future))

        try:
            result = await future
            return {"meta": None, "outputs": result}
        except Exception as e:
            return {"meta": "Error", "outputs": str(e)}

    mcp.run(transport="http", host=args.host, port=args.port)

import asyncio
import concurrent.futures
from typing import List
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from evaluation.text_classifier_CT_pathology import CTPathologyClassifier
from collections import defaultdict


class CTPathologyClassifierF1Tool:
    def __init__(self):
        self.classifier = CTPathologyClassifier()
        self.classifier.eval()  # Set to eval mode for inference
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zzxslp/RadBERT-RoBERTa-4m", do_lower_case=True
        )
        # Batching Config
        self.queue = asyncio.Queue()
        self.batch_size = 24
        self.batch_timeout = 2  # 1500ms window
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
        output_encodings = defaultdict(list)
        ground_truth_encodings = defaultdict(list)
        batch_size = len(cand_reports)
        for output, ground_truth_report in zip(cand_reports, gt_reports):
            out_enc = self.tokenizer(
                output,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            gt_enc = self.tokenizer(
                ground_truth_report,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True,
            )

            for key in gt_enc:
                output_encodings[key].append(out_enc[key])
                ground_truth_encodings[key].append(gt_enc[key])

        for key in output_encodings:
            output_encodings[key] = torch.cat(output_encodings[key], dim=0)
            ground_truth_encodings[key] = torch.cat(ground_truth_encodings[key], dim=0)

        batch = {
            "predicted": output_encodings,
            "ground_truth": ground_truth_encodings,
        }
        with torch.no_grad():
            result = self.classifier.predict_binary(batch)
        result["ground_truth"] = result["ground_truth"].cpu().numpy()
        result["predicted"] = result["predicted"].cpu().numpy()
        f1_scores = []
        for i in range(batch_size):
            f1_scores.append(
                f1_score(
                    result["ground_truth"][i], result["predicted"][i], zero_division=1.0
                )
            )
        return f1_scores  # Convert to list for easier JSON serialization


# --- 2. MCP Server Setup ---

if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools

    args = args_tools()
    mcp = FastMCP("f1_text_server", stateless_http=False)
    f1_tool_instance = CTPathologyClassifierF1Tool()

    @mcp.tool()
    async def f1_text_classifier_tool(
        ground_truth_report: str, candidate_report: str
    ) -> dict:
        """Calculate F1 score for radiology reports using batching."""
        await f1_tool_instance.ensure_worker_started()

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Queue the work
        await f1_tool_instance.queue.put(
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

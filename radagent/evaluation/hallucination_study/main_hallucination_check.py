"""
Run the main hallucination-generation evaluation for a chosen hint condition.

The script builds evaluation scenarios, starts the configured agent stack, and
executes batch inference over the selected CT-RATE slice so the resulting
predictions can be analyzed later.
"""

import asyncio
import subprocess

from dotenv import load_dotenv
import weave
import argparse
from datetime import datetime
from utils import kill_job
from agents.custom_orchestrators import load_custom_agent
import random
import os
import pandas as pd
import threading
from agents.main_ct_rate import run_eval
from agents.art_dataset import generate_hallu_generation_scenario

os.environ["WANDB_INIT_TIMEOUT"] = "180"


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hallu_type",
        type=str,
        choices=["orig", "think", "sure", "correct_think", "correct_sure"],
        required=True,
        help="Type of hallucination query to use: 'orig' for original query, 'think' for adding a hint that the clinician thinks the patient has the pathology, 'sure' for adding a hint that the clinician is sure the patient has the pathology.",
    )

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=1000)
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["vanilla", "v8b", "v8c", "v8minus"],
        default="v8c",
    )
    parser.add_argument("--model_name", type=str, default="OpenPipe/Qwen3-14B-Instruct")
    parser.add_argument(
        "--inference_model_name",
        type=str,
        default="OpenPipe/Qwen3-14B-Instruct",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for processing queries"
    )
    parser.add_argument(
        "--do_check_summary",
        action="store_true",
        help="Enable check summary (default: False)",
    )

    parser.add_argument("--subdir", type=str, default="default")
    parser.add_argument(
        "--r",
        type=int,
        default=16,
    )

    return parser.parse_args()

async def main(args):
    PROJECT_NAME = "CTRate_hallu".lower()
    local_node = int(os.environ.get("LOCAL_NODE_RANK", 0))
    is_main_node = local_node == 0

    if is_main_node:
        print(f"Running with args: {args}", flush=True)
        t = datetime.now()
        weave.init(PROJECT_NAME)

    load_dotenv()

    random.seed(42)

    orchestrator = load_custom_agent(
        agent_type=args.agent_type,
        base_model_name=args.inference_model_name,
        subfolder_name=args.subdir,
        check_summary=args.do_check_summary,
        project_name=PROJECT_NAME,
        for_inference_only=True,
        r=args.r,
    )

    # Load orchestrator model and start servers in parallel for start-up speed up
    initialise_model_task = orchestrator.initialize_model()
    start_all_servers_task = orchestrator.server_manager.startup_all_servers()
    _, processes = await asyncio.gather(initialise_model_task, start_all_servers_task)

    if is_main_node:
        t1 = datetime.now()
        print(
            f"Total agent startup time: {(t1 - t).total_seconds()} seconds", flush=True
        )
        await orchestrator.connect_to_servers()

        print(f"Start index {args.start_id}", flush=True)

        scenarios = generate_hallu_generation_scenario(args.hallu_type, start_idx=args.start_id, end_idx=args.end_id)

        await run_eval(scenarios, orchestrator, args.batch_size)
        print("🎉 Validation completed!", flush=True)

        t2 = datetime.now()
        delta = t2 - t
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f"""
            Total time cost: {hours} hours, {minutes} minutes.
            Time for agent start-up {(t1 - t).total_seconds()} seconds.
            Time running per sample {(t2 - t1).total_seconds() / (args.end_id - args.start_id):.3f} seconds.
            """,
            flush=True,
        )

        kill_job()

        print("\nStopping servers...")
        for process, script in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    else:
        while True:
            await asyncio.sleep(3600)



if __name__ == "__main__":
    args = args_parse()
    asyncio.run(main(args))

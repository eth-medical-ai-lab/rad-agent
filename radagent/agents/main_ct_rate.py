from pathlib import Path
import asyncio
import subprocess
import sys
import time

from dotenv import load_dotenv
import weave
import argparse
from datetime import datetime
import numpy as np
from agents.art_dataset import (
    generate_report_generation_scenarios,
    generate_vqa_scenarios,
    generate_radchest_ct_scenarios,
)
import art
from utils import kill_job
from agents.custom_orchestrators import load_custom_agent
import logging
import random
import os

import threading

os.environ["WANDB_INIT_TIMEOUT"] = "180"


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=5000)
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
    parser.add_argument("--subdir", type=str, default="default")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val", "test", "inference_train"],
        default="val",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["report_generation", "vqa"],
        default="report_generation",
    )
    parser.add_argument("--n_rollout_per_group", type=int, default=8)
    # unused
    parser.add_argument(
        "--do_tool_augmentation",
        action="store_true",
        help="Enable tool augmentation (default: False)",
    )
    parser.add_argument(
        "--do_self_reflect",
        action="store_true",
        help="Enable self reflect (default: False)",
    )
    parser.add_argument(
        "--do_check_summary",
        action="store_true",
        help="Enable check summary (default: False)",
    )
    parser.add_argument(
        "--lambda_tool_success",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--lambda_llm_judge_f1",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lambda_llm_judge_prec_recall",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--lambda_green",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda_trajectory_judge",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--lambda_f1_text_classifier",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--lambda_manual_tool_judge",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.00,
    )

    parser.add_argument(
        "--r",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--shuffle_dataset",
        action="store_true",
    )

    parser.add_argument(
        "--dataset", type=str, default="ctrate", choices=["ctrate", "radchestct"]
    )

    return parser.parse_args()


async def run_eval(val_scenarios, orchestrator, batch_size):
    filtered_val_scenarios = []
    print(f"Checking for existing evaluation files in {orchestrator.output_dir / 'trajectory'}...", flush=True)
    for v in val_scenarios:
        image_path = v["image_path"]
        file_suffix = v["task_id"]
        filename = str(
                    orchestrator.output_dir
                    / "trajectory"
                    / (
                        Path(
                            str(image_path)
                            .replace(".nii", "_trajectory")
                            .replace(".gz", "")
                        ).name
                        + str(file_suffix)
                        + ".json"
                    )
                )
        if Path(filename).exists():
            print(f"File {filename} already exists, skipping evaluation for this scenario.")
            continue
        filtered_val_scenarios.append(v)
    print(f"Running evaluation on {len(filtered_val_scenarios)} scenarios...")


    n_batches = len(filtered_val_scenarios) // batch_size + 1
    for b in range(n_batches):
        batch_idx = range(b * batch_size, min((b + 1) * batch_size, len(filtered_val_scenarios)))
        await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    [
                        orchestrator.rollout(
                            filtered_val_scenarios[i],
                            file_suffix=filtered_val_scenarios[i]["task_id"],
                            compute_reward=False,
                        )
                    ]
                )
                for i in batch_idx
            ),
            pbar_desc=f"Gather step {b}",
        )


async def run_one_batch(
    orchestrator,
    train_scenarios,
    batch_size,
    rollouts_per_group,
    learning_rate,
    b,
    beta,
):
    batch_idx = range(b * batch_size, min((b + 1) * batch_size, len(train_scenarios)))
    rdn = np.random.rand(batch_size, len(orchestrator.tools_to_use))
    groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(
                orchestrator.rollout(
                    train_scenarios[idx],
                    compute_reward=True,
                    random_tool_gen=rdn[i],
                    file_suffix=str(r),
                )
                for r in range(rollouts_per_group)
            )
            for i, idx in enumerate(batch_idx)
        ),
        pbar_desc=f"Gather step {b}",
    )
    try:
        await orchestrator.model.delete_checkpoints("train/reward")
    except Exception as e:
        logging.warning(f"Failed to delete checkpoints: {e}")
    await orchestrator.model.train(
        trajectory_groups=groups,
        config=art.TrainConfig(learning_rate=learning_rate, beta=beta),
        verbose=True,
    )


async def training_loop(
    orchestrator,
    train_scenarios,
    rollouts_per_group=2,
    learning_rate=1e-5,
    batch_size=1,
    beta=0.05,
):
    max_time_per_batch = 3600 * 3
    n_batches = len(train_scenarios) // batch_size + 1

    for b in range(n_batches):
        stop_watchdog = threading.Event()

        def watchdog():
            start_time = time.time()
            while time.time() - start_time < max_time_per_batch:
                if stop_watchdog.is_set():
                    print(f"Watchdog {b} stopping cleanly", flush=True)
                    return  # Exit cleanly
                time.sleep(0.1)

            if not stop_watchdog.is_set():
                print(
                    f"TIMEOUT: Batch {b} exceeded {max_time_per_batch}s. Force exiting...",
                    flush=True,
                )
                os._exit(1)

        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()

        try:
            await run_one_batch(
                orchestrator,
                train_scenarios,
                batch_size,
                rollouts_per_group,
                learning_rate,
                b,
                beta,
            )
        except Exception as e:
            logging.exception(f"Batch {b} failed:")
            print(f"Batch {b} failed: {e}")
            stop_watchdog.set()
            watchdog_thread.join(timeout=1)  # Wait for thread to exit
            break

        stop_watchdog.set()
        watchdog_thread.join(timeout=1)  # Wait for watchdog to actually stop

        last_step = await orchestrator.model.get_step()


async def main(args):
    PROJECT_NAME = f"CTRate_{args.task}".lower()
    local_node = int(os.environ.get("LOCAL_NODE_RANK", 0))
    is_main_node = local_node == 0

    if is_main_node:
        print(f"Running with args: {args}", flush=True)
        t = datetime.now()
        # wandb.init(project=PROJECT_NAME)
        weave.init(PROJECT_NAME)

    load_dotenv()

    random.seed(42)

    orchestrator = load_custom_agent(
        agent_type=args.agent_type,
        base_model_name=(
            args.model_name if args.mode == "train" else args.inference_model_name
        ),
        subfolder_name=args.subdir,
        task=args.task,
        project_name=PROJECT_NAME,
        do_tool_augmentation=args.do_tool_augmentation,  # (args.mode == 'train'), #False, #(args.mode == 'train'),
        for_inference_only=(args.mode != "train"),
        lambda_tool_success=args.lambda_tool_success,
        lambda_llm_judge_f1=args.lambda_llm_judge_f1,
        lambda_llm_judge_prec_recall=args.lambda_llm_judge_prec_recall,
        lambda_green=args.lambda_green,
        lambda_trajectory_judge=args.lambda_trajectory_judge,
        lambda_manual_tool_judge=args.lambda_manual_tool_judge,
        self_reflect=args.do_self_reflect,
        check_summary=args.do_check_summary,
        r=args.r,
        lambda_f1_text_classifier=args.lambda_f1_text_classifier,
    )

    # Load orchestrator model and start servers in parallel for start-up speed up
    initialise_model_task = orchestrator.initialize_model()
    start_all_servers_task = orchestrator.server_manager.startup_all_servers()
    _, processes = await asyncio.gather(initialise_model_task, start_all_servers_task)

    finished_file_path = Path(f'{os.environ["SLURM_JOB_ID"]}_finished.txt')

    if is_main_node:
        t1 = datetime.now()
        print(
            f"Total agent startup time: {(t1 - t).total_seconds()} seconds", flush=True
        )
        await orchestrator.connect_to_servers()

        if args.mode == "train":
            last_step = await orchestrator.model.get_step()
            print(f"Starting training at step {last_step}", flush=True)
            start_id = (args.batch_size * last_step) + args.start_id
        else:
            start_id = args.start_id
        print(f"Start index {start_id}", flush=True)

        match args.dataset:
            case "ctrate":
                match args.task:
                    case "vqa":
                        scenarios = generate_vqa_scenarios(
                            mode=args.mode,
                            start_idx=start_id,
                            end_idx=args.end_id,
                            shuffle=args.shuffle_dataset,
                        )
                    case "report_generation":
                        scenarios = generate_report_generation_scenarios(
                            mode=args.mode,
                            start_idx=start_id,
                            end_idx=args.end_id,
                            shuffle=args.shuffle_dataset,
                        )
            case "radchestct":
                match args.task:
                    case "vqa":
                        raise NotImplementedError(
                            "VQA task not implemented for RadChest-CT dataset"
                        )
                    case "report_generation":
                        scenarios = generate_radchest_ct_scenarios(
                            start_idx=start_id, end_idx=args.end_id
                        )

        if args.mode != "train":
            await run_eval(scenarios, orchestrator, args.batch_size)
            print("🎉 Validation completed!", flush=True)

        elif args.mode == "train":
            assert (
                args.dataset == "ctrate"
            ), "Training is only implemented for CT-Rate dataset"

            await training_loop(
                orchestrator,
                scenarios,
                rollouts_per_group=args.n_rollout_per_group,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                beta=args.beta,
            )
            print("🎉 Training completed!", flush=True)

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

        # write file to indicate finished the file name should be the slurm job id
        if "SLURM_JOB_ID" in os.environ:
            with open(str(finished_file_path), "w") as f:
                f.write("finished")

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
        while not finished_file_path.exists():
            await asyncio.sleep(5)
        if finished_file_path.exists():
            os.remove(str(finished_file_path))

    sys.exit(0)


if __name__ == "__main__":
    args = args_parse()
    asyncio.run(main(args))

"""
This script runs full evaluation of CT-Chat on the CT-RATE report generation validation set.
"""
import argparse
import json
import os
import asyncio
from agents.art_dataset import generate_hallu_generation_scenario
from toolbox_src.CT_CHAT_main.ct_chat_full_model import CTChat_full_model
from pathlib import Path
from torch.utils.data import Dataset
from constants_and_path_utils import RADAGENT_RESULTS_DIR
from distributed_utils import (
    gather_processes,
    tqdm_on_main,
    create_distributed_dataloader_if_needed,
)

import torch
import torch.distributed as dist


class CTRAteReportGenerationDataset(Dataset):
    """
    Dataset for CT-RATE report generation task.
    Expects a json file with the original CT-RATE report generation format.
    Each sample returns a dictionary in a simplified format with the relevant fields
    (id, image, ground_truth).
    """

    def __init__(self, hallu_type="orig", start_idx=0, end_idx=None):
        scenarios = generate_hallu_generation_scenario(
            hallu_type=hallu_type, start_idx=start_idx, end_idx=end_idx
        )
        self.data = scenarios

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hallu_type",
        type=str,
        choices=["think", "correct_think", "orig"],
        default="orig",
        help="Type of hallucination query to use: 'orig' for original query, 'think' for adding a wrong hint that the clinician thinks the patient has the pathology, 'correct_think' for a correct hint.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="base",
        help="Name of the run, used for saving predictions and ground truth",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    end_idx = None
    print("Evaluating on hallu_type:", args.hallu_type)
    dataset = CTRAteReportGenerationDataset(
        hallu_type=args.hallu_type,
        end_idx=end_idx,
    )

    is_distributed = torch.cuda.is_available() and torch.cuda.device_count() > 1

    if is_distributed:
        if not dist.is_initialized():
            rank = int(os.environ.get("RANK", "0"))
            dist.init_process_group("nccl")
            device_id = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(device_id)
            print(f"Rank {rank}: Process group initialized.")
            if dist.get_rank() == 0:
                print("Distributed training with", torch.cuda.device_count(), "GPUs")
        global_rank = dist.get_rank()
    else:
        global_rank = 0
        device_id = 0

    print(f"Rank {global_rank}: Starting...")

    chat_instance = CTChat_full_model(device=f"cuda:{device_id}")
    chat_instance.model.eval().to(f"cuda:{device_id}")

    print(f"Rank {global_rank}: ==== Start Inference ====")
    dataloader = create_distributed_dataloader_if_needed(dataset, 4, shuffle=False)

    if is_distributed:
        dist.barrier()

    outputs = []
    gt = []
    ids = []

    for batch in tqdm_on_main(
        iterable=dataloader,
        total=len(dataloader),
    ):
        prompts = ["<provided>" + q for q in batch["hallu_query"]]

        output = chat_instance.run_batch(
            prompts,
            batch["image_path"],
        )

        print(
            output,
            batch["image_path"],
            prompts,
            flush=True,
        )

        outputs.extend(output)
        gt.extend(batch["gt"])
        ids.extend(batch["task_id"])

    if is_distributed:
        all_gt_labels = gather_processes(gt)
        all_outputs = gather_processes(outputs)
        all_ids = gather_processes(ids)
    else:
        all_gt_labels = gt
        all_outputs = outputs
        all_ids = ids

    print(f"Rank {global_rank}: Gathering done")

    test_ground_truth = {all_ids[i]: all_gt_labels[i] for i in range(len(all_ids))}
    test_output_exact_match = {all_ids[i]: all_outputs[i] for i in range(len(all_ids))}

    if args.hallu_type == "correct_think":
        subdir = "with_correct_hint_in_prompt"
    elif args.hallu_type == "think":
        subdir = "with_wrong_hint_in_prompt"
    else:
        subdir = "original_without_hint"
    
    if global_rank == 0:
        outdir = RADAGENT_RESULTS_DIR / subdir
        outdir.mkdir(parents=True, exist_ok=True)
        predictions_path = (
            outdir / f"ct_chat_predictions_{args.run_name}_{args.hallu_type}.json"
        )
        with open(predictions_path, "w") as f:
            json.dump(test_output_exact_match, f, indent=4)
            print("Predictions saved to", str(predictions_path))

        ground_truth_path = (
            outdir / f"ct_chat_ground_truth_{args.run_name}_{args.hallu_type}.json"
        )
        with open(ground_truth_path, "w") as f:
            json.dump(test_ground_truth, f, indent=4)
            print("Ground truth saved to", str(ground_truth_path))

        print("Predictions and ground truth saved")


if __name__ == "__main__":
    asyncio.run(main())

"""
This script runs full evaluation of CT-Chat on the CT-RATE report generation validation set.
It computes NLP metrics (BLEU, METEOR, ROUGE, CIDER), GREEN score, and multi-label classification metrics for CT pathology classification.
"""

import json
import os

from agents.art_dataset import (
    generate_report_generation_scenarios,
    generate_radchest_ct_scenarios,
)
from constants_and_path_utils import RADAGENT_RESULTS_DIR
from toolbox_src.CT_CHAT_main.ct_chat_full_model import CTChat_full_model
from pathlib import Path
from torch.utils.data import Dataset
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

    def __init__(self, mode="val", start_idx=0, end_idx=None):
        scenarios = generate_report_generation_scenarios(
            mode=mode, start_idx=start_idx, end_idx=end_idx
        )
        self.data = scenarios

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class RadChestCTReportGenerationDataset(Dataset):
    """
    Dataset for CT-RATE report generation task.
    Expects a json file with the original CT-RATE report generation format.
    Each sample returns a dictionary in a simplified format with the relevant fields
    (id, image, ground_truth).
    """

    def __init__(self, start_idx=0, end_idx=None):
        scenarios = generate_radchest_ct_scenarios(start_idx=start_idx, end_idx=end_idx)
        self.data = scenarios

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


if __name__ == "__main__":
    split_to_evaluate = "test"
    end_idx = None
    run_name = "base"
    dataset_name = "ctrate"
    print("Evaluating on split:", split_to_evaluate)
    if dataset_name == "ctrate":
        dataset = CTRAteReportGenerationDataset(mode=split_to_evaluate, end_idx=end_idx)
    elif dataset_name == "radchestct":
        dataset = RadChestCTReportGenerationDataset(end_idx=end_idx)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    is_distributed = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if is_distributed:
        if not dist.is_initialized():
            rank = int(os.environ.get("RANK", "0"))
            dist.init_process_group("nccl")
            device_id = int(os.environ["LOCAL_RANK"]) if is_distributed else 0
            torch.cuda.set_device(device_id)
            print(f"Rank {rank}: Process group initialized.")
            if dist.get_rank() == 0:
                print("Distributed training with", torch.cuda.device_count(), "GPUs")
        global_rank = dist.get_rank()  # This is your device ID
    else:
        global_rank = 0

    print(f"Rank {global_rank}: Starting...")

    # Baseline
    chat_instance = CTChat_full_model(device=f"cuda:{device_id}")
    chat_instance.model.eval().to(f"cuda:{device_id}")

    print(f"Rank {global_rank}: ==== Start Inference ====")
    dataloader = create_distributed_dataloader_if_needed(dataset, 4, shuffle=False)

    # Add barrier to ensure all processes are ready
    if is_distributed:
        dist.barrier()

    outputs = []
    gt = []
    ids = []
    for batch in tqdm_on_main(
        iterable=dataloader,
        total=len(dataloader),
    ):
        # print(batch['original_query'])
        output = chat_instance.run_batch(
            ["<provided>" + q for q in batch["original_query"]], batch["image_path"]
        )
        print(
            output,
            batch["image_path"],
            ["<provided>" + q for q in batch["original_query"]],
            flush=True,
        )
        outputs.extend(output)
        gt.extend(batch["gt"])
        ids.extend(batch["task_id"])

    # Gather results from all processes
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

    # Save predictions and ground truth to json files
    if global_rank == 0:
        outdir = RADAGENT_RESULTS_DIR / "baseline"
        outdir.mkdir(parents=True, exist_ok=True)
        with open(
            outdir / f"ct_chat_predictions_{run_name}_{split_to_evaluate}_{dataset_name}.json"
            ,"w"
        ) as f:
            json.dump(test_output_exact_match, f, indent=4)
            print(
                "Predictions saved to ",
                str(
                    outdir / f"ct_chat_predictions_{run_name}_{split_to_evaluate}_{dataset_name}.json"
                ),
            )
        with open(
            outdir / f"ct_chat_ground_truth__{run_name}_{split_to_evaluate}_{dataset_name}.json"
            ,"w"
        ) as f:
            json.dump(test_ground_truth, f, indent=4)
            print(
                "Ground truth saved to ",
                str(
                    outdir / f"ct_chat_ground_truth__{run_name}_{split_to_evaluate}_{dataset_name}.json"
                ),
            )
        print("Predictions and ground truth saved")

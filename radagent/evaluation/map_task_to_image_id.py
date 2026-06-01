"""
This code generates the mapping between Ids used in CT Chat generation and RadAgent generation.
"""

import argparse
import json
from pathlib import Path
from torch.utils.data import Dataset
from pathlib import Path

from agents.art_dataset import (
    generate_report_generation_scenarios,
    generate_radchest_ct_scenarios,
)


class CTRAteReportGenerationDataset(Dataset):
    def __init__(self, mode="val", start_idx=0, end_idx=None):
        scenarios = generate_report_generation_scenarios(
            mode=mode, start_idx=start_idx, end_idx=end_idx
        )
        self.data = scenarios

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RadChestCTReportGenerationDataset(Dataset):
    def __init__(self, start_idx=0, end_idx=None):
        scenarios = generate_radchest_ct_scenarios(
            start_idx=start_idx, end_idx=end_idx
        )
        self.data = scenarios

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_to_evaluate", type=str, default="test")
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--run_name", type=str, default="base")
    parser.add_argument(
        "--dataset", type=str, default="ctrate", choices=["ctrate", "radchestct"]
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional explicit output path. If omitted, a default filename is used.",
    )
    return parser.parse_args()


def normalize_image_id(image_id):
    """
    Makes image_id JSON serializable and consistent.

    Handles:
    - str
    - pathlib.Path
    - list/tuple of paths
    """
    if isinstance(image_id, Path):
        return str(image_id)
    if isinstance(image_id, (list, tuple)):
        return [str(p) if isinstance(p, Path) else p for p in image_id]
    return image_id

def to_image_id(image_id):
    if isinstance(image_id, (list, tuple)):
        if len(image_id) == 1:
            image_id = image_id[0]
        else:
            raise ValueError(f"Expected a single image path, got: {image_id}")

    name = Path(image_id).name
    return name.replace(".nii.gz", "")


def main():
    args = parse_args()

    split_to_evaluate = args.split_to_evaluate
    start_idx = 0 if args.start_idx is None else args.start_idx
    end_idx = args.end_idx
    dataset_name = args.dataset
    run_name = args.run_name

    if dataset_name == "ctrate":
        dataset = CTRAteReportGenerationDataset(
            mode=split_to_evaluate,
            start_idx=start_idx,
            end_idx=end_idx,
        )
    elif dataset_name == "radchestct":
        dataset = RadChestCTReportGenerationDataset(
            start_idx=start_idx,
            end_idx=end_idx,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    task_id_to_image_id = {}

    for sample in dataset:
        task_id = sample["task_id"]
        image_id = to_image_id(sample["image_path"])
        task_id_to_image_id[task_id] = image_id

    if args.output_file is not None:
        output_path = Path(args.output_file)
    else:
        output_path = (
            Path(__file__).parent
            / f"task_id_to_image_id_{run_name}_{split_to_evaluate}_{dataset_name}_start{start_idx}_end{end_idx}.json"
        )

    with open(output_path, "w") as f:
        json.dump(task_id_to_image_id, f, indent=4)

    print(f"Saved task_id -> image_id mapping to: {output_path}")
    print(f"Number of entries: {len(task_id_to_image_id)}")


if __name__ == "__main__":
    main()

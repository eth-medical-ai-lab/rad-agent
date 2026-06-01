import argparse
from collections import defaultdict
import os
import json
from pathlib import Path
import time
import numpy as np
import torch.distributed as dist
from evaluation.compute_metrics import (
    compute_nlp_metrics,
    compute_green_score,
    compute_multilabel_ct_classification_metrics,
)
from constants_and_path_utils import CT_RATE_ROOT, PATHOLOGIES_LIST
import pandas as pd


def generate_gt_list(ground_truth_report_path):
    """
    prepare ground truth list
    :param ground_truth_report_path:
    :return:
    """
    test_ground_truth = {}
    with open(ground_truth_report_path, "r") as file:
        all_reports = json.load(file)
    for sample in all_reports:
        id = sample["image"].split(".nii")[0]
        test_ground_truth[id] = sample["conversations"][1]["value"]

    return test_ground_truth


def generate_test_list_from_agent_outputs(generated_report_root_path):
    """
    prepare generated report list from agent outputs
    e.g. path = radagent/outputs/v1b/Mistral-Small-24B-Instruct-2501/v1_chunked
    has two subfolders 'trajectories' and 'fails'
    :param generated_report_root_path:
    :return:
    """
    generated_result = {}
    sucessful_reports_path = os.path.join(generated_report_root_path, "trajectory")
    failed_reports_path = os.path.join(generated_report_root_path, "fails")

    for path in os.listdir(sucessful_reports_path):
        with open(os.path.join(sucessful_reports_path, path), "r") as file:
            data = json.load(file)
        id = (
            path.replace(".json", "").split("_result")[0]
            if "_result" in path
            else path.replace(".json", "").split("_trajectory")[0]
        )
        try:
            if data[-1].get("reward", None) is not None:
                # New format with reward at the end
                last_message = json.loads(data[-2]["content"])
            else:
                # old format without reward
                last_message = json.loads(data[-1]["content"])
            if (
                last_message.get("action", "No") == "final_answer"
                and "answer" in last_message
            ):
                generated_result[id] = last_message["answer"]
                if not isinstance(generated_result[id], str):
                    print(
                        f"Warning: The generated answer for ID {id} is not a string. Got {generated_result[id]}"
                    )
                    generated_result[id] = ""
            else:
                generated_result[id] = ""
        except json.JSONDecodeError:
            generated_result[id] = ""
    for path in os.listdir(failed_reports_path):
        id = (
            path.replace(".json", "").split("_result")[0]
            if "_result" in path
            else path.replace(".json", "").split("_trajectory")[0]
        )
        generated_result[id] = "Failed"
    return generated_result


def test_if_paired_data(test_ground_truth, generated_result):
    missing_reports = set(test_ground_truth.keys()) - set(generated_result.keys())
    missing_reports_id = list(missing_reports) if missing_reports is not None else None

    print(len(missing_reports_id), "cases are missing!")

    if missing_reports_id:
        for id in missing_reports_id:
            test_ground_truth.pop(id)
    return test_ground_truth, generated_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default=0)
    parser.add_argument(
        "--is_ct_chat_eval",
        action="store_true",
        help="Whether to evaluate the CT-Chat outputs. If set, it will load the CT-Chat specific ground truth and predictions instead of the ones from the generated report folder.",
    )
    parser.add_argument(
        "--ct_chat_predictions_path",
        type=str,
        default="",
        help="Path to CT-Chat predictions JSON file (used only if --is_ct_chat_eval is set)",
    )
    parser.add_argument(
        "--ct_chat_ground_truth_path",
        type=str,
        default="",
        help="Path to CT-Chat ground truth JSON file (used only if --is_ct_chat_eval is set)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which split to evaluate on (default: val)",
    )

    args = parser.parse_args()
    generated_report_root_path = args.output_folder
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    match args.split:
        case "val":
            ground_truth_report_path = (
                CT_RATE_ROOT / "labels/report_generation/report_generation_valid.json"
            )
        case "test":
            ground_truth_report_path = (
                CT_RATE_ROOT / "labels/report_generation/report_generation_test.json"
            )

    if not args.is_ct_chat_eval:
        test_ground_truth = generate_gt_list(ground_truth_report_path)
        test_generated_output = generate_test_list_from_agent_outputs(
            generated_report_root_path
        )

    # For CT-Chat eval
    else:
        with open(
            args.ct_chat_ground_truth_path,
            "r",
        ) as f:
            test_ground_truth = json.load(f)
        with open(
            args.ct_chat_predictions_path,
            "r",
        ) as f:
            test_generated_output = json.load(f)

    print(len(test_ground_truth), len(test_generated_output))
    # Drop all the unpaired data (to allow for computing metrics on partial outputs)
    test_ground_truth, test_generated_output = test_if_paired_data(
        test_ground_truth, test_generated_output
    )

    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    print("Is distributed:", is_distributed, flush=True)
    if is_distributed:
        if not dist.is_initialized():
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            print(f"Rank {rank}: Process group initialized.")
            if dist.get_rank() == 0:
                print(
                    "Distributed training with",
                    int(os.environ.get("WORLD_SIZE", 1)),
                    "GPUs",
                )
        global_rank = dist.get_rank()  # This is your device ID
    else:
        global_rank = 0

    device_id = global_rank if is_distributed else 0

    # Add barrier to ensure all processes are ready
    if is_distributed:
        dist.barrier()

    print("##### GREEN SCORE #####")
    all_green_results = compute_green_score(test_ground_truth, test_generated_output)

    all_results = {}
    if global_rank == 0:
        all_classification_results = compute_multilabel_ct_classification_metrics(
            test_ground_truth, test_generated_output, batch_size=16
        )
        classification_results, detailed_predictions_results = (
            all_classification_results
        )
        average_green_score, green_score_list, green_df = all_green_results

        all_results.update(classification_results)
        all_results.update(average_green_score)
        print("Starting NLP metrics computation")
        average_nlp_metrics = compute_nlp_metrics(
            test_ground_truth, test_generated_output
        )[0]
        all_results.update(average_nlp_metrics)
        print("Finished NLP metrics computation")
        print(average_nlp_metrics)
        print(average_green_score)
        print(classification_results)

        t3 = time.time()

        with open(os.path.join(args.output_folder, "results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
        with open(
            os.path.join(args.output_folder, "green_score_results.json"), "w"
        ) as f:
            json.dump(green_score_list, f, indent=4)
        green_df.to_csv(
            os.path.join(args.output_folder, "green_score_detailed_results.csv"),
            index=False,
        )
        # Save all_resutls to a json file

        ids = []
        gt_report = []
        generated_report = []
        image_id = []
        df_classification_results = pd.DataFrame(
            detailed_predictions_results["pred"], columns=PATHOLOGIES_LIST
        )
        result_dict = defaultdict(list)

        for i, id in enumerate(test_ground_truth.keys()):
            result_dict["id"].append(i)
            result_dict["image_id"].append(id)
            result_dict["gt_report"].append(test_ground_truth[id])
            result_dict["generated_report"].append(test_generated_output[id])
            idx = np.where(np.asarray(detailed_predictions_results["volume_id"]) == id)[
                0
            ][0]
            for j, p in enumerate(PATHOLOGIES_LIST):
                result_dict[f"pred_{p}"].append(
                    int(detailed_predictions_results["pred"][idx, j])
                )
                result_dict[f"gt_{p}"].append(
                    int(detailed_predictions_results["gt"][idx, j])
                )
                result_dict[f"is_correct_{p}"].append(
                    int(detailed_predictions_results["equal"][idx, j])
                )
            result_dict["accuracy_18findings"].append(
                sum(
                    [
                        int(detailed_predictions_results["equal"][idx][j])
                        for j in range(len(PATHOLOGIES_LIST))
                    ]
                )
                / len(PATHOLOGIES_LIST)
            )
        result_dict["green_score"].extend(green_score_list["GREEN"])
        df = pd.DataFrame(result_dict)
        df.to_csv(os.path.join(args.output_folder, "detailed_results.csv"), index=False)

        print(f'Results saved to {os.path.join(args.output_folder, "results.json")}')

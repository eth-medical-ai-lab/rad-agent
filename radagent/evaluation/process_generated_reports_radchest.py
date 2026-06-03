import argparse
import os
import json
from pathlib import Path
import numpy as np

from evaluation.compute_metrics import compute_multilabel_ct_predictions
from constants_and_path_utils import RADCHEST_CT_ROOT, PATHOLOGIES_LIST
import pandas as pd


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
            path.split("_result")[0]
            if "_result" in path
            else path.split("_trajectory")[0]
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
            path.split("_result")[0]
            if "_result" in path
            else path.split("_trajectory")[0]
        )
        generated_result[id] = "Failed"
    return generated_result


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

    args = parser.parse_args()
    generated_report_root_path = args.output_folder
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    gt_df = pd.read_csv(RADCHEST_CT_ROOT / "broad_labels.csv")

    if not args.is_ct_chat_eval:
        test_generated_output = generate_test_list_from_agent_outputs(
            generated_report_root_path
        )

    # ## for CT-Chat eval
    else:
        with open(
            args.ct_chat_predictions_path,
            "r",
        ) as f:
            test_generated_output = json.load(f)

    all_results = {}

    classification_predictions = compute_multilabel_ct_predictions(
        test_generated_output, batch_size=16
    )

    rename_dict = {p: f"gt_{p}" for p in PATHOLOGIES_LIST}
    rename_dict["NoteAcc_DEID"] = "image_id"
    gt_df.rename(columns=rename_dict, inplace=True)
    print("GT DF columns after rename: ", gt_df.columns)
    print("Classification predictions columns: ", classification_predictions.columns)
    merged_df = pd.merge(gt_df, classification_predictions, on="image_id", how="inner")
    merged_df["id"] = np.arange(len(merged_df))
    merged_df.to_csv(
        os.path.join(args.output_folder, "detailed_results.csv"), index=False
    )

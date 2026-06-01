import json
import random
from typing import Dict, List, Any
from constants_and_path_utils import CT_RATE_ROOT, path_parse, RADCHEST_CT_ROOT
import pandas as pd

MAX_TURNS = 60


def generate_radchest_ct_scenarios(start_idx=0, end_idx=None) -> List[Dict[str, Any]]:
    gt_df = pd.read_csv(RADCHEST_CT_ROOT / "broad_labels.csv")
    scenarios = []
    for i, row in gt_df.iterrows():
        if i < start_idx:
            continue
        if end_idx is not None and i > end_idx:
            break
        image_id = row["NoteAcc_DEID"]
        image_path = str(RADCHEST_CT_ROOT / "NIFTI" / f"{image_id}.nii.gz")
        gt_report = ""
        orig_query = "<image>\nCan you generate the report for the following chest CT scan?<report_generation>"
        query = f"{orig_query}. The image file path is {image_path}."
        scenarios.append(
            {
                "task": query,
                "task_id": image_id,
                "gt": gt_report,
                "difficulty": 1,
                "image_path": image_path,
                "original_query": orig_query,
            }
        )
    return scenarios


def generate_report_generation_scenarios(
    mode="val", start_idx=0, end_idx=None, shuffle=False
) -> List[Dict[str, Any]]:
    if mode == "val":
        path = CT_RATE_ROOT / "labels/report_generation/report_generation_valid.json"
    elif (mode == "train") or (mode == "inference_train"):
        path = CT_RATE_ROOT / "labels/report_generation/report_generation_train.json"
    elif mode == "test":
        path = CT_RATE_ROOT / "labels/report_generation/report_generation_test.json"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    scenarios = []
    for i, data_point in enumerate(data):
        if i < start_idx:
            continue
        if end_idx is not None and i > end_idx:
            break
        user_query = data_point["conversations"][0]["value"]
        image_path = path_parse(data_point["image"])

        query = f"{user_query}. The image file path is {image_path}. "
        scenarios.append(
            {
                "task": query,
                "task_id": data_point["id"],
                "gt": data_point["conversations"][1]["value"],
                "difficulty": 1,
                "image_path": image_path,
                "original_query": user_query,
            }
        )

    if shuffle:
        random.shuffle(scenarios)
    return scenarios


def generate_vqa_scenarios(
    mode="val", start_idx=0, end_idx=None, shuffle=False
) -> List[Dict[str, Any]]:
    if mode == "val":
        df = pd.read_csv(
            CT_RATE_ROOT / "labels/multiple_choice/multiple_choice_valid.csv"
        )
    elif mode == "train":
        df = pd.read_csv(
            CT_RATE_ROOT / "labels/multiple_choice/multiple_choice_train.csv"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    end = end_idx if end_idx < len(df) else len(df)
    data = df.loc[start_idx:end]
    scenarios = []
    for i, data_point in data.iterrows():
        user_query = data_point["question"]
        image_path = path_parse(data_point["image_id"])
        query = f"{user_query}. The image file path is {image_path}. "
        scenarios.append(
            {
                "task": query,
                "task_id": data_point["qid"],
                "gt": data_point["answer"],
                "difficulty": 1,
                "image_path": image_path,
            }
        )
    if shuffle:
        random.shuffle(scenarios)
    return scenarios


def generate_hallu_generation_scenario(hallu_type, start_idx=0, end_idx=None):
    df = pd.read_csv(CT_RATE_ROOT / 'labels/hallucination_detection_dataset_long.csv')
    if end_idx is None:
        end_idx = len(df)
    df = df.iloc[start_idx:end_idx]
    scenarios = []
    for _, row in df.iterrows():
        image_path = path_parse(row['VolumeName'])
        if hallu_type == 'think':
            user_query = row['hallu_query_think']
        elif hallu_type == 'sure':
            user_query = row['hallu_query_sure']
        elif hallu_type == 'orig':
            user_query = row['query']
        elif hallu_type == 'correct_think':
            user_query = row['hallu_query_correct_think']
        elif hallu_type == 'correct_sure':
            user_query = row['hallu_query_correct_sure']
        else:
            raise ValueError("hallu_type must be either 'orig', 'think' or 'sure'")
        
        query = f"{user_query}. The image file path is {image_path}. "

        scenario = {
            'id': row['id'],
            'task_id': row['id'],
            'gt': row['gt_report'],
            'image_path': image_path,
            'query': query,
            'task': query,
            "difficulty": 1,
            'gt_report': row['gt_report'],
            'original_query': row['query'],
            'hallu_query': user_query
        }

        scenarios.append(scenario)
    return scenarios


if __name__ == "__main__":
    scenarios_collection = generate_report_generation_scenarios()
    print(scenarios_collection)

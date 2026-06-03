"""
Code to get the absolute F1 scores without further bootstrapping analysis.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from constants_and_path_utils import RADAGENT_REPO_ROOT, RADAGENT_RESULTS_DIR
from evaluation.plotting_utils import BASELINE_NAME, PATHOLOGIES_LIST

RADAGENT_NAME = "RadAgent"

# Provided in repo for convenience
TASK_ID_IMAGE_ID_MAPS = {
    "radchest": str(
        RADAGENT_REPO_ROOT
        / "radagent/outputs/task_to_id_maps/task_id_to_image_id_base_test_radchestct_start0_endNone.json"
    ),
    "ct_rate_test": str(
        RADAGENT_REPO_ROOT
        / "radagent/outputs/task_to_id_maps/task_id_to_image_id_base_test_ctrate_start0_endNone.json"
    ),
    "ct_rate_val": str(
        RADAGENT_REPO_ROOT
        / "radagent/outputs/task_to_id_maps/task_id_to_image_id_base_val_ctrate_start0_end1000.json"
    ),
}

# Set to output directories of running inference on the respective datasets
# Use same paths as in main_plot_training_no_ablation.py
DATASET_PATHS = {
    "radchest": {
        RADAGENT_NAME: str(
            RADAGENT_RESULTS_DIR / "<radagent_radchest_results/detailed_results.csv>"
        ),
        BASELINE_NAME: str(
            RADAGENT_RESULTS_DIR / "<ct_chat_radchest_results/detailed_results.csv>"
        ),
        "title": "RadChestCT",
    },
    "ct_rate_test": {
        RADAGENT_NAME: str(
            RADAGENT_RESULTS_DIR / "<radagent_ct_rate_test_results/detailed_results.csv>"
        ),
        BASELINE_NAME: str(
            RADAGENT_RESULTS_DIR / "<ct_chat_ct_rate_test_results/detailed_results.csv>"
        ),
        "title": "CT-RATE Test Set",
    },
    "ct_rate_val": {
        RADAGENT_NAME: str(
            RADAGENT_RESULTS_DIR / "<radagent_ct_rate_val_results/detailed_results.csv>"
        ),
        BASELINE_NAME: str(
            RADAGENT_RESULTS_DIR / "<ct_chat_ct_rate_val_results/detailed_results.csv>"
        ),
        "title": "CT-RATE Validation Set",
    },
}


def normalize_path(path_str: str) -> Path:
    if path_str.startswith("/"):
        return Path(path_str)
    return Path("/" + path_str)


def print_separator(char: str = "=", width: int = 100) -> None:
    print(char * width)


def merge_image_id(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    with open(TASK_ID_IMAGE_ID_MAPS[dataset_key], "r") as f:
        task_to_image = json.load(f)

    df = df.rename(columns={"image_id": "task_id"})
    mapping_df = pd.DataFrame(task_to_image.items(), columns=["task_id", "image_id"])
    df = df.merge(mapping_df, on="task_id", how="left")

    missing = df[df["image_id"].isna()]
    if len(missing) > 0:
        raise ValueError(f"ID merge left {len(missing)} missing image_ids for {dataset_key}")

    return df


def load_result_maps(dataset_key: str, dataset_config: dict) -> dict[str, pd.DataFrame]:
    df_maps: dict[str, pd.DataFrame] = {}

    for model_name in (BASELINE_NAME, RADAGENT_NAME):
        csv_path = normalize_path(dataset_config[model_name])
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {model_name}: {csv_path}")

        df = pd.read_csv(csv_path)

        if model_name == BASELINE_NAME:
            df = merge_image_id(df, dataset_key=dataset_key)

        df_maps[model_name] = df

    return df_maps


def align_on_shared_ids(
    baseline_df: pd.DataFrame, radagent_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "id" not in baseline_df.columns or "id" not in radagent_df.columns:
        raise ValueError("Both dataframes need an 'id' column to align rows")

    shared_ids = set(baseline_df["id"].dropna()) & set(radagent_df["id"].dropna())
    if not shared_ids:
        raise ValueError("No shared ids found between baseline and RadAgent")

    baseline_df = baseline_df[baseline_df["id"].isin(shared_ids)].copy()
    radagent_df = radagent_df[radagent_df["id"].isin(shared_ids)].copy()

    baseline_df = baseline_df.sort_values("id").reset_index(drop=True)
    radagent_df = radagent_df.sort_values("id").reset_index(drop=True)

    if not baseline_df["id"].equals(radagent_df["id"]):
        raise ValueError("Aligned ids still do not match after sorting")

    return baseline_df, radagent_df


def compute_f1_scores(df: pd.DataFrame) -> pd.Series:
    results = {}

    for pathology in PATHOLOGIES_LIST:
        gt_col = f"gt_{pathology}"
        pred_col = f"pred_{pathology}"

        if gt_col not in df.columns or pred_col not in df.columns:
            continue

    gt_all = df[[f"gt_{p}" for p in PATHOLOGIES_LIST if f"gt_{p}" in df.columns]].values
    pred_all = df[[f"pred_{p}" for p in PATHOLOGIES_LIST if f"pred_{p}" in df.columns]].values

    results["Macro-F1"] = f1_score(gt_all, pred_all, average="macro", zero_division=0)
    results["Micro-F1"] = f1_score(gt_all, pred_all, average="micro", zero_division=0)

    return pd.Series(results)


def evaluate_dataset(dataset_key: str, dataset_config: dict) -> pd.DataFrame:
    df_maps = load_result_maps(dataset_key, dataset_config)

    baseline_df, radagent_df = align_on_shared_ids(
        df_maps[BASELINE_NAME],
        df_maps[RADAGENT_NAME],
    )

    baseline_scores = compute_f1_scores(baseline_df)
    radagent_scores = compute_f1_scores(radagent_df)

    out_df = pd.concat(
        [
            baseline_scores.rename(BASELINE_NAME),
            radagent_scores.rename(RADAGENT_NAME),
        ],
        axis=1,
    )
    out_df["Diff"] = out_df[RADAGENT_NAME] - out_df[BASELINE_NAME]

    return out_df.sort_index()


def main() -> None:
    print("Computing plain F1 scores without bootstrapping")
    print_separator()

    all_results = {}

    for dataset_key, dataset_config in DATASET_PATHS.items():
        print(f"\nDataset: {dataset_key} | {dataset_config['title']}")
        print_separator("-")

        result_df = evaluate_dataset(dataset_key, dataset_config)
        all_results[dataset_key] = result_df

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(result_df.round(4))

    print("\nDone.")


if __name__ == "__main__":
    main()

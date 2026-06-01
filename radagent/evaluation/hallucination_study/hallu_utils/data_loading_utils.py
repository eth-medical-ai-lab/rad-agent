"""
Load and align the CSV inputs used throughout the hallucination study.

These helpers read the original, wrong-hint, correct-hint, and corrupted-report
outputs, then merge them into consistent dataframes for downstream metrics.
"""

from typing import Dict, Tuple

import pandas as pd

from .paths_utils import get_input_paths


def align_multiple_dfs_by_vol_name(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Keep only cases shared across all provided systems and sort by VolumeName
    so rows are aligned across systems.
    """
    if not dfs:
        return {}

    common_ids = None
    for df in dfs.values():
        ids = set(df["VolumeName"])
        common_ids = ids if common_ids is None else common_ids.intersection(ids)

    if common_ids is None:
        return {}

    aligned = {}
    for system_name, df in dfs.items():
        filtered = df[df["VolumeName"].isin(common_ids)].copy()
        filtered = filtered.sort_values("VolumeName").reset_index(drop=True)
        aligned[system_name] = filtered

    return aligned


def keep_only_same_image_ids(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Keep only shared VolumeName cases across all systems.
    """
    return align_multiple_dfs_by_vol_name(dfs)


def load_data(system_type: str, injected_prompt_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the input data.
    """
    gt_path, hallu_wrong_path, hallu_correct_path, hallu_orig_path = get_input_paths(
        system_type,
        injected_prompt_type,
    )

    gt_df = pd.read_csv(gt_path)
    hallu_wrong = pd.read_csv(hallu_wrong_path)
    hallu_correct = pd.read_csv(hallu_correct_path)
    hallu_orig = pd.read_csv(hallu_orig_path)

    if system_type == "CT-Chat":
        gt_df["image_id"] = gt_df["id"].apply(lambda x: x.replace(".nii.gz", ""))
    else:
        gt_df["image_id"] = gt_df["VolumeName"].apply(lambda x: x.replace(".nii.gz", ""))

    gt_df["VolumeName"] = gt_df["VolumeName"].apply(lambda x: x.replace(".nii.gz", ""))
    return gt_df, hallu_wrong, hallu_correct, hallu_orig


def build_full_dataframe(
    gt_df: pd.DataFrame,
    hallu_wrong: pd.DataFrame,
    hallu_correct: pd.DataFrame,
    hallu_orig: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge outcomes of the different detailed_results tables from the prompt injection experiments.
    """
    pred_path_columns = [col for col in hallu_orig.columns if col.startswith("pred_")]
    gt_path_columns = [col for col in hallu_orig.columns if col.startswith("gt_")]

    orig_columns = ["image_id", "generated_report"] + pred_path_columns + gt_path_columns
    if "admits_using_hint" in hallu_orig.columns:
        orig_columns.append("admits_using_hint")

    wrong_columns = ["image_id", "generated_report"] + pred_path_columns
    if "admits_using_hint" in hallu_wrong.columns:
        wrong_columns.append("admits_using_hint")

    correct_columns = ["image_id", "generated_report"] + pred_path_columns
    if "admits_using_hint" in hallu_correct.columns:
        correct_columns.append("admits_using_hint")

    hallu_orig = hallu_orig[orig_columns].copy()
    hallu_wrong = hallu_wrong[wrong_columns].copy()
    hallu_correct = hallu_correct[correct_columns].copy()

    hallu_orig.rename(
        columns={
            col: f"{col}_orig"
            for col in pred_path_columns + ["generated_report", "admits_using_hint"]
            if col in hallu_orig.columns
        },
        inplace=True,
    )

    hallu_wrong.rename(
        columns={
            col: f"{col}_hallu_wrong"
            for col in pred_path_columns + ["generated_report", "admits_using_hint"]
            if col in hallu_wrong.columns
        },
        inplace=True,
    )

    hallu_correct.rename(
        columns={
            col: f"{col}_hallu_correct"
            for col in pred_path_columns + ["generated_report", "admits_using_hint"]
            if col in hallu_correct.columns
        },
        inplace=True,
    )

    df_full = pd.merge(
        gt_df[
            [
                "image_id",
                "pathology_to_flip",
                "hallu_query_think",
                "hallu_query_correct_think",
                "hallu_query_sure",
                "query",
                "VolumeName",
            ]
        ],
        hallu_orig,
        on="image_id",
        how="inner",
    )

    df_full = pd.merge(
        df_full,
        hallu_wrong,
        on="image_id",
        how="inner",
    )

    df_full = pd.merge(
        df_full,
        hallu_correct,
        on="image_id",
        how="inner",
    )

    return df_full

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Any
import re
from pathlib import Path
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from typing import Tuple, Any, Literal, Optional, Dict
from dataclasses import dataclass
from functools import partial
from scipy.stats import permutation_test
from sklearn.metrics import f1_score, recall_score
from constants_and_path_utils import PATHOLOGIES_LIST

# Metrics without graound truth
DIRECT_METRIC_COLUMNS: dict[str, str] = {
    "AbnormalityJudge-F1": "Qwen_Qwen3-30B-A3B-Thinking-2507_abnormal_f1",
    "ChecklistAdherenceJudge": "Qwen_Qwen3-30B-A3B-Thinking-2507_checklist_adherence",
    "ToolSequenceCoherenceJudge": "Qwen_Qwen3-30B-A3B-Thinking-2507_tool_seq_coherence",
    "NumUniqueTools": "num_tools",
}

# Metrics that need to be aggregation in computation over a sample
AGGREGATE_METRICS: set[str] = {
    "Macro-F1",
    "Micro-F1",
    "Macro-Sensitivity",
    "Micro-Sensitivity",
    "Macro-Specificity",
    "Micro-Specificity",
}

BASELINE_NAME = "CT-Chat report generation"

@dataclass(frozen=True)
class MetricInputs:
    kind: Literal["direct_mean_column", "computed_metric"]
    baseline_values: np.ndarray
    model_values: np.ndarray
    gt: np.ndarray | None = None


def _to_python_float(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"Expected scalar result, got shape {arr.shape}.")
    return float(arr.reshape(-1)[0])


def _compute_specificity_macro_micro(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
) -> tuple[float, float]:
    specificities: list[float] = []
    total_tn = 0
    total_fp = 0

    for col_idx in range(gt_arr.shape[1]):
        gt_col = gt_arr[:, col_idx]
        pred_col = pred_arr[:, col_idx]

        tn = int(((gt_col == 0) & (pred_col == 0)).sum())
        fp = int(((gt_col == 0) & (pred_col == 1)).sum())

        total_tn += tn
        total_fp += fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    macro_spec = float(np.mean(specificities)) if specificities else 0.0
    micro_spec = float(total_tn / (total_tn + total_fp)) if (total_tn + total_fp) > 0 else 0.0
    return macro_spec, micro_spec


def _compute_metric(
    metric_name: str,
    gt: np.ndarray,
    pred: np.ndarray,
) -> float:
    for pathology in PATHOLOGIES_LIST:
        if metric_name == pathology:
            return float(f1_score(gt, pred, zero_division=0.0))

        if metric_name == f"{pathology}_Sensitivity":
            return float(recall_score(gt, pred, pos_label=1, zero_division=0.0))

        if metric_name == f"{pathology}_Specificity":
            return float(recall_score(gt, pred, pos_label=0, zero_division=0.0))

    if metric_name == "Macro-F1":
        return float(f1_score(gt, pred, average="macro", zero_division=0.0))

    if metric_name == "Micro-F1":
        return float(f1_score(gt, pred, average="micro", zero_division=0.0))

    if metric_name == "Macro-Sensitivity":
        return float(recall_score(gt, pred, average="macro", zero_division=0.0))

    if metric_name == "Micro-Sensitivity":
        return float(recall_score(gt, pred, average="micro", zero_division=0.0))

    if metric_name == "Macro-Specificity":
        macro_spec, _ = _compute_specificity_macro_micro(gt, pred)
        return float(macro_spec)

    if metric_name == "Micro-Specificity":
        _, micro_spec = _compute_specificity_macro_micro(gt, pred)
        return float(micro_spec)

    raise ValueError(f"Unsupported metric: {metric_name!r}")


def _mean_diff_statistic(
    baseline_sample: np.ndarray,
    model_sample: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """
    Mean difference statistic for direct numeric per case columns.

    Returns:
        model mean - baseline mean
    """
    baseline_arr = np.asarray(baseline_sample, dtype=float)
    model_arr = np.asarray(model_sample, dtype=float)
    return np.asarray(
        model_arr.mean(axis=axis) - baseline_arr.mean(axis=axis),
        dtype=float,
    )

def _compute_metric_from_obs_last(
    metric_name: str,
    gt_obs_last: np.ndarray,
    pred_obs_last: np.ndarray,
) -> float:
    """
    Compute one metric when the observation axis is the last axis.

    Supported shapes:
        binary per pathology:
            gt_obs_last.shape == (n_obs,)
            pred_obs_last.shape == (n_obs,)

        multilabel aggregate:
            gt_obs_last.shape == (n_labels, n_obs)
            pred_obs_last.shape == (n_labels, n_obs)
    """
    if gt_obs_last.ndim == 1:
        return _compute_metric(metric_name, gt_obs_last, pred_obs_last)

    if gt_obs_last.ndim == 2:
        # sklearn expects (n_obs, n_labels) for multilabel arrays
        return _compute_metric(metric_name, gt_obs_last.T, pred_obs_last.T)

    raise ValueError(
        f"Unsupported dimensionality for metric computation: gt.ndim={gt_obs_last.ndim}"
    )


def _compute_metric_over_permuted_samples(
    metric_name: str,
    gt: np.ndarray,
    pred_sample: np.ndarray,
    axis: int,
) -> np.ndarray:
    """
    Compute metric values for observed or batched permuted samples.

    SciPy's vectorized permutation_test moves the observation axis around.
    This helper standardizes everything to:
        observations on the last axis

    Then:
        gt_obs_last has shape
            (n_obs,) for binary metrics
            (n_labels, n_obs) for aggregate multilabel metrics

        pred_obs_last has shape
            observed case: same as gt_obs_last
            batched null:  (*batch_dims, ...) + gt_obs_last.shape

    Returns:
        scalar np.ndarray for observed call
        array over batch dimensions for batched calls
    """
    gt_arr = np.asarray(gt)
    pred_arr = np.asarray(pred_sample)

    gt_obs_last = np.moveaxis(gt_arr, 0, -1)
    pred_obs_last = np.moveaxis(pred_arr, axis, -1)

    if pred_obs_last.ndim == gt_obs_last.ndim:
        return np.asarray(
            _compute_metric_from_obs_last(metric_name, gt_obs_last, pred_obs_last),
            dtype=float,
        )

    if pred_obs_last.ndim < gt_obs_last.ndim:
        raise ValueError(
            "Predicted sample has fewer dimensions than ground truth after axis normalization."
        )

    batch_shape = pred_obs_last.shape[: pred_obs_last.ndim - gt_obs_last.ndim]
    pred_flat = pred_obs_last.reshape((-1,) + gt_obs_last.shape)

    scores = np.empty(pred_flat.shape[0], dtype=float)
    for idx in range(pred_flat.shape[0]):
        scores[idx] = _compute_metric_from_obs_last(
            metric_name,
            gt_obs_last,
            pred_flat[idx],
        )

    return scores.reshape(batch_shape)


def _computed_metric_diff_statistic(
    baseline_sample: np.ndarray,
    model_sample: np.ndarray,
    *,
    metric_name: str,
    gt: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """
    Difference statistic for computed metrics.

    Returns:
        model score - baseline score
    """
    baseline_scores = _compute_metric_over_permuted_samples(
        metric_name=metric_name,
        gt=gt,
        pred_sample=baseline_sample,
        axis=axis,
    )
    model_scores = _compute_metric_over_permuted_samples(
        metric_name=metric_name,
        gt=gt,
        pred_sample=model_sample,
        axis=axis,
    )

    return np.asarray(model_scores - baseline_scores, dtype=float)


def _run_permutation_test_for_metric(
    metric_name: str,
    metric_inputs: MetricInputs,
    *,
    n_resampled: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    baseline_values = metric_inputs.baseline_values
    model_values = metric_inputs.model_values

    if len(model_values) == 0:
        return np.nan, np.nan, np.nan, np.nan

    if metric_inputs.kind == "direct_mean_column":
        baseline_score = float(np.mean(baseline_values))
        model_score = float(np.mean(model_values))
        statistic = _mean_diff_statistic
    else:
        if metric_inputs.gt is None:
            raise ValueError(f"Ground truth is required for computed metric {metric_name!r}.")

        gt = np.asarray(metric_inputs.gt)
        baseline_score = _compute_metric(metric_name, gt, baseline_values)
        model_score = _compute_metric(metric_name, gt, model_values)
        statistic = partial(
            _computed_metric_diff_statistic,
            metric_name=metric_name,
            gt=gt,
        )

    result = permutation_test(
        data=(baseline_values, model_values),
        statistic=statistic,
        permutation_type="samples",
        n_resamples=n_resampled,
        alternative="two-sided",
        vectorized=True,
        axis=0,
        rng=rng,
    )

    observed_difference = _to_python_float(result.statistic)
    p_value = _to_python_float(result.pvalue)

    return model_score, baseline_score, observed_difference, p_value

def _extract_direct_metric_inputs(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    metric_name: str,
) -> MetricInputs | None:
    if metric_name not in DIRECT_METRIC_COLUMNS:
        return None

    col = DIRECT_METRIC_COLUMNS[metric_name]
    if col not in df.columns or col not in baseline_df.columns:
        raise ValueError(f"Required column {col!r} not found for metric {metric_name!r}.")

    model_values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    baseline_values = pd.to_numeric(baseline_df[col], errors="coerce").to_numpy(dtype=float)

    valid_mask = ~np.isnan(model_values) & ~np.isnan(baseline_values)

    return MetricInputs(
        kind="direct_mean_column",
        baseline_values=baseline_values[valid_mask],
        model_values=model_values[valid_mask],
        gt=None,
    )


def _extract_pathology_metric_inputs(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    metric_name: str,
) -> MetricInputs | None:
    for pathology in PATHOLOGIES_LIST:
        matches_pathology_metric = (
            metric_name == pathology
            or metric_name == f"{pathology}_Sensitivity"
            or metric_name == f"{pathology}_Specificity"
        )
        if not matches_pathology_metric:
            continue

        gt_col = f"gt_{pathology}"
        pred_col = f"pred_{pathology}"

        if gt_col not in baseline_df.columns:
            raise ValueError(f"Missing ground truth column {gt_col!r}.")
        if pred_col not in df.columns or pred_col not in baseline_df.columns:
            raise ValueError(f"Missing prediction column {pred_col!r}.")

        gt = pd.to_numeric(baseline_df[gt_col], errors="coerce").to_numpy(dtype=float)
        baseline_pred = pd.to_numeric(baseline_df[pred_col], errors="coerce").to_numpy(dtype=float)
        model_pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)

        valid_mask = ~np.isnan(gt) & ~np.isnan(baseline_pred) & ~np.isnan(model_pred)

        return MetricInputs(
            kind="computed_metric",
            baseline_values=baseline_pred[valid_mask].astype(int),
            model_values=model_pred[valid_mask].astype(int),
            gt=gt[valid_mask].astype(int),
        )

    return None


def _extract_aggregate_metric_inputs(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    metric_name: str,
) -> MetricInputs | None:
    if metric_name not in AGGREGATE_METRICS:
        return None

    gt_cols = [f"gt_{pathology}" for pathology in PATHOLOGIES_LIST]
    pred_cols = [f"pred_{pathology}" for pathology in PATHOLOGIES_LIST]

    missing_gt = [col for col in gt_cols if col not in baseline_df.columns]
    missing_model_pred = [col for col in pred_cols if col not in df.columns]
    missing_baseline_pred = [col for col in pred_cols if col not in baseline_df.columns]

    if missing_gt:
        raise ValueError(f"Missing GT columns for metric {metric_name!r}: {missing_gt}")
    if missing_model_pred:
        raise ValueError(f"Missing model prediction columns for metric {metric_name!r}: {missing_model_pred}")
    if missing_baseline_pred:
        raise ValueError(f"Missing baseline prediction columns for metric {metric_name!r}: {missing_baseline_pred}")

    gt = baseline_df[gt_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    baseline_pred = baseline_df[pred_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    model_pred = df[pred_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    valid_mask = (
        ~np.isnan(gt).any(axis=1)
        & ~np.isnan(baseline_pred).any(axis=1)
        & ~np.isnan(model_pred).any(axis=1)
    )

    return MetricInputs(
        kind="computed_metric",
        baseline_values=baseline_pred[valid_mask].astype(int),
        model_values=model_pred[valid_mask].astype(int),
        gt=gt[valid_mask].astype(int),
    )


def _get_metric_inputs(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    metric_name: str,
) -> MetricInputs:
    direct_inputs = _extract_direct_metric_inputs(df, baseline_df, metric_name)
    if direct_inputs is not None:
        return direct_inputs

    pathology_inputs = _extract_pathology_metric_inputs(df, baseline_df, metric_name)
    if pathology_inputs is not None:
        return pathology_inputs

    aggregate_inputs = _extract_aggregate_metric_inputs(df, baseline_df, metric_name)
    if aggregate_inputs is not None:
        return aggregate_inputs

    raise ValueError(f"Unsupported variable_to_compute: {metric_name!r}")



def assess_signficance(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    variables_to_compute: list[str],
    names: list[str],
    n_resampled: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    df, baseline_df = align_dfs_by_id(df, baseline_df)
    rng = np.random.default_rng(seed)

    baseline_name = names[0] 
    model_name = names[1]

    results: list[dict[str, str | int | float | bool]] = []

    for metric_name in variables_to_compute:
        try:
            metric_inputs = _get_metric_inputs(df, baseline_df, metric_name)
        except ValueError as e: 
            print(f"Metrics {metric_name} skipped, due to {e}")
            continue
        n_pairs = int(len(metric_inputs.model_values))

        if n_pairs == 0:
            results.append(
                {
                    "variable": metric_name,
                    "n_pairs": 0,
                    model_name: np.nan,
                    baseline_name: np.nan,
                    "observed_difference": np.nan,
                    "p_value": np.nan,
                    "significant_at_0_05": False,
                }
            )
            continue

        model_score, baseline_score, observed_difference, p_value = _run_permutation_test_for_metric(
            metric_name,
            metric_inputs,
            n_resampled=n_resampled,
            rng=rng,
        )

        results.append(
            {
                "variable": metric_name,
                "n_pairs": n_pairs,
                model_name: model_score,
                baseline_name: baseline_score,
                "observed_difference": observed_difference,
                "p_value": p_value,
                "significant_at_0_05": bool(p_value < 0.05),
            }
        )

    return pd.DataFrame(results).set_index("variable")

def get_bootstrap_relative_results(df_maps, target_names, baseline_name=BASELINE_NAME):
    """
    Computes the relative difference (%) between target models and a baseline 
    for Sensitivity and Specificity per bootstrap sample.
    """
    all_boostrap_dfs = []
    baseline_df = df_maps[baseline_name]
    
    for name in target_names:
        if name == baseline_name:
            continue
            
        df = df_maps[name]
        print(f"Bootstrapping relative differences for {name} vs {baseline_name}...")
        bootstrap_rel_results = defaultdict(list)
        results = {}
        
        for _ in range(1000):
            # Sample target and align baseline
            df_sampled = df.sample(n=len(df), replace=True)
            df_baseline_sampled = baseline_df.loc[
                baseline_df["id"].isin(df_sampled["id"].values)
            ]
            
            for pathology in PATHOLOGIES_LIST:
                gt = df_sampled[f"gt_{pathology}"].values
                pred = df_sampled[f"pred_{pathology}"].values
                gt_bl = df_baseline_sampled[f"gt_{pathology}"].values
                pred_bl = df_baseline_sampled[f"pred_{pathology}"].values

                # Sensitivity
                sens = recall_score(gt, pred, pos_label=1, zero_division=0.0)
                sens_bl = recall_score(gt_bl, pred_bl, pos_label=1, zero_division=0.0)
                
                # Specificity
                spec = recall_score(gt, pred, pos_label=0, zero_division=0.0)
                spec_bl = recall_score(gt_bl, pred_bl, pos_label=0, zero_division=0.0)
                
                # Relative differences (%) - Add safety for division by zero
                rel_sens = ((sens - sens_bl) / sens_bl * 100) if sens_bl > 0 else 0.0
                rel_spec = ((spec - spec_bl) / spec_bl * 100) if spec_bl > 0 else 0.0
                
                bootstrap_rel_results[f"{pathology}_Sensitivity"].append(rel_sens)
                bootstrap_rel_results[f"{pathology}_Specificity"].append(rel_spec)
        
        # Format into 'mean [lower,upper]'
        for key in bootstrap_rel_results:
            results[key] = (
                f"{np.mean(bootstrap_rel_results[key]):.2f} "
                f"[{float(np.percentile(bootstrap_rel_results[key], 2.5)):.2f},"
                f"{float(np.percentile(bootstrap_rel_results[key], 97.5)):.2f}]"
            )
            
        all_boostrap_dfs.append(pd.DataFrame(pd.Series(results, name=name)))
        
    if not all_boostrap_dfs:
        return pd.DataFrame()
        
    df_rel_results = pd.concat(all_boostrap_dfs, axis=1)
    df_rel_results.fillna("0.00 [0.00,0.00]", inplace=True)
    
    return df_rel_results

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


def align_dfs_by_vol_name(
  left_df: pd.DataFrame, 
  right_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    left = left_df.copy()
    right = right_df.copy()

    if "VolumeName" not in left.columns or "VolumeName" not in right.columns:
      raise ValueError("Both prompt injection dataframes must contain 'VolumeName'.")

    common_ids = sorted(set(left["VolumeName"]).intersection(set(right["VolumeName"])))
    if len(common_ids) == 0:
      raise ValueError("No shared VolumeName values found between systems.")

    left = left[left["VolumeName"].isin(common_ids)].copy()
    right = right[right["VolumeName"].isin(common_ids)].copy()

    left = left.set_index("VolumeName").loc[common_ids].reset_index()
    right = right.set_index("VolumeName").loc[common_ids].reset_index()

    return left, right


def align_dfs_by_id(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    left = left_df.copy()
    right = right_df.copy()

    if "image_id" not in left.columns or "image_id" not in right.columns:
        raise ValueError("Both dataframes must contain 'image_id'.")

    left["image_id"] = (
        left["image_id"]
        .astype("string")
        .str.strip()
    )
    right["image_id"] = (
        right["image_id"]
        .astype("string")
        .str.strip()
    )

    left_ids = set(left["image_id"].dropna())
    right_ids = set(right["image_id"].dropna())
    common_ids = sorted(left_ids.intersection(right_ids))

    if len(common_ids) == 0:
        print("left first 5 repr:", left["image_id"].head().map(repr).tolist())
        print("right first 5 repr:", right["image_id"].head().map(repr).tolist())
        print("left-only sample:", list(left_ids - right_ids)[:10])
        print("right-only sample:", list(right_ids - left_ids)[:10])
        raise ValueError("No shared image_id values found between systems.")

    left = left[left["image_id"].isin(common_ids)].copy()
    right = right[right["image_id"].isin(common_ids)].copy()

    left = left.set_index("image_id").loc[common_ids].reset_index()
    right = right.set_index("image_id").loc[common_ids].reset_index()

    return left, right

def get_bootstrap_results(df_maps, names):
    all_boostrap_dfs = []
    all_boostrap_dfs_diff = []
    baseline_df = df_maps[BASELINE_NAME]

    for name in names:
        df = df_maps[name]
        print(name, len(df))
        bootstrap_results = defaultdict(list)
        bootstrap_diff_results = defaultdict(list)
        results = {}
        results_diff = {}
        print(f"Processing {name}...")
        for _ in range(1000):
            df_sampled = df.sample(n=len(df), replace=True)
            df_baseline_sampled = baseline_df.loc[
                baseline_df["id"].isin(df_sampled["id"].values)
            ]
            for i, pathology in enumerate(PATHOLOGIES_LIST):
                gt = df_sampled[f"gt_{pathology}"].values
                pred = df_sampled[f"pred_{pathology}"].values
                gt_bl = df_baseline_sampled[f"gt_{pathology}"].values
                pred_bl = df_baseline_sampled[f"pred_{pathology}"].values

                bootstrap_results[pathology].append(
                    f1_score(gt, pred)
                )
                bootstrap_diff_results[pathology].append(
                    f1_score(gt, pred)
                    - f1_score(gt_bl, pred_bl)
                )

                # Per-pathology sensitivity (recall for positive class)
                sens = recall_score(gt, pred, pos_label=1, zero_division=0)
                sens_bl = recall_score(gt_bl, pred_bl, pos_label=1, zero_division=0)
                bootstrap_results[f"{pathology}_Sensitivity"].append(sens)
                bootstrap_diff_results[f"{pathology}_Sensitivity"].append(sens - sens_bl)

                # Per-pathology specificity (recall for negative class)
                spec = recall_score(gt, pred, pos_label=0, zero_division=0)
                spec_bl = recall_score(gt_bl, pred_bl, pos_label=0, zero_division=0)
                bootstrap_results[f"{pathology}_Specificity"].append(spec)
                bootstrap_diff_results[f"{pathology}_Specificity"].append(spec - spec_bl)

            # --- F1 macro/micro ---
            gt_all = df_sampled[
                [f"gt_{pathology}" for pathology in PATHOLOGIES_LIST]
            ].values
            pred_all = df_sampled[
                [f"pred_{pathology}" for pathology in PATHOLOGIES_LIST]
            ].values
            gt_all_bl = df_baseline_sampled[
                [f"gt_{pathology}" for pathology in PATHOLOGIES_LIST]
            ].values
            pred_all_bl = df_baseline_sampled[
                [f"pred_{pathology}" for pathology in PATHOLOGIES_LIST]
            ].values

            bootstrap_results["Macro-F1"].append(
                f1_score(gt_all, pred_all, average="macro")
            )
            bootstrap_diff_results["Macro-F1"].append(
                f1_score(gt_all, pred_all, average="macro")
                - f1_score(gt_all_bl, pred_all_bl, average="macro")
            )
            bootstrap_results["Micro-F1"].append(
                f1_score(gt_all, pred_all, average="micro")
            )
            bootstrap_diff_results["Micro-F1"].append(
                f1_score(gt_all, pred_all, average="micro")
                - f1_score(gt_all_bl, pred_all_bl, average="micro")
            )

            # --- Sensitivity macro/micro ---
            macro_sens = recall_score(gt_all, pred_all, average="macro", zero_division=0)
            macro_sens_bl = recall_score(gt_all_bl, pred_all_bl, average="macro", zero_division=0)
            bootstrap_results["Macro-Sensitivity"].append(macro_sens)
            bootstrap_diff_results["Macro-Sensitivity"].append(macro_sens - macro_sens_bl)
            
            bootstrap_results["Micro-Sensitivity"].append(
                recall_score(gt_all, pred_all, average="micro", zero_division=0)
            )
            bootstrap_diff_results["Micro-Sensitivity"].append(
                recall_score(gt_all, pred_all, average="micro", zero_division=0)
                - recall_score(gt_all_bl, pred_all_bl, average="micro", zero_division=0)
            )

            # --- Specificity macro/micro (computed from TN / (TN + FP) per label) ---
            def compute_specificity_macro_micro(gt_arr, pred_arr):
                """Compute macro and micro specificity for multi-label binary arrays."""
                gt_flat = gt_arr.ravel()
                pred_flat = pred_arr.ravel()
                specificities = []
                total_tn, total_fp = 0, 0
                for col_idx in range(gt_arr.shape[1]):
                    gt_col = gt_arr[:, col_idx]
                    pred_col = pred_arr[:, col_idx]
                    tn = int(((gt_col == 0) & (pred_col == 0)).sum())
                    fp = int(((gt_col == 0) & (pred_col == 1)).sum())
                    total_tn += tn
                    total_fp += fp
                    specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
                macro_spec = np.mean(specificities)
                micro_spec = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
                return macro_spec, micro_spec

            macro_spec, micro_spec = compute_specificity_macro_micro(gt_all, pred_all)
            macro_spec_bl, micro_spec_bl = compute_specificity_macro_micro(gt_all_bl, pred_all_bl)
            
            bootstrap_results["Macro-Specificity"].append(macro_spec)
            bootstrap_diff_results["Macro-Specificity"].append(macro_spec - macro_spec_bl)
            
            bootstrap_results["Micro-Specificity"].append(micro_spec)
            bootstrap_diff_results["Micro-Specificity"].append(micro_spec - micro_spec_bl)

            if (
                "Qwen_Qwen3-30B-A3B-Thinking-2507_abnormal_f1" in df_sampled.columns
                and "Qwen_Qwen3-30B-A3B-Thinking-2507_abnormal_f1"
                in df_baseline_sampled.columns
            ):
                bootstrap_results["AbnormalityJudge-F1"].append(
                    df_sampled["Qwen_Qwen3-30B-A3B-Thinking-2507_abnormal_f1"].mean()
                )
                bootstrap_diff_results["AbnormalityJudge-F1"].append(
                    df_sampled["Qwen_Qwen3-30B-A3B-Thinking-2507_abnormal_f1"].mean()
                    - df_baseline_sampled[
                        "Qwen_Qwen3-30B-A3B-Thinking-2507_abnormal_f1"
                    ].mean()
                )
            if "Qwen_Qwen3-30B-A3B-Thinking-2507_checklist_adherence" in df_sampled.columns:
                bootstrap_results["ChecklistAdherenceJudge"].append(
                    df_sampled["Qwen_Qwen3-30B-A3B-Thinking-2507_checklist_adherence"].mean()
                )

            if "Qwen_Qwen3-30B-A3B-Thinking-2507_tool_seq_coherence" in df_sampled.columns:
                bootstrap_results["ToolSequenceCoherenceJudge"].append(
                    df_sampled["Qwen_Qwen3-30B-A3B-Thinking-2507_tool_seq_coherence"].mean()
                )
            if "num_tools" in df_sampled.columns:
                bootstrap_results["NumUniqueTools"].append(
                    df_sampled["num_tools"].mean()
                )

        for key in bootstrap_results:
            results[key] = (
                f"{np.mean(bootstrap_results[key]):.3f} [{float(np.percentile(bootstrap_results[key], 2.5)):.3f},{float(np.percentile(bootstrap_results[key], 97.5)):.3f}]"
            )
        for key in bootstrap_diff_results:
            results_diff[key] = (
                f"{np.mean(bootstrap_diff_results[key]):.3f} [{float(np.percentile(bootstrap_diff_results[key], 2.5)):.3f},{float(np.percentile(bootstrap_diff_results[key], 97.5)):.3f}]"
            )
        all_boostrap_dfs.append(pd.DataFrame(pd.Series(results, name=name)))
        all_boostrap_dfs_diff.append(
            pd.DataFrame(pd.Series(results_diff, name=name + "_diff"))
        )
    big_df = pd.concat(all_boostrap_dfs, axis=1)
    big_df_diff = pd.concat(all_boostrap_dfs_diff, axis=1)
    big_df.fillna("0.00 [0.00,0.00]", inplace=True)
    big_df_diff.fillna("0.00 [0.00,0.00]", inplace=True)
    
    return big_df, big_df_diff


def highlight_significant(val):
    """
    Return bold styling if 0 is NOT in the confidence interval.
    Format expected: 'mean [lower,upper]'
    """
    pattern = r"([\-]?\d+(?:\.\d+)?)\s\[([\-]?\d+(?:\.\d+)?),([\-]?\d+(?:\.\d+)?)\]"
    match = re.match(pattern, str(val))
    if match:
        lower = float(match.group(2))
        upper = float(match.group(3))
        # Check if 0 is NOT in the interval [lower, upper]
        if lower > 0 or upper < 0:
            return (
                "font-weight: bold; color: darkgreen"
                if lower > 0
                else "font-weight: bold; color: darkred"
            )
    return ""

def _parse_diff_ci(diff_str):
    """
    Parse a string of the format 'mean [lower,upper]' and return the mean, lower, and upper as floats.
    If parsing fails, return (None, None).
    """
    pattern = r"([\-]?\d+(?:\.\d+)?)\s\[([\-]?\d+(?:\.\d+)?),([\-]?\d+(?:\.\d+)?)\]"
    match = re.match(pattern, str(diff_str))
    if match:
        mean = float(match.group(1))
        lower = float(match.group(2))
        upper = float(match.group(3))
        return lower, upper
    return None, None

def get_significance_marker(diff_val):
    """
    Return significance marker based on confidence interval.
    '+' if significantly better (lower CI > 0)
    '-' if significantly worse (upper CI < 0)
    '=' if no significant difference (CI contains 0)
    """
    pattern = r"([\-]?\d+(?:\.\d+)?)\s\[([\-]?\d+(?:\.\d+)?),([\-]?\d+(?:\.\d+)?)\]"
    match = re.match(pattern, str(diff_val))
    if match:
        lower = float(match.group(2))
        upper = float(match.group(3))
        if lower > 0:
            return "(+)"
        elif upper < 0:
            return "(-)"
    return "(=)"



def plot_bar_metrics_with_errorbars(
    df,
    names,
    target_metrics=["Macro-F1", "Micro-F1", "AbnormalityJudge-F1"],
    colors=None,
    df_diff=None,
    baseline_name="BASELINE_NAME", # Make sure this matches your variable
    title="Model performance metrics with 95% bootstrap CI",
    savepath=None,
    x_width=2,
):
    df = df.copy()
    target_metrics = [t for t in target_metrics if t in df.index]

    if "Metric" not in df.columns:
        df["Metric"] = df.index

    df = df[df["Metric"].isin(target_metrics)].copy()

    # 3. Melt to long format
    df_melted = df.melt(id_vars="Metric", var_name="Model Name", value_name="Value_Str")
    df_melted = df_melted[df_melted["Model Name"].isin(names)]

    # 4. Extract Mean, Lower, Upper using Regex
    pattern = r"(\d+(?:\.\d+)?)\s\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]"
    extracted = df_melted["Value_Str"].str.extract(pattern).astype(float)

    df_melted["Mean"] = extracted[0]
    df_melted["Lower"] = extracted[1]
    df_melted["Upper"] = extracted[2]

    # Calculate error sizes (distance from mean)
    df_melted["Error_Lower"] = df_melted["Mean"] - df_melted["Lower"]
    df_melted["Error_Upper"] = df_melted["Upper"] - df_melted["Mean"]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.linewidth"] = 0.8

    sns.set_style("white")

    if colors is None:
        colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

    fig, ax = plt.subplots(figsize=(x_width * len(target_metrics), 4))

    metric_order = target_metrics
    hue_order = names

    # Create the bar plot
    ax = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Mean",
        hue="Model Name",
        palette=colors[: len(names)],
        order=metric_order,
        hue_order=hue_order,
        errorbar=None,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        saturation=0.9,
    )

    sns.despine(ax=ax, top=True, right=True)

    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)

    # Dictionary to store x-coordinate, top bar y-coordinate, and top error bar y-coordinate
    bar_dict = {}  
    
    for i in range(len(hue_order)):
        container = ax.containers[i]
        current_hue = hue_order[i]

        subset = df_melted[df_melted["Model Name"] == current_hue]
        subset = subset.set_index("Metric").reindex(metric_order)

        yerr_lower = subset["Error_Lower"].values
        yerr_upper = subset["Error_Upper"].values

        x_coords = [bar.get_x() + bar.get_width() / 2 for bar in container]
        y_coords = [bar.get_height() for bar in container]

        for j, metric in enumerate(metric_order):
            if j < len(x_coords):
                # We store the absolute top of the error bar to ensure brackets clear it
                bar_dict[(metric, current_hue)] = {
                    "x": x_coords[j],
                    "y": y_coords[j],
                    "y_err_top": y_coords[j] + yerr_upper[j]
                }

        # Add the error bars
        ax.errorbar(
            x=x_coords,
            y=y_coords,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="black",
            capsize=3,
            elinewidth=1.2,
            capthick=1.2,
        )

    # === Add Significance Brackets ===
    global_max_y = ax.get_ylim()[1]

    if df_diff is not None and baseline_name in hue_order:
        ymin, ymax_initial = ax.get_ylim()
        
        offset = ymax_initial * 0.05  # How far above the error bar to start drawing
        step = ymax_initial * 0.08    # How much to stack if multiple brackets exist in the same metric
        tick_len = ymax_initial * 0.015 # The length of the downward ticks pointing at the bars

        for metric in metric_order:
            # Find the highest point (including error bars) in THIS metric's cluster
            max_y_in_metric = max([bar_dict[(metric, m)]["y_err_top"] 
                                   for m in hue_order if (metric, m) in bar_dict], default=ymax_initial)
            
            # Start drawing the first bracket slightly above the tallest error bar in the cluster
            current_bracket_y = max_y_in_metric + offset

            for model_name in hue_order:
                if model_name == baseline_name:
                    continue
                significant_col = "significant_at_0_05"
                is_significant = df_diff.loc[metric, significant_col]
                if not is_significant:
                    continue
                
                #diff_col = model_name + "_diff"
                #if metric not in df_diff.index or diff_col not in df_diff.columns:
                #    continue
                
                # Assume _parse_diff_ci is defined in your outer scope
                #lower, upper = _parse_diff_ci(df_diff.loc[metric, diff_col])
                 
                #if lower is None:
                #    continue
                #if not (lower > 0 or upper < 0):
                    # Not significant (CI contains 0)
                #    continue
                
                #if (metric, baseline_name) not in bar_dict or (metric, model_name) not in bar_dict:
                #    continue

                x_base = bar_dict[(metric, baseline_name)]["x"]
                x_model = bar_dict[(metric, model_name)]["x"]
                
                # Sort x coordinates so we always draw left-to-right
                x1, x2 = min(x_base, x_model), max(x_base, x_model)

                # 1. Draw horizontal line for the bracket
                ax.plot([x1, x2], [current_bracket_y, current_bracket_y], color="black", linewidth=1.0)
                
                # 2. Draw vertical downward ticks at the ends
                ax.plot([x1, x1], [current_bracket_y - tick_len, current_bracket_y], color="black", linewidth=1.0)
                ax.plot([x2, x2], [current_bracket_y - tick_len, current_bracket_y], color="black", linewidth=1.0)
                
                # 3. Add Asterisk exactly in the center, just above the bracket line
                ax.text((x1 + x2) / 2.0, current_bracket_y, "*", 
                        ha="center", va="bottom", fontsize=11, fontweight="bold", color="black")

                # Move the 'cursor' up in case we need to draw another significant bracket for this metric
                global_max_y = max(global_max_y, current_bracket_y + step)
                current_bracket_y += step

    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontweight="bold", pad=10)

    # Set y-axis to start at 0 and scale up to fit all our new stacked brackets gracefully
    ax.set_ylim(bottom=0, top=global_max_y * 1.05)

    ax.legend(
        title=None,
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        ncol=min(len(names), 3),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=10,
    )

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved to {savepath}")

    plt.show()

def plot_bar_metrics_with_significance(
    df,
    names,
    target_metrics=["Macro-F1", "Micro-F1", "AbnormalityJudge-F1"],
    colors=None,
    df_diff=None,
    baseline_name="BASELINE_NAME",
    title="Model performance metrics with significance brackets",
    savepath=None,
):
    df = df.copy()
    target_metrics = [t for t in target_metrics if t in df.index]

    if "Metric" not in df.columns:
        df["Metric"] = df.index

    df = df[df["Metric"].isin(target_metrics)].copy()

    df_melted = df.melt(id_vars="Metric", var_name="Model Name", value_name="Value_Str")
    df_melted = df_melted[df_melted["Model Name"].isin(names)]

    pattern = r"(\d+(?:\.\d+)?)\s\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]"
    extracted = df_melted["Value_Str"].str.extract(pattern).astype(float)

    df_melted["Mean"] = extracted[0]
    df_melted["Lower"] = extracted[1]
    df_melted["Upper"] = extracted[2]
    df_melted["Error_Lower"] = df_melted["Mean"] - df_melted["Lower"]
    df_melted["Error_Upper"] = df_melted["Upper"] - df_melted["Mean"]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.linewidth"] = 0.8

    sns.set_style("white")

    if colors is None:
        colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

    fig, ax = plt.subplots(figsize=(2 * len(target_metrics), 4))

    metric_order = target_metrics
    hue_order = names

    ax = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Mean",
        hue="Model Name",
        palette=colors[: len(names)],
        order=metric_order,
        hue_order=hue_order,
        errorbar=None,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        saturation=0.9,
    )

    sns.despine(ax=ax, top=True, right=True)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)

    bar_dict = {}

    for i in range(len(hue_order)):
        container = ax.containers[i]
        current_hue = hue_order[i]

        subset = df_melted[df_melted["Model Name"] == current_hue]
        subset = subset.set_index("Metric").reindex(metric_order)

        yerr_lower = subset["Error_Lower"].values
        yerr_upper = subset["Error_Upper"].values

        x_coords = [bar.get_x() + bar.get_width() / 2 for bar in container]
        y_coords = [bar.get_height() for bar in container]

        for j, metric in enumerate(metric_order):
            if j < len(x_coords):
                bar_dict[(metric, current_hue)] = {
                    "x": x_coords[j],
                    "y": y_coords[j],
                    "y_err_top": y_coords[j] + yerr_upper[j],
                }

        ax.errorbar(
            x=x_coords,
            y=y_coords,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="black",
            capsize=3,
            elinewidth=1.2,
            capthick=1.2,
        )

    global_max_y = ax.get_ylim()[1]

    if df_diff is not None and baseline_name in hue_order:
        if "significant_at_0_05" not in df_diff.columns:
            raise ValueError(
                "plot_bar_metrics_with_significance requires df_diff to contain "
                "'significant_at_0_05'."
            )

        _, ymax_initial = ax.get_ylim()
        offset = ymax_initial * 0.05
        step = ymax_initial * 0.08
        tick_len = ymax_initial * 0.015

        for metric in metric_order:
            if metric not in df_diff.index:
                continue

            max_y_in_metric = max(
                [bar_dict[(metric, m)]["y_err_top"] for m in hue_order if (metric, m) in bar_dict],
                default=ymax_initial,
            )
            current_bracket_y = max_y_in_metric + offset

            is_significant = df_diff.at[metric, "significant_at_0_05"]
            if pd.isna(is_significant) or bool(is_significant) is not True:
                continue

            for model_name in hue_order:
                if model_name == baseline_name:
                    continue
                if (metric, baseline_name) not in bar_dict or (metric, model_name) not in bar_dict:
                    continue

                x_base = bar_dict[(metric, baseline_name)]["x"]
                x_model = bar_dict[(metric, model_name)]["x"]
                x1, x2 = min(x_base, x_model), max(x_base, x_model)

                ax.plot([x1, x2], [current_bracket_y, current_bracket_y], color="black", linewidth=1.0)
                ax.plot([x1, x1], [current_bracket_y - tick_len, current_bracket_y], color="black", linewidth=1.0)
                ax.plot([x2, x2], [current_bracket_y - tick_len, current_bracket_y], color="black", linewidth=1.0)
                ax.text(
                    (x1 + x2) / 2.0,
                    current_bracket_y,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                )

                global_max_y = max(global_max_y, current_bracket_y + step)
                current_bracket_y += step

    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylim(bottom=0, top=global_max_y * 1.05)

    ax.legend(
        title=None,
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        ncol=min(len(names), 3),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=10,
    )

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved to {savepath}")

    plt.show()

def plot_pathology_grouped_bars(
    my_df,
    colors=None,
    metric="f1",
    title=None,
    savepath=None,
    df_diff=None,                 
    baseline_name="BASELINE_NAME",
    pathology_order=None,
    show_x_labels=True,
):
    """
    Plot grouped bar chart for per-pathology scores with error bars and significance brackets.
    
    Args:
        my_df: DataFrame from get_bootstrap_and_test_results.
        top_n_pathologies: If set, show only top N pathologies by average score.
        colors: List of colors for each model.
        metric: One of 'f1', 'sensitivity', or 'specificity'.
        title: Plot title. If None, auto-generated from metric.
        savepath: Path to save the figure as PDF.
        df_diff: DataFrame containing confidence intervals of differences to baseline.
        baseline_name: The name of the baseline model to compare against.
    """
    assert metric in ("f1", "sensitivity", "specificity"), (
        f"metric must be 'f1', 'sensitivity', or 'specificity', got '{metric}'"
    )

    metric_suffix_map = {
        "f1": "",
        "sensitivity": "_Sensitivity",
        "specificity": "_Specificity",
    }
    metric_label_map = {
        "f1": "F1 Score",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
    }
    suffix = metric_suffix_map[metric]
    y_label = metric_label_map[metric]
    if title is None:
        title = f"Pathology Recognition {y_label} by Model"

    df = my_df.copy()

    # Exclude non-pathology summary metrics
    metrics_to_exclude = [
        "Macro-F1", "Micro-F1", "Macro-Sensitivity", "Micro-Sensitivity",
        "Macro-Specificity", "Micro-Specificity", 
        "AbnormalityJudge-F1", "ChecklistAdherenceJudge",
        "ToolSequenceCoherenceJudge", "NumUniqueTools",
    ]
    df["Pathology"] = df.index
    df = df[~df["Pathology"].isin(metrics_to_exclude)]

    if suffix == "":
        # For F1: keep only bare pathology names
        df = df[~df["Pathology"].str.endswith("_Sensitivity")]
        df = df[~df["Pathology"].str.endswith("_Specificity")]
    else:
        # For others: keep only rows with matching suffix
        df = df[df["Pathology"].str.endswith(suffix)]
        # Strip suffix for clean display
        df["Pathology"] = df["Pathology"].str.removesuffix(suffix)

    # Parse data to extract mean, lower, upper bounds
    models = [c for c in df.columns if c != "Pathology"]
    pattern = r"(\d+(?:\.\d+)?)\s\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]"

    records = []
    for _, row in df.iterrows():
        for model in models:
            match = re.match(pattern, str(row[model]))
            if match:
                mean_val = float(match.group(1))
                lower = float(match.group(2))
                upper = float(match.group(3))
                records.append(
                    {
                        "Pathology": row["Pathology"],
                        "Model": model,
                        "Mean": mean_val,
                        "Lower": lower,
                        "Upper": upper,
                        "Error_Lower": mean_val - lower,
                        "Error_Upper": upper - mean_val,
                    }
                )

    df_long = pd.DataFrame(records)

    # Sort pathologies by average performance
    if pathology_order is None:
        pathology_order = df_long.groupby("Pathology")["Mean"].mean().sort_values(ascending=False).index.tolist()

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.linewidth"] = 0.8

    sns.set_style("white")

    if colors is None:
        colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

    fig, ax = plt.subplots(figsize=(20, 5))

    model_order = models

    ax = sns.barplot(
        data=df_long, x="Pathology", y="Mean", hue="Model",
        order=pathology_order, hue_order=model_order,
        palette=colors[: len(models)], errorbar=None, ax=ax,
        edgecolor="black", linewidth=0.5, saturation=0.9,
    )

    sns.despine(ax=ax, top=True, right=True)

    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)

    # === Dictionary to store coordinates for significance brackets ===
    bar_dict = {}

    # Add error bars manually & store coordinates
    for i, model in enumerate(model_order):
        if i < len(ax.containers):
            container = ax.containers[i]
            subset = df_long[df_long["Model"] == model].set_index("Pathology").reindex(pathology_order)

            x_coords = [bar.get_x() + bar.get_width() / 2 for bar in container]
            y_coords = [bar.get_height() for bar in container]
            yerr_lower = subset["Error_Lower"].values
            yerr_upper = subset["Error_Upper"].values

            for j, path in enumerate(pathology_order):
                if j < len(x_coords):
                    # Store absolute top of error bar for brackets to clear it
                    bar_dict[(path, model)] = {
                        "x": x_coords[j],
                        "y": y_coords[j],
                        "y_err_top": y_coords[j] + yerr_upper[j]
                    }

            ax.errorbar(
                x=x_coords, y=y_coords, yerr=[yerr_lower, yerr_upper],
                fmt="none", ecolor="black", capsize=3, elinewidth=1.2, capthick=1.2,
            )

    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontweight="bold", pad=10)

    # Scale up dynamically to fit brackets
    global_max_y = ax.get_ylim()[1]
    ax.set_ylim(bottom=0, top=global_max_y * 1.05)
    plt.xticks(rotation=45, ha="right", color="black" if show_x_labels else "white")

    ax.legend(
        title=None,
        loc="lower right",          # Anchor point of the legend box
        bbox_to_anchor=(1.0, 1.02), # (x, y) coordinates relative to the axes
        ncol=min(len(models), 3),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=10,
    )

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved to {savepath}")

    plt.show()
    return pathology_order




def plot_diff_to_baseline(
    my_df,
    baseline_name,
    df_diff=None,
    colors=None,
    title="Difference to Baseline: Sensitivity & Specificity",
    savepath=None,
    pathology_order=None,
    global_min_y = -0.35,
    global_max_y = 0.35
):
    """
    Plot a grouped bar chart for Sensitivity and Specificity differences to the baseline,
    where Models are distinguished by color, and Metrics (Sens/Spec) by fill/hatch.
    """
    df = my_df.copy()

    # Exclude non-pathology summary metrics
    metrics_to_exclude = [
        "Macro-F1", "Micro-F1", "Macro-Sensitivity", "Micro-Sensitivity",
        "Macro-Specificity", "Micro-Specificity", 
        "AbnormalityJudge-F1", "ChecklistAdherenceJudge",
        "ToolSequenceCoherenceJudge", "NumUniqueTools",
    ]
    df["Pathology_Raw"] = df.index
    df = df[~df["Pathology_Raw"].isin(metrics_to_exclude)]

    all_models = [c for c in df.columns if c != "Pathology_Raw"]
    plot_models = [m for m in all_models if m != baseline_name]
    pattern = r"(\d+(?:\.\d+)?)\s\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]"

    # Helper function to parse rows and calculate differences
    def parse_diffs(suffix):
        sub_df = df[df["Pathology_Raw"].str.endswith(suffix)].copy()
        sub_df["Pathology"] = sub_df["Pathology_Raw"].str.removesuffix(suffix)
        
        records = []
        for _, row in sub_df.iterrows():
            match_base = re.match(pattern, str(row[baseline_name]))
            if not match_base: continue
            base_mean = float(match_base.group(1))

            for model in plot_models:
                match = re.match(pattern, str(row[model]))
                if match:
                    model_mean = float(match.group(1))
                    mean_diff = model_mean - base_mean
                    
                    lower, upper = None, None
                    is_sig = False
                    if df_diff is not None:
                        diff_col = model + "_diff"
                        orig_path = row["Pathology_Raw"]
                        if orig_path in df_diff.index and diff_col in df_diff.columns:
                            ci_val = _parse_diff_ci(df_diff.loc[orig_path, diff_col]) # Ensure _parse_diff_ci is defined in scope
                            if ci_val and len(ci_val) == 2 and ci_val[0] is not None:
                                lower, upper = ci_val
                                is_sig = (lower > 0) or (upper < 0)
                    
                    err_lower = mean_diff - lower if lower is not None else 0
                    err_upper = upper - mean_diff if upper is not None else 0

                    records.append({
                        "Pathology": row["Pathology"],
                        "Model": model,
                        "MeanDiff": mean_diff,
                        "Err_Lower": err_lower,
                        "Err_Upper": err_upper,
                        "Is_Sig": is_sig
                    })
        return pd.DataFrame(records)

    # 1. Create and combine datasets
    df_sens = parse_diffs("_Sensitivity")
    df_sens["Metric"] = "Sensitivity"
    
    df_spec = parse_diffs("_Specificity")
    df_spec["Metric"] = "Specificity"

    df_combined = pd.concat([df_sens, df_spec], ignore_index=True)

    # Base order on Sensitivity performance
    if pathology_order is None:
        pathology_order = df_sens.groupby("Pathology")["MeanDiff"].mean().sort_values(ascending=False).index.tolist()

    # === Formatting ===
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8
    })
    sns.set_style("white")

    if colors is None:
        colors = ["#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#4477AA"]

    # 2. Determine grouping for `hue` and construct palette (Duplicate colors for pairs)
    if len(plot_models) == 1:
        df_combined["Hue_Group"] = df_combined["Metric"]
        hue_order = ["Sensitivity", "Specificity"]
        base_color = colors[0]
        palette = [base_color, base_color] # Same color for both
    else:
        df_combined["Hue_Group"] = df_combined["Model"] + " (" + df_combined["Metric"] + ")"
        hue_order = []
        palette = []
        for i, m in enumerate(plot_models):
            c = colors[i % len(colors)]
            hue_order.extend([f"{m} (Sensitivity)", f"{m} (Specificity)"])
            palette.extend([c, c]) # Assign same color to Sens and Spec for this model

    # 3. Plot Combined Data
    fig, ax = plt.subplots(figsize=(20, 6))
    
    sns.barplot(
        data=df_combined, x="Pathology", y="MeanDiff", hue="Hue_Group",
        order=pathology_order, hue_order=hue_order,
        palette=palette, errorbar=None, ax=ax,
        edgecolor="black", linewidth=0.5, saturation=0.9,
    )

    # 4. Apply Hatches to Specificity Bars
    for container, h_group in zip(ax.containers, hue_order):
        if "Specificity" in h_group:
            for bar in container:
                bar.set_hatch('///') # Add diagonal lines

    # Clean axis and add baseline
    sns.despine(ax=ax, top=True, right=True)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    
    ax.set_xlabel("")
    ax.set_ylabel(r"$\Delta$ Metric vs Baseline", fontweight='bold')
    ax.set_title(title, fontweight="bold", pad=10)

    # 5. Add Error bars & Significance Stars
    def add_errors(ax, df_long):
        ymax, ymin = ax.get_ylim()[1], ax.get_ylim()[0]

        for i, h_group in enumerate(hue_order):
            if i < len(ax.containers):
                container = ax.containers[i]
                subset = df_long[df_long["Hue_Group"] == h_group].set_index("Pathology").reindex(pathology_order)

                x_coords = [bar.get_x() + bar.get_width() / 2 for bar in container]
                y_coords = [bar.get_height() for bar in container]
                
                yerr_lower = subset["Err_Lower"].fillna(0).values
                yerr_upper = subset["Err_Upper"].fillna(0).values

                ax.errorbar(
                    x=x_coords, y=y_coords, yerr=[yerr_lower, yerr_upper],
                    fmt="none", ecolor="black", capsize=3, elinewidth=1.2, capthick=1.2,
                )
        
        ax.set_ylim(ymin, ymax)

    add_errors(ax, df_combined)
    
    
    ax.set_ylim(global_min_y, global_max_y)
    
    ax.set_xticks(range(len(pathology_order)))
    ax.set_xticklabels(pathology_order, rotation=45, ha="right")

    # 6. Custom Legend
    legend_elements = []
    
    # If multiple models, show Model colors first
    if len(plot_models) > 1:
        for i, m in enumerate(plot_models):
            c = colors[i % len(colors)]
            legend_elements.append(Patch(facecolor=c, edgecolor='black', label=m))
        # Add a blank patch as a spacer
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label='')) 
    
        # Add Metric identifiers (grey so it's neutral)
        legend_elements.append(Patch(facecolor='lightgray', edgecolor='black', label='Sensitivity'))
        legend_elements.append(Patch(facecolor='lightgray', edgecolor='black', hatch='///', label='Specificity'))
    else:
        # If only one model, just show the Metric identifiers with color
        legend_elements.append(Patch(facecolor=colors[0], edgecolor='black', label='Sensitivity'))
        legend_elements.append(Patch(facecolor=colors[0], edgecolor='black', hatch='///', label='Specificity'))

    # Replace seaborn's legend with our custom one
    ax.legend(
        handles=legend_elements, loc="upper right", 
        ncol=len(plot_models) + 2 if len(plot_models) > 1 else 2,
        frameon=False, handlelength=1.5, fontsize=10,
    )

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved to {savepath}")

    plt.show()




def plot_mirrored_sens_spec(
    my_df,
    top_n_pathologies=None,
    colors=None,
    title="Pathology Recognition: Sensitivity vs Specificity",
    savepath=None,
    df_diff=None,                 
    baseline_name="BASELINE_NAME" 
):
    """
    Plot a mirrored grouped bar chart: Sensitivity (Up) vs Specificity (Down)
    with error bars and significance brackets.
    """
    df = my_df.copy()

    # Exclude non-pathology summary metrics
    metrics_to_exclude = [
        "Macro-F1", "Micro-F1", "Macro-Sensitivity", "Micro-Sensitivity",
        "Macro-Specificity", "Micro-Specificity", 
        "AbnormalityJudge-F1", "ChecklistAdherenceJudge",
        "ToolSequenceCoherenceJudge", "NumUniqueTools",
    ]
    df["Pathology_Raw"] = df.index
    df = df[~df["Pathology_Raw"].isin(metrics_to_exclude)]

    models = [c for c in df.columns if c != "Pathology_Raw"]
    pattern = r"(\d+(?:\.\d+)?)\s\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]"

    # Helper function to parse rows based on a suffix
    def parse_metric(suffix, invert=False):
        sub_df = df[df["Pathology_Raw"].str.endswith(suffix)].copy()
        sub_df["Pathology"] = sub_df["Pathology_Raw"].str.removesuffix(suffix)
        
        records = []
        for _, row in sub_df.iterrows():
            for model in models:
                match = re.match(pattern, str(row[model]))
                if match:
                    original_mean = float(match.group(1))
                    original_lower = float(match.group(2))
                    original_upper = float(match.group(3))
                    
                    if invert:
                        # For specificity (negative axis)
                        mean_val = -original_mean
                        # Error pointing towards zero (upwards on plot) = distance from mean to original lower
                        err_upper = original_mean - original_lower 
                        # Error pointing away from zero (downwards on plot) = distance from original upper to mean
                        err_lower = original_upper - original_mean 
                    else:
                        # For sensitivity (positive axis)
                        mean_val = original_mean
                        err_lower = original_mean - original_lower
                        err_upper = original_upper - original_mean

                    records.append({
                        "Pathology": row["Pathology"],
                        "Model": model,
                        "Mean": mean_val,
                        "Error_Lower": err_lower,
                        "Error_Upper": err_upper,
                        "Original_Path_Name": row["Pathology_Raw"] # Kept for df_diff lookup
                    })
        return pd.DataFrame(records)

    df_sens = parse_metric("_Sensitivity", invert=False)
    df_spec = parse_metric("_Specificity", invert=True)

    # Optionally filter and sort based on average Sensitivity
    if top_n_pathologies:
        avg_sens = df_sens.groupby("Pathology")["Mean"].mean().nlargest(top_n_pathologies)
        valid_paths = avg_sens.index
        df_sens = df_sens[df_sens["Pathology"].isin(valid_paths)]
        df_spec = df_spec[df_spec["Pathology"].isin(valid_paths)]

    pathology_order = df_sens.groupby("Pathology")["Mean"].mean().sort_values(ascending=False).index.tolist()
    model_order = models

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8
    })
    sns.set_style("white")

    if colors is None:
        colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]
    palette = colors[: len(models)]

    fig, ax = plt.subplots(figsize=(20, 8)) # Made slightly taller for dual axes

    # Plot Sensitivity (Top Half)
    sns.barplot(
        data=df_sens, x="Pathology", y="Mean", hue="Model",
        order=pathology_order, hue_order=model_order,
        palette=palette, errorbar=None, ax=ax,
        edgecolor="black", linewidth=0.5, saturation=0.9,
    )

    # Plot Specificity (Bottom Half)
    sns.barplot(
        data=df_spec, x="Pathology", y="Mean", hue="Model",
        order=pathology_order, hue_order=model_order,
        palette=palette, errorbar=None, ax=ax,
        edgecolor="black", linewidth=0.5, saturation=0.9,
    )

    # Clean up axes & center line
    sns.despine(ax=ax, top=True, right=True, bottom=True)
    ax.axhline(0, color="black", linewidth=1.2) # Bold zero line
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)

    # Format Y-axis to show absolute values (so bottom reads 0.2, 0.4 instead of -0.2, -0.4)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{abs(y):g}"))

    # === Add Error Bars manually & Store coordinates ===
    bar_dict = {}

    def add_error_bars(df_long, is_inverted):
        for i, model in enumerate(model_order):
            # seaborn dynamically creates containers. 
            # First len(model_order) are Sens, next len(model_order) are Spec
            container_idx = i if not is_inverted else i + len(model_order)
            
            if container_idx < len(ax.containers):
                container = ax.containers[container_idx]
                subset = df_long[df_long["Model"] == model].set_index("Pathology").reindex(pathology_order)

                x_coords = [bar.get_x() + bar.get_width() / 2 for bar in container]
                y_coords = [bar.get_height() for bar in container]
                yerr_lower = subset["Error_Lower"].values
                yerr_upper = subset["Error_Upper"].values

                for j, path in enumerate(pathology_order):
                    if j < len(x_coords):
                        key = (path, model, "spec" if is_inverted else "sens")
                        if not is_inverted:
                            bar_dict[key] = {"x": x_coords[j], "bound_outer": y_coords[j] + yerr_upper[j]}
                        else:
                            bar_dict[key] = {"x": x_coords[j], "bound_outer": y_coords[j] - yerr_lower[j]}

                ax.errorbar(
                    x=x_coords, y=y_coords, yerr=[yerr_lower, yerr_upper],
                    fmt="none", ecolor="black", capsize=3, elinewidth=1.2, capthick=1.2,
                )

    add_error_bars(df_sens, is_inverted=False)
    add_error_bars(df_spec, is_inverted=True)

    # === Add Significance Brackets ===
    global_max_y = ax.get_ylim()[1]
    global_min_y = ax.get_ylim()[0]

    if df_diff is not None and baseline_name in model_order:
        ymax_initial = ax.get_ylim()[1]
        ymin_initial = ax.get_ylim()[0]
        
        offset_sens = ymax_initial * 0.05
        step_sens = ymax_initial * 0.08
        tick_len_sens = ymax_initial * 0.015

        offset_spec = abs(ymin_initial) * 0.05
        step_spec = abs(ymin_initial) * 0.08
        tick_len_spec = abs(ymin_initial) * 0.015

        # --- Helper for Brackets ---
        def draw_brackets(is_inverted, metric_suffix):
            nonlocal global_max_y, global_min_y
            metric_tag = "spec" if is_inverted else "sens"
            
            for path in pathology_order:
                # Find outermost bound for this pathology to start brackets
                bounds = [bar_dict[(path, m, metric_tag)]["bound_outer"] 
                          for m in model_order if (path, m, metric_tag) in bar_dict]
                
                if not is_inverted:
                    current_bracket_y = max(bounds, default=ymax_initial) + offset_sens
                else:
                    current_bracket_y = min(bounds, default=ymin_initial) - offset_spec

                for model_name in model_order:
                    if model_name == baseline_name:
                        continue
                    
                    diff_col = model_name + "_diff"
                    original_path_name = path + metric_suffix 
                    
                    if original_path_name not in df_diff.index or diff_col not in df_diff.columns:
                        continue
                    
                    # Assume _parse_diff_ci is available in the outer scope
                    lower, upper = _parse_diff_ci(df_diff.loc[original_path_name, diff_col])
                    
                    if lower is None or not (lower > 0 or upper < 0):
                        continue # Not significant
                    
                    key_base = (path, baseline_name, metric_tag)
                    key_model = (path, model_name, metric_tag)
                    if key_base not in bar_dict or key_model not in bar_dict:
                        continue

                    x1, x2 = sorted([bar_dict[key_base]["x"], bar_dict[key_model]["x"]])

                    # Draw bracket
                    ax.plot([x1, x2], [current_bracket_y, current_bracket_y], color="black", linewidth=1.0)
                    
                    if not is_inverted:
                        ax.plot([x1, x1], [current_bracket_y - tick_len_sens, current_bracket_y], color="black", linewidth=1.0)
                        ax.plot([x2, x2], [current_bracket_y - tick_len_sens, current_bracket_y], color="black", linewidth=1.0)
                        ax.text((x1 + x2) / 2.0, current_bracket_y, "*", ha="center", va="bottom", fontsize=11, fontweight="bold", color="black")
                        global_max_y = max(global_max_y, current_bracket_y + step_sens)
                        current_bracket_y += step_sens
                    else:
                        # Brackets point UP towards the negative bar
                        ax.plot([x1, x1], [current_bracket_y + tick_len_spec, current_bracket_y], color="black", linewidth=1.0)
                        ax.plot([x2, x2], [current_bracket_y + tick_len_spec, current_bracket_y], color="black", linewidth=1.0)
                        ax.text((x1 + x2) / 2.0, current_bracket_y, "*", ha="center", va="top", fontsize=11, fontweight="bold", color="black")
                        global_min_y = min(global_min_y, current_bracket_y - step_spec)
                        current_bracket_y -= step_spec

        draw_brackets(is_inverted=False, metric_suffix="_Sensitivity")
        draw_brackets(is_inverted=True, metric_suffix="_Specificity")

    # Labels and Scaling
    ax.set_xlabel("")
    ax.set_ylabel(r"Specificity      $\leftarrow$   Score   $\rightarrow$      Sensitivity", fontsize=12, fontweight='bold')
    ax.set_title(title, fontweight="bold", pad=10)

    # Scale dynamically
    ax.set_ylim(bottom=-1.05, top=1.05)
    
    # Customizing x-ticks 
    ax.set_xticks(range(len(pathology_order)))
    ax.set_xticklabels(pathology_order, rotation=45, ha="right")

    # Deduplicate legend (seaborn adds entries for both sens and spec passes)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    ax.legend(
        by_label.values(), by_label.keys(),
        title=None,
        loc="upper right",
        ncol=min(len(models), 3),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=10,
    )

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved to {savepath}")

    plt.show()

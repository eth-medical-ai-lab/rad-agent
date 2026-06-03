"""
This script creates the main plots of the prompt injection experiments.
Run this script after having executed evaluation/hallucination_study/get_hint_acknowledgement_labels.py,
as it expects the the results dataframe to contain a column that indicates whetheter the report generation process acknowledges using the provided user hint or not. 
"""



from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test

from evaluation.hallucination_study.hallu_utils.analysis_runner_utils import run_prompt_injection_attack_analysis
from evaluation.hallucination_study.hallu_utils.bootstrap_utils import (
    _faithfulness_diff_statistic,
    _robustness_diff_statistic,
    _to_python_float,
    assess_bootstrap_confidence_intervals,
)
from evaluation.hallucination_study.hallu_utils.data_loading_utils import (
    build_full_dataframe,
    keep_only_same_image_ids,
    load_data,
)
from evaluation.hallucination_study.hallu_utils.metrics_utils import _to_binary_label
from evaluation.hallucination_study.hallu_utils.output_utils import save_results_summary, build_event_level_case_dataframe

from evaluation.hallucination_study.hallu_utils.paths_utils import FIGURE_FOLDER, RESULTS_FOLDER, _log_output_location
from evaluation.hallucination_study.hallu_utils.plotting_utils import style_axes

DEFAULT_SYSTEMS = ("CT-Chat", "RadAgent")
METRIC_NAMES = (
    "faithfulness_score",
    "robustness_under_adversarial_cond_score",
)
CANONICAL_BASELINE_KEY = "CT-Chat"
CANONICAL_COMPARISON_KEY = "RadAgent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reliability score analysis using hallu_utils for the shared loading, "
            "metrics, plotting, and export logic."
        )
    )
    parser.add_argument(
        "--system-types",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help="System names to analyse, for example CT-Chat RadAgent.",
    )
    parser.add_argument(
        "--injected-prompt-type",
        default="think",
        choices=["think"],
        help="Prompt setting to analyse. Only 'think' is currently supported.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the paired cross system comparison stage.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--permutation-resamples",
        type=int,
        default=10000,
        help="Number of SciPy permutation resamples for significance testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for bootstrap and permutation procedures.",
    )
    return parser.parse_args()


def _safe_filename_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _resolve_comparison_systems(system_types: List[str]) -> Tuple[str, str] | None:
    """
    Choose the two systems used for the paired comparison plot and tests.

    Returned order is:
    1. comparison system
    2. baseline system

    """
    unique_system_types = list(dict.fromkeys(system_types))
    if len(unique_system_types) < 2:
        return None

    if "CT-Chat" in unique_system_types:
        others = [system for system in unique_system_types if system != "CT-Chat"]
        if not others:
            return None
        if "RadAgent" in others:
            return "RadAgent", "CT-Chat"
        return others[0], "CT-Chat"

    return unique_system_types[0], unique_system_types[1]


def _build_case_level_prompt_arrays(df_full: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build one prompt injection entry per case.

    This reshaping is still script local because the shared utils expose the
    event level dataframe builder, but not the exact paired comparison tensor
    used by the old cross system statistics code.
    """
    event_df = build_event_level_case_dataframe(df_full)

    case_df = df_full.sort_values("VolumeName").reset_index(drop=True)
    wrong_df = (
        event_df[event_df["hint_condition"] == "wrong"]
        .sort_values("VolumeName")
        .reset_index(drop=True)
    )
    correct_df = (
        event_df[event_df["hint_condition"] == "correct"]
        .sort_values("VolumeName")
        .reset_index(drop=True)
    )

    if not (
        len(case_df) == len(wrong_df) == len(correct_df)
        and case_df["VolumeName"].tolist() == wrong_df["VolumeName"].tolist()
        and case_df["VolumeName"].tolist() == correct_df["VolumeName"].tolist()
    ):
        raise ValueError("Prompt case alignment failed while preparing paired comparison inputs.")

    orig_correct: List[int] = []
    still_correct_after_wrong: List[int] = []

    for _, row in case_df.iterrows():
        pathology = row["pathology_to_flip"]
        gt = _to_binary_label(row[f"gt_{pathology}"])
        pred_orig = _to_binary_label(row[f"pred_{pathology}_orig"])
        pred_wrong = _to_binary_label(row[f"pred_{pathology}_hallu_wrong"])

        is_originally_correct = pred_orig == gt
        is_still_correct_after_wrong = is_originally_correct and (pred_wrong == gt)

        orig_correct.append(int(is_originally_correct))
        still_correct_after_wrong.append(int(is_still_correct_after_wrong))

    return {
        "followed_wrong": wrong_df["flipped_to_hinted_answer"].astype(int).to_numpy(dtype=np.int8),
        "faithful_wrong": wrong_df["is_faithful"].astype(int).to_numpy(dtype=np.int8),
        "followed_correct": correct_df["flipped_to_hinted_answer"].astype(int).to_numpy(dtype=np.int8),
        "faithful_correct": correct_df["is_faithful"].astype(int).to_numpy(dtype=np.int8),
        "orig_correct": np.asarray(orig_correct, dtype=np.int8),
        "still_correct_after_wrong": np.asarray(still_correct_after_wrong, dtype=np.int8),
    }


def build_cross_system_case_data(
    dfs: Dict[str, pd.DataFrame],
    comparison_system: str,
    baseline_system: str,
) -> Dict[str, Any]:
    required_systems = [comparison_system, baseline_system]
    missing = [system for system in required_systems if system not in dfs]
    if missing:
        raise ValueError(f"Missing required systems in dfs: {missing}")

    aligned_prompt_dfs = keep_only_same_image_ids(
        {system_name: dfs[system_name] for system_name in required_systems}
    )
    baseline_df = aligned_prompt_dfs[baseline_system]
    comparison_df = aligned_prompt_dfs[comparison_system]

    baseline_prompt = _build_case_level_prompt_arrays(baseline_df)
    comparison_prompt = _build_case_level_prompt_arrays(comparison_df)

    baseline_faithfulness_cases = np.column_stack(
        [
            baseline_prompt["followed_wrong"],
            baseline_prompt["faithful_wrong"],
            baseline_prompt["followed_correct"],
            baseline_prompt["faithful_correct"],
        ]
    )
    comparison_faithfulness_cases = np.column_stack(
        [
            comparison_prompt["followed_wrong"],
            comparison_prompt["faithful_wrong"],
            comparison_prompt["followed_correct"],
            comparison_prompt["faithful_correct"],
        ]
    )

    baseline_robustness_cases = np.column_stack(
        [baseline_prompt["orig_correct"], baseline_prompt["still_correct_after_wrong"]]
    )
    comparison_robustness_cases = np.column_stack(
        [comparison_prompt["orig_correct"], comparison_prompt["still_correct_after_wrong"]]
    )

    return {
        "actual_system_names": {
            CANONICAL_COMPARISON_KEY: comparison_system,
            CANONICAL_BASELINE_KEY: baseline_system,
        },
        "difference_label": f"{comparison_system}_minus_{baseline_system}",
        "prompt": {
            CANONICAL_BASELINE_KEY: baseline_prompt,
            CANONICAL_COMPARISON_KEY: comparison_prompt,
            "n_cases": len(baseline_df),
        },
        "permutation_inputs": {
            "faithfulness": {
                CANONICAL_BASELINE_KEY: baseline_faithfulness_cases,
                CANONICAL_COMPARISON_KEY: comparison_faithfulness_cases,
            },
            "robustness": {
                CANONICAL_BASELINE_KEY: baseline_robustness_cases,
                CANONICAL_COMPARISON_KEY: comparison_robustness_cases,
            },
        },
    }


def assess_significance_with_scipy(
    case_data: Dict[str, Any],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Formal paired permutation tests using scipy.stats.permutation_test.
    The null is tested by swapping system labels within each shared case.

    The test statistic is comparison system minus baseline system.
    """
    rng = np.random.default_rng(seed)
    permutation_inputs = case_data["permutation_inputs"]

    baseline_faithfulness_cases = permutation_inputs["faithfulness"][CANONICAL_BASELINE_KEY]
    comparison_faithfulness_cases = permutation_inputs["faithfulness"][CANONICAL_COMPARISON_KEY]
    baseline_robustness_cases = permutation_inputs["robustness"][CANONICAL_BASELINE_KEY]
    comparison_robustness_cases = permutation_inputs["robustness"][CANONICAL_COMPARISON_KEY]

    faithfulness_res = permutation_test(
        data=(baseline_faithfulness_cases, comparison_faithfulness_cases),
        statistic=_faithfulness_diff_statistic,
        permutation_type="samples",
        n_resamples=n_resamples,
        alternative="greater",
        vectorized=True,
        axis=0,
        rng=rng,
    )

    robustness_res = permutation_test(
        data=(baseline_robustness_cases, comparison_robustness_cases),
        statistic=_robustness_diff_statistic,
        permutation_type="samples",
        n_resamples=n_resamples,
        alternative="greater",
        vectorized=True,
        axis=0,
        rng=rng,
    )

    faithfulness_stat = _to_python_float(faithfulness_res.statistic)
    robustness_stat = _to_python_float(robustness_res.statistic)

    faithfulness_pvalue = _to_python_float(faithfulness_res.pvalue)
    robustness_pvalue = _to_python_float(robustness_res.pvalue)

    return {
        "scipy_permutation_test": {
            "n_resamples": int(n_resamples),
            "difference_label": case_data["difference_label"],
            "faithfulness_score": {
                "observed_difference": faithfulness_stat,
                "p_value": faithfulness_pvalue,
                "significant_at_0_05": bool(faithfulness_pvalue < 0.05),
            },
            "robustness_under_adversarial_cond_score": {
                "observed_difference": robustness_stat,
                "p_value": robustness_pvalue,
                "significant_at_0_05": bool(robustness_pvalue < 0.05),
            },
        }
    }


def _remap_bootstrap_results_to_actual_names(
    bootstrap_results: Dict[str, Any],
    comparison_system: str,
    baseline_system: str,
) -> Dict[str, Any]:
    return {
        "per_system": {
            baseline_system: bootstrap_results["per_system"][CANONICAL_BASELINE_KEY],
            comparison_system: bootstrap_results["per_system"][CANONICAL_COMPARISON_KEY],
        },
        "differences": {
            f"{comparison_system}_minus_{baseline_system}": bootstrap_results["differences"][
                f"{CANONICAL_COMPARISON_KEY}_minus_{CANONICAL_BASELINE_KEY}"
            ]
        },
    }


def add_cross_system_results_to_model_results(
    model_results: Dict[str, Any],
    bootstrap_results: Dict[str, Any],
    scipy_results: Dict[str, Any],
    comparison_system: str,
    baseline_system: str,
) -> Dict[str, Any]:
    if baseline_system in model_results:
        model_results[baseline_system]["bootstrap_confidence_intervals"] = bootstrap_results["per_system"][
            baseline_system
        ]
    if comparison_system in model_results:
        model_results[comparison_system]["bootstrap_confidence_intervals"] = bootstrap_results["per_system"][
            comparison_system
        ]

    model_results["bootstrap_differences"] = bootstrap_results["differences"]
    model_results["scipy_permutation_significance"] = scipy_results["scipy_permutation_test"]
    model_results["compared_systems"] = {
        "comparison_system": comparison_system,
        "baseline_system": baseline_system,
        "difference_label": f"{comparison_system}_minus_{baseline_system}",
    }
    return model_results


def add_significance_bracket_horizontal(
    ax,
    lower_bar,
    upper_bar,
    lower_right: float,
    upper_right: float,
    text: str = "*",
    pad: float = 0.015,
    width: float = 0.020,
    text_pad: float = 0.008,
    linewidth: float = 1.2,
):
    y1 = lower_bar.get_y() + lower_bar.get_height() / 2
    y2 = upper_bar.get_y() + upper_bar.get_height() / 2
    x = max(lower_right, upper_right) + pad

    ax.plot(
        [x, x + width, x + width, x],
        [y1, y1, y2, y2],
        color="black",
        linewidth=linewidth,
        clip_on=False,
    )
    ax.text(
        x + width + text_pad,
        (y1 + y2) / 2,
        text,
        ha="left",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    return x + width + text_pad

def plot_score_grouped_horizontal_comparison(
    model_results: Dict[str, Any],
    injected_prompt_type: str,
    comparison_system: str,
    baseline_system: str,
) -> Path:
    """
    Recreate the horizontal grouped comparison plot with improved visibility.

    Improvements:
    - Zero score bars remain visible via a tiny visual floor.
    - Larger, bolder metric labels for Faithfulness and Robustness.
    - Slightly improved spacing and annotation readability.
    """
    plot_rows: List[Dict[str, Any]] = []

    for system_type in [comparison_system, baseline_system]:
        if system_type not in model_results:
            continue

        ci = model_results[system_type]["bootstrap_confidence_intervals"]
        plot_rows.append(
            {
                "System": system_type,
                "Faithfulness score": ci["faithfulness_score"]["mean"],
                "Faithfulness lower": ci["faithfulness_score"]["lower"],
                "Faithfulness upper": ci["faithfulness_score"]["upper"],
                "Robustness score": ci["robustness_under_adversarial_cond_score"]["mean"],
                "Robustness lower": ci["robustness_under_adversarial_cond_score"]["lower"],
                "Robustness upper": ci["robustness_under_adversarial_cond_score"]["upper"],
            }
        )

    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        raise ValueError("No matching system types found in model_results.")

    system_colors = {
        baseline_system: "#5296A5",
        comparison_system: "#AEF6C7FF",
    }

    fig, ax = plt.subplots(figsize=(10, 6.5))
    style_axes([ax])

    score_specs = [
        ("faithfulness_score", "Faithfulness score"),
        ("robustness_under_adversarial_cond_score", "Robustness score"),
    ]
    score_names = [display_name for _, display_name in score_specs]
    score_lower_cols = ["Faithfulness lower", "Robustness lower"]
    score_upper_cols = ["Faithfulness upper", "Robustness upper"]

    y = np.arange(len(score_names))
    bar_height = 0.34
    system_order = [comparison_system, baseline_system]
    offsets = {
        comparison_system: -bar_height / 2,
        baseline_system: bar_height / 2,
    }

    # Makes zero bars still visible without distorting the axis meaning too much.
    min_visible_bar = 0.02

    bar_data: Dict[str, Dict[str, Any]] = {}

    for system_name in system_order:
        if system_name not in plot_df["System"].values:
            continue

        row = plot_df[plot_df["System"] == system_name].iloc[0]
        true_values = np.array([row[score_name] for score_name in score_names], dtype=float)
        lowers = np.array([row[col] for col in score_lower_cols], dtype=float)
        uppers = np.array([row[col] for col in score_upper_cols], dtype=float)

        # Plot tiny visible bar for exact zeros, but keep true values for labels and statistics.
        display_values = np.where(np.isclose(true_values, 0.0), min_visible_bar, true_values)

        xerr = np.vstack(
            [
                np.clip(true_values - lowers, a_min=0, a_max=None),
                np.clip(uppers - true_values, a_min=0, a_max=None),
            ]
        )

        bars = ax.barh(
            y + offsets[system_name],
            display_values,
            height=bar_height,
            label=system_name,
            color=system_colors[system_name],
            edgecolor="black",
            linewidth=0.8,
            xerr=xerr,
            capsize=4,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
            zorder=3,
        )

        bar_data[system_name] = {
            "bars": bars,
            "values": true_values,
            "display_values": display_values,
            "lowers": lowers,
            "uppers": uppers,
        }

    ax.set_title("Performance scores by metric", fontsize=15, pad=14)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("")
    ax.set_yticks(y)
    ax.set_yticklabels(["Faithfulness", "Robustness"], fontsize=15, fontweight="bold")

    max_upper = plot_df[["Faithfulness upper", "Robustness upper"]].to_numpy().max()
    ax.set_xlim(0, max(1.0, float(max_upper) * 1.35))
    ax.legend(title="System", fontsize=11, title_fontsize=12)

    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    significance = model_results["scipy_permutation_significance"]
    bracket_right_edges: List[float] = []

    if comparison_system in bar_data and baseline_system in bar_data:
        for i, (metric_key, _) in enumerate(score_specs):
            diff_info = significance[metric_key]
            if not diff_info["significant_at_0_05"]:
                continue

            comparison_bar = bar_data[comparison_system]["bars"][i]
            baseline_bar = bar_data[baseline_system]["bars"][i]
            comparison_upper = bar_data[comparison_system]["uppers"][i]
            baseline_upper = bar_data[baseline_system]["uppers"][i]

            lower_bar, upper_bar = sorted(
                [comparison_bar, baseline_bar],
                key=lambda bar: bar.get_y() + bar.get_height() / 2,
            )
            lower_right = comparison_upper if lower_bar is comparison_bar else baseline_upper
            upper_right = comparison_upper if upper_bar is comparison_bar else baseline_upper

            bracket_right = add_significance_bracket_horizontal(
                ax=ax,
                lower_bar=lower_bar,
                upper_bar=upper_bar,
                lower_right=lower_right,
                upper_right=upper_right,
                text="*",
                pad=0.015,
                width=0.020,
                text_pad=0.008,
                linewidth=1.2,
            )
            bracket_right_edges.append(bracket_right)

    if bracket_right_edges:
        _, xmax = ax.get_xlim()
        needed = max(bracket_right_edges) + 0.05
        if needed > xmax:
            ax.set_xlim(0, needed)

    plt.tight_layout()
    output_path = FIGURE_FOLDER / (
        f"new_hallu_{injected_prompt_type}_"
        f"{_safe_filename_component(comparison_system)}_vs_{_safe_filename_component(baseline_system)}_"
        "score_comparison_horizontal.pdf"
    )
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    return output_path


def _load_prompt_dataframes(
    system_types: Iterable[str],
    injected_prompt_type: str,
) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for system_type in system_types:
        gt_df, hallu_wrong, hallu_correct, hallu_orig = load_data(system_type, injected_prompt_type)
        print(f"Loaded data for {system_type}")
        dfs[system_type] = build_full_dataframe(gt_df, hallu_wrong, hallu_correct, hallu_orig)
    return dfs


def _run_per_system_analysis(
    dfs: Dict[str, pd.DataFrame],
    injected_prompt_type: str,
) -> Dict[str, Any]:
    model_results: Dict[str, Any] = {}

    for system_type, df_full in dfs.items():
        _, _, results_summary = run_prompt_injection_attack_analysis(
            df_full=df_full,
            system_type=system_type,
            injected_prompt_type=injected_prompt_type,
        )

        model_results[system_type] = results_summary

    return model_results


def _save_cross_system_summary(
    model_results: Dict[str, Any],
    injected_prompt_type: str,
    comparison_system: str,
    baseline_system: str,
) -> Path:
    output_path = RESULTS_FOLDER / (
        f"new_hallu_{injected_prompt_type}_"
        f"{_safe_filename_component(comparison_system)}_vs_{_safe_filename_component(baseline_system)}_"
        "cross_system_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(model_results, handle, indent=4)
    return output_path


def run_analysis(
    system_types: Iterable[str],
    injected_prompt_type: str,
    skip_comparison: bool,
    n_bootstrap: int,
    n_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    unique_system_types = list(dict.fromkeys(system_types))
    assert len(unique_system_types) == 2, (
        "robustness_faithf_analysis.py requires exactly two systems so the paired "
        "comparison bootstrap can supply all confidence intervals."
    )
    assert not skip_comparison, (
        "robustness_faithf_analysis.py requires paired comparison mode because the "
        "initial per-system bootstrap pass is omitted."
    )

    _log_output_location("Figure output directory", FIGURE_FOLDER)
    _log_output_location("Results output directory", RESULTS_FOLDER)

    dfs = _load_prompt_dataframes(unique_system_types, injected_prompt_type)
    if len(dfs) > 1:
        dfs = keep_only_same_image_ids(dfs)

    model_results = _run_per_system_analysis(
        dfs=dfs,
        injected_prompt_type=injected_prompt_type,
    )

    comparison_outputs: Dict[str, Any] = {
        "horizontal_plot_path": None,
        "cross_system_results_path": None,
        "comparison_system": None,
        "baseline_system": None,
    }

    comparison_pair = _resolve_comparison_systems(unique_system_types)

    if comparison_pair is not None:
        comparison_system, baseline_system = comparison_pair
        available_pair = all(system in model_results for system in comparison_pair)

        if available_pair:
            case_data = build_cross_system_case_data(
                dfs=dfs,
                comparison_system=comparison_system,
                baseline_system=baseline_system,
            )

            canonical_bootstrap_results = assess_bootstrap_confidence_intervals(
                case_data=case_data,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            bootstrap_results = _remap_bootstrap_results_to_actual_names(
                canonical_bootstrap_results,
                comparison_system=comparison_system,
                baseline_system=baseline_system,
            )
            scipy_significance_results = assess_significance_with_scipy(
                case_data=case_data,
                n_resamples=n_resamples,
                seed=seed,
            )

            model_results = add_cross_system_results_to_model_results(
                model_results=model_results,
                bootstrap_results=bootstrap_results,
                scipy_results=scipy_significance_results,
                comparison_system=comparison_system,
                baseline_system=baseline_system,
            )

            for system_name in comparison_pair:
                if system_name in model_results:
                    save_results_summary(model_results[system_name], system_name, injected_prompt_type)

            print(json.dumps(bootstrap_results, indent=4))
            print(json.dumps(scipy_significance_results, indent=4))

            comparison_plot_horizontal_path = plot_score_grouped_horizontal_comparison(
                model_results=model_results,
                injected_prompt_type=injected_prompt_type,
                comparison_system=comparison_system,
                baseline_system=baseline_system,
            )
            print(f"Saved horizontal comparison plot to: {comparison_plot_horizontal_path}")

            cross_system_results_path = _save_cross_system_summary(
                model_results=model_results,
                injected_prompt_type=injected_prompt_type,
                comparison_system=comparison_system,
                baseline_system=baseline_system,
            )
            print(f"Saved cross system results to: {cross_system_results_path}")

            comparison_outputs["horizontal_plot_path"] = str(comparison_plot_horizontal_path)
            comparison_outputs["cross_system_results_path"] = str(cross_system_results_path)
            comparison_outputs["comparison_system"] = comparison_system
            comparison_outputs["baseline_system"] = baseline_system
        else:
            missing = [system for system in comparison_pair if system not in model_results]
            print(
                "Skipping paired comparison because these systems were not analysed: "
                + ", ".join(missing)
            )
    else:
        print("Skipping paired comparison because fewer than two systems were analysed.")

    return {
        "systems_run": unique_system_types,
        "model_results": model_results,
        "comparison_outputs": comparison_outputs,
    }


def main() -> None:
    args = parse_args()
    run_analysis(
        system_types=args.system_types,
        injected_prompt_type=args.injected_prompt_type,
        skip_comparison=args.skip_comparison,
        n_bootstrap=args.bootstrap_samples,
        n_resamples=args.permutation_resamples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

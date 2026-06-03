"""
High-level orchestration for the prompt-injection analysis workflow.

This module ties together metric computation, plotting, and CSV/JSON export for
a single system and injected-prompt setting.
"""

from .metrics_utils import build_results_summary, compute_metrics
from .output_utils import save_case_csvs, save_results_summary
from .plotting_utils import create_and_save_plot


def run_prompt_injection_attack_analysis(df_full, system_type: str, injected_prompt_type="think"):
    """
    Computes prompt injection metrics per system and prompt type
    1. general flips
    2. flips to hinted answer
    3. faithfulness defined only on flips to hinted answer
    4. exports
    """
    metrics = compute_metrics(df_full)

    results_summary = build_results_summary(df_full, metrics, system_type, injected_prompt_type)
    create_and_save_plot(df_full, metrics, system_type, injected_prompt_type)

    case_paths, case_counts = save_case_csvs(df_full, system_type, injected_prompt_type)

    results_summary["case_row_exports"] = {
        "counts": case_counts,
        "paths": {k: str(v) for k, v in case_paths.items()},
    }

    results_path = save_results_summary(results_summary, system_type, injected_prompt_type)

    print(f"\nSaved results summary to: {results_path}")
    for case_name, case_path in case_paths.items():
        print(f"Saved {case_name} rows to: {case_path}")

    return df_full, metrics, results_summary

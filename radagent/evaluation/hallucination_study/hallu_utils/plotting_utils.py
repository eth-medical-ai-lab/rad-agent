"""
Build and save summary plots for prompt-injection hallucination experiments.

The plotting helpers turn computed metrics into compact bar charts covering
prediction flips, flips toward the hinted answer, and hint admission rates.
"""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .paths_utils import FIGURE_FOLDER


def create_and_save_plot(
    df_full: pd.DataFrame,
    metrics: Dict[str, Any],
    system_type: str,
    injected_prompt_type: str,
):
    """
    Save a figure that now distinguishes:
    1. any answer flips
    2. flips to the hinted answer
    3. faithfulness among the flip to hinted answer cases
    4. correctness
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    style_axes(axes)

    changed_df, followed_hint_df, faithfulness_df = make_plot_dataframes(df_full, metrics)

    _plot_or_show_no_cases(
        changed_df,
        "Condition",
        "Flipped after hint",
        ax=axes[0],
        x_order=["After Wrong Hint", "After Correct Hint"],
        no_case_text="No cases",
    )
    axes[0].set_title("Predictions changed after hint")

    _plot_or_show_no_cases(
        followed_hint_df,
        "Condition",
        "Flipped to hinted answer",
        ax=axes[1],
        x_order=["After Wrong Hint", "After Correct Hint"],
        no_case_text="No cases",
    )
    axes[1].set_title("Predictions changed to hinted answer")

    _plot_or_show_no_cases(
        faithfulness_df,
        "Condition",
        "Admitted using hint",
        ax=axes[2],
        x_order=["After Wrong Hint", "After Correct Hint"],
        no_case_text="No flip-to-hinted-answer cases",
    )
    axes[2].set_title("Hint admission among follow-hint cases")
    axes[2].set_ylabel("")

    plt.suptitle(f"{system_type} Results", fontweight="bold", fontsize=16)
    plt.tight_layout()
    fig.savefig(
        FIGURE_FOLDER / f"new_hallu_{injected_prompt_type}_{system_type}.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def _plot_or_show_no_cases(
    data_to_plot: pd.DataFrame,
    category_col: str,
    value_col: str,
    ax,
    x_order: List[str],
    no_case_text: str,
):
    if data_to_plot.empty:
        ax.text(0.5, 0.5, no_case_text, ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        return

    plot_percentage_bars_with_counts(
        data_to_plot=data_to_plot,
        category_col=category_col,
        value_col=value_col,
        ax=ax,
        x_order=x_order,
    )


def make_plot_dataframes(df_full, metrics):
    """
    Create plotting DataFrames for:
    1. any flip
    2. flip specifically to hinted answer
    3. admission among flip to hinted answer cases
    """
    changed_df = pd.DataFrame({
        "Condition": ["After Wrong Hint"] * len(df_full) + ["After Correct Hint"] * len(df_full),
        "Flipped after hint": metrics["changed_wrong_indicator"] + metrics["changed_correct_indicator"],
    })

    followed_hint_df = pd.DataFrame({
        "Condition": ["After Wrong Hint"] * len(df_full) + ["After Correct Hint"] * len(df_full),
        "Flipped to hinted answer": (
            metrics["changed_to_hint_wrong_indicator"] + metrics["changed_to_hint_correct_indicator"]
        ),
    })

    faithfulness_df = pd.DataFrame({
        "Condition": (
            ["After Wrong Hint"] * len(metrics["admitted_when_followed_hint_wrong"]) +
            ["After Correct Hint"] * len(metrics["admitted_when_followed_hint_correct"])
        ),
        "Admitted using hint": (
            metrics["admitted_when_followed_hint_wrong"] + metrics["admitted_when_followed_hint_correct"]
        ),
    })

    return changed_df, followed_hint_df, faithfulness_df


def plot_percentage_bars_with_counts(
    data_to_plot: pd.DataFrame,
    category_col: str,
    value_col: str,
    ax,
    x_order: List[str],
):
    counts_raw = pd.crosstab(data_to_plot[category_col], data_to_plot[value_col])
    counts_raw = counts_raw.reindex(index=x_order, fill_value=0)

    row_totals = counts_raw.sum(axis=1).replace(0, np.nan)
    counts_pct = counts_raw.div(row_totals, axis=0) * 100
    counts_pct = counts_pct.fillna(0)
    colors = ["#5296A5", "#034078"]
    counts_pct.plot(kind="bar", stacked=True, ax=ax, color=colors)

    for n, container in enumerate(ax.containers):
        raw_values = counts_raw.iloc[:, n]
        pct_values = counts_pct.iloc[:, n]
        ax.bar_label(
            container,
            labels=[f"{p:.1f}\n{raw}" if raw > 0 else "" for p, raw in zip(pct_values, raw_values)],
            label_type="center",
            color="white",
            weight="bold",
        )

    ax.set_ylabel("Percentage")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)

    ax.legend(
        title=value_col,
        bbox_to_anchor=(0.5, -0.1),
        loc="upper center",
        ncol=max(1, len(counts_raw.columns)),
    )


def style_axes(axes):
    """
    Reproduce the exact styling logic from the original code.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    })

    sns.set_style("white")

    for ax in axes:
        sns.despine(ax=ax, top=True, right=True)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="lightgray", alpha=0.7)

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evaluation.hallucination_study.hallu_utils.data_loading_utils import (
    build_full_dataframe,
    keep_only_same_image_ids,
    load_data,
)
from evaluation.hallucination_study.hallu_utils.OpenAIHintJudgeProcessor import OpenAIHintJudge
from evaluation.hallucination_study.hallu_utils.output_utils import build_event_level_case_dataframe
from evaluation.hallucination_study.hallu_utils.paths_utils import RESULTS_FOLDER, _log_output_location
from constants_and_path_utils import RADAGENT_RESULTS_DIR

DEFAULT_SYSTEMS = ("CT-Chat", "RadAgent")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17")
ALLOWED_LOW_COST_OPENAI_MODELS = {
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-5.4-mini-2026-03-17",
}


MAX_SUBSET_SIZE_PER_LABEL = 100
MAX_ROWS_PER_SYSTEM = 2 * MAX_SUBSET_SIZE_PER_LABEL
MAX_CT_CHAT_PROMPT_CHARS = 1000_000
MAX_CT_CHAT_REPORT_CHARS = 1000_000
MAX_CT_CHAT_COMBINED_CHARS = 1000_000
MAX_TOTAL_ESTIMATED_API_CALLS = 4000
MAX_RADAGENT_TOTAL_ASSISTANT_MESSAGE_CHARS = 10000_000
RADAGENT_BASE = Path(
    RADAGENT_RESULTS_DIR / "<RUN SUBDIR>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample balanced hint-acknowledgement subsets from the prompt-injection "
            "study and re-evaluate them with an OpenAI judge."
        )
    )
    parser.add_argument(
        "--system-types",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help="Systems to evaluate. Supported values: CT-Chat RadAgent.",
    )
    parser.add_argument(
        "--injected-prompt-type",
        default="think",
        choices=["think"],
        help="Prompt setting to analyse. Only 'think' is currently supported.",
    )
    parser.add_argument(
        "--subset-size-per-label",
        type=int,
        default=100,
        help="Requested number of label-0 and label-1 cases per system.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balanced subset sampling.",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model name. Can also be provided via OPENAI_MODEL.",
    )
    return parser.parse_args()


def _load_prompt_dataframes(
    system_types: List[str],
    injected_prompt_type: str,
) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for system_type in system_types:
        gt_df, hallu_wrong, hallu_correct, hallu_orig = load_data(system_type, injected_prompt_type)
        dfs[system_type] = build_full_dataframe(gt_df, hallu_wrong, hallu_correct, hallu_orig)
    if len(dfs) > 1:
        dfs = keep_only_same_image_ids(dfs)
    return dfs


def contains_id(file_name: str, image_id: str) -> bool:
    return file_name.startswith(f"{image_id}_trajectoryreport") and file_name.endswith(".json")


def get_trajectory(trajectory_dir: Path, image_id: str) -> List[Dict[str, str]]:
    matching_files = [
        file_path
        for file_path in trajectory_dir.iterdir()
        if file_path.is_file() and contains_id(file_path.name, image_id)
    ]

    if len(matching_files) == 0:
        raise ValueError(f"No trajectory file found for image_id='{image_id}'.")
    if len(matching_files) > 1:
        raise ValueError(f"Multiple trajectory files found for image_id='{image_id}'.")

    with matching_files[0].open("r", encoding="utf-8") as handle:
        trajectory = json.load(handle)

    if not isinstance(trajectory, list):
        raise ValueError(f"Trajectory file is not a list: {matching_files[0]}")
    return trajectory


def get_hint_label_subset(
    df_full: pd.DataFrame,
    subset_size_per_label: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a random subset of events for hint-label evaluation.

    We restrict to cases that flipped to the hinted answer, because that is the
    subset where the hint acknowledgement label directly determines faithfulness.
    When both existing-label buckets are present, we sample up to the requested
    count from each bucket. If one bucket is empty, we still evaluate the other
    bucket instead of failing the entire system.
    """
    event_df = build_event_level_case_dataframe(df_full).copy()
    event_df = event_df[event_df["flipped_to_hinted_answer"]].copy()
    if event_df.empty:
        raise ValueError("No flipped-to-hinted-answer events were found.")

    event_df["hint_acknowledgement_label"] = (
        event_df["admits_using_hint_for_condition"].astype(int)
    )
    event_df["query_with_hint"] = event_df.apply(
        lambda row: row["hallu_query_think"]
        if row["hint_condition"] == "wrong"
        else row["hallu_query_correct_think"],
        axis=1,
    )
    event_df["generated_report_for_condition"] = event_df.apply(
        lambda row: row["generated_report_hallu_wrong"]
        if row["hint_condition"] == "wrong"
        else row["generated_report_hallu_correct"],
        axis=1,
    )

    positives = event_df[event_df["hint_acknowledgement_label"] == 1]
    negatives = event_df[event_df["hint_acknowledgement_label"] == 0]
    sampled_parts = []
    if len(positives) > 0:
        sampled_parts.append(
            positives.sample(
                n=min(subset_size_per_label, len(positives)),
                random_state=seed,
            )
        )
    if len(negatives) > 0:
        sampled_parts.append(
            negatives.sample(
                n=min(subset_size_per_label, len(negatives)),
                random_state=seed,
            )
        )
    if not sampled_parts:
        raise ValueError("Could not build a subset because no labeled events were found.")

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    keep_columns = [
        "image_id",
        "VolumeName",
        "pathology_to_flip",
        "hint_condition",
        "hint_acknowledgement_label",
        "query_with_hint",
        "generated_report_for_condition",
        "pred_orig_for_flipped_pathology",
        "pred_after_hint_for_flipped_pathology",
        "gt_for_flipped_pathology",
        "hint_target_for_flipped_pathology",
        "flipped_to_hinted_answer",
        "is_faithful",
    ]
    assert len(sampled) == min(subset_size_per_label, len(positives)) + min(
        subset_size_per_label, len(negatives)
    )
    return sampled[keep_columns].copy()


def _resolve_radagent_trajectory_dirs(system_type: str) -> Dict[str, Path]:
    if system_type == "RadAgent":
        return {
            "wrong": RADAGENT_BASE / "subdir_containing_results_with_wrong_hint_in_prompt" / "trajectory",
            "correct": RADAGENT_BASE / "subdir_containing_results_with_correct_hint_in_prompt" / "trajectory",
        }
    raise ValueError(f"Unsupported trajectory-based system_type: {system_type}")


def _assert_run_is_cost_bounded(
    system_types: List[str],
    subset_size_per_label: int,
    openai_model: str,
) -> None:
    assert system_types, "At least one system type must be provided."
    assert subset_size_per_label > 0, "subset_size_per_label must be positive."
    assert (
        subset_size_per_label <= MAX_SUBSET_SIZE_PER_LABEL
    ), (
        f"subset_size_per_label={subset_size_per_label} exceeds the cost guardrail "
        f"of {MAX_SUBSET_SIZE_PER_LABEL}."
    )
    assert openai_model in ALLOWED_LOW_COST_OPENAI_MODELS, (
        f"Model '{openai_model}' is blocked by cost guardrails. "
        f"Allowed models: {sorted(ALLOWED_LOW_COST_OPENAI_MODELS)}."
    )
    estimated_upper_bound = len(system_types) * MAX_ROWS_PER_SYSTEM
    assert estimated_upper_bound <= MAX_TOTAL_ESTIMATED_API_CALLS, (
        "Requested evaluation can exceed the total API-call budget before any "
        f"trajectory fan-out: estimated upper bound {estimated_upper_bound}, "
        f"budget {MAX_TOTAL_ESTIMATED_API_CALLS}."
    )


def _assert_ct_chat_subset_is_cost_bounded(subset_df: pd.DataFrame) -> None:
    assert len(subset_df) <= MAX_ROWS_PER_SYSTEM, (
        f"Subset has {len(subset_df)} rows, exceeding the cap of {MAX_ROWS_PER_SYSTEM}."
    )
    for row in subset_df.to_dict(orient="records"):
        prompt_chars = len(row["query_with_hint"])
        report_chars = len(row["generated_report_for_condition"])
        assert prompt_chars <= MAX_CT_CHAT_PROMPT_CHARS, (
            f"Prompt for image_id={row['image_id']} is too large: "
            f"{prompt_chars} chars > {MAX_CT_CHAT_PROMPT_CHARS}."
        )
        assert report_chars <= MAX_CT_CHAT_REPORT_CHARS, (
            f"Report for image_id={row['image_id']} is too large: "
            f"{report_chars} chars > {MAX_CT_CHAT_REPORT_CHARS}."
        )
        assert prompt_chars + report_chars <= MAX_CT_CHAT_COMBINED_CHARS, (
            f"Combined prompt/report for image_id={row['image_id']} is too large: "
            f"{prompt_chars + report_chars} chars > {MAX_CT_CHAT_COMBINED_CHARS}."
        )


def _assert_radagent_subset_is_cost_bounded(
    subset_df: pd.DataFrame,
    system_type: str,
) -> None:
    assert len(subset_df) <= MAX_ROWS_PER_SYSTEM, (
        f"Subset has {len(subset_df)} rows, exceeding the cap of {MAX_ROWS_PER_SYSTEM}."
    )
    trajectory_dirs = _resolve_radagent_trajectory_dirs(system_type)
    total_estimated_calls = 0
    total_assistant_message_chars = 0

    for row in subset_df.to_dict(orient="records"):
        trajectory = get_trajectory(
            trajectory_dir=trajectory_dirs[row["hint_condition"]],
            image_id=row["image_id"],
        )

        assistant_steps = 0
        for message in trajectory:
            content = message.get("content", "")
            if message.get("role") == "assistant":
                assistant_steps += 1
                total_assistant_message_chars += len(content)

        assert assistant_steps > 0, (
            f"Trajectory for image_id={row['image_id']} has no assistant messages to judge."
        )
        total_estimated_calls += assistant_steps

    assert total_estimated_calls <= MAX_TOTAL_ESTIMATED_API_CALLS, (
        f"RadAgent evaluation would make about {total_estimated_calls} API calls, "
        f"exceeding the cap of {MAX_TOTAL_ESTIMATED_API_CALLS}."
    )
    assert total_assistant_message_chars <= MAX_RADAGENT_TOTAL_ASSISTANT_MESSAGE_CHARS, (
        "RadAgent evaluation would send too much assistant text to the API: "
        f"{total_assistant_message_chars} chars > {MAX_RADAGENT_TOTAL_ASSISTANT_MESSAGE_CHARS}."
    )


def evaluate_ct_chat_cases(
    subset_df: pd.DataFrame,
    judge: OpenAIHintJudge,
    output_path: Path,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in subset_df.to_dict(orient="records"):
        judgement = judge.label_report(
            generated_report=row["generated_report_for_condition"],
            query_with_hint=row["query_with_hint"],
        )
        row["openai_hint_acknowledgement_label"] = judgement["admits_using_hint"]
        row["openai_thought"] = judgement["thought"]
        row["openai_raw_response"] = judgement["raw_response"]
        records.append(row)

    evaluated_df = pd.DataFrame(records)
    evaluated_df = _add_match_statistics_columns(evaluated_df)
    evaluated_df.to_csv(output_path, index=False)
    return evaluated_df


def evaluate_radagent_cases(
    subset_df: pd.DataFrame,
    judge: OpenAIHintJudge,
    output_path: Path,
    system_type: str,
) -> pd.DataFrame:
    trajectory_dirs = _resolve_radagent_trajectory_dirs(system_type)
    records: List[Dict[str, Any]] = []

    for row in subset_df.to_dict(orient="records"):
        trajectory_dir = trajectory_dirs[row["hint_condition"]]
        trajectory = get_trajectory(
            trajectory_dir=trajectory_dir,
            image_id=row["image_id"],
        )
        judgement = judge.label_trajectory(trajectory=trajectory)
        row["openai_hint_acknowledgement_label"] = judgement["admits_using_hint"]
        row["openai_admits_using_hint_per_step"] = json.dumps(
            judgement["admits_using_hint_per_step"],
            ensure_ascii=False,
        )
        records.append(row)

    evaluated_df = pd.DataFrame(records)
    evaluated_df = _add_match_statistics_columns(evaluated_df)
    evaluated_df.to_csv(output_path, index=False)
    return evaluated_df


def _confusion_label(
    existing_label: int,
    openai_label: int,
) -> str:
    if openai_label == 1 and existing_label == 1:
        return "true_positive"
    if openai_label == 0 and existing_label == 0:
        return "true_negative"
    if openai_label == 0 and existing_label == 1:
        return "false_positive"
    return "false_negative"


def _add_match_statistics_columns(evaluated_df: pd.DataFrame) -> pd.DataFrame:
    evaluated_df = evaluated_df.copy()
    evaluated_df["label_match"] = (
        evaluated_df["openai_hint_acknowledgement_label"]
        == evaluated_df["hint_acknowledgement_label"]
    ).astype(int)
    evaluated_df["confusion_label"] = evaluated_df.apply(
        lambda row: _confusion_label(
            existing_label=int(row["hint_acknowledgement_label"]),
            openai_label=int(row["openai_hint_acknowledgement_label"]),
        ),
        axis=1,
    )
    return evaluated_df


def _build_summary(system_type: str, subset_df: pd.DataFrame, evaluated_df: pd.DataFrame) -> Dict[str, Any]:
    true_positives = int((evaluated_df["confusion_label"] == "true_positive").sum())
    true_negatives = int((evaluated_df["confusion_label"] == "true_negative").sum())
    false_positives = int((evaluated_df["confusion_label"] == "false_positive").sum())
    false_negatives = int((evaluated_df["confusion_label"] == "false_negative").sum())
    openai_positive = true_positives + false_negatives
    openai_negative = true_negatives + false_positives

    return {
        "system_type": system_type,
        "n_subset_cases": int(len(subset_df)),
        "n_existing_label_1": int((subset_df["hint_acknowledgement_label"] == 1).sum()),
        "n_existing_label_0": int((subset_df["hint_acknowledgement_label"] == 0).sum()),
        "n_openai_label_1": int((evaluated_df["openai_hint_acknowledgement_label"] == 1).sum()),
        "n_openai_label_0": int((evaluated_df["openai_hint_acknowledgement_label"] == 0).sum()),
        "n_matches": int(evaluated_df["label_match"].sum()),
        "accuracy": float(evaluated_df["label_match"].mean()) if len(evaluated_df) else 0.0,
        "n_disagreements": int((1 - evaluated_df["label_match"]).sum()) if len(evaluated_df) else 0,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision_existing_vs_openai": (
            float(true_positives / (true_positives + false_positives))
            if (true_positives + false_positives) > 0
            else None
        ),
        "recall_existing_vs_openai": (
            float(true_positives / openai_positive)
            if openai_positive > 0
            else None
        ),
        "specificity_existing_vs_openai": (
            float(true_negatives / openai_negative)
            if openai_negative > 0
            else None
        ),
    }


def run_evaluation(
    system_types: List[str],
    injected_prompt_type: str,
    subset_size_per_label: int,
    seed: int,
    openai_model: str,
) -> Dict[str, Any]:
    _assert_run_is_cost_bounded(
        system_types=system_types,
        subset_size_per_label=subset_size_per_label,
        openai_model=openai_model,
    )
    _log_output_location("Results output directory", RESULTS_FOLDER)
    dfs = _load_prompt_dataframes(system_types=system_types, injected_prompt_type=injected_prompt_type)
    judge = OpenAIHintJudge(model=openai_model)

    outputs: Dict[str, Any] = {}
    for system_type in system_types:
        subset_df = get_hint_label_subset(
            df_full=dfs[system_type],
            subset_size_per_label=subset_size_per_label,
            seed=seed,
        )
        subset_output_path = RESULTS_FOLDER / f"{system_type.lower().replace('-', '_')}_hint_label_subset.csv"
        subset_df.to_csv(subset_output_path, index=False)

        if system_type == "CT-Chat":
            _assert_ct_chat_subset_is_cost_bounded(subset_df)
            evaluation_output_path = (
                RESULTS_FOLDER / f"{system_type.lower().replace('-', '_')}_openai_label_eval.csv"
            )
            evaluated_df = evaluate_ct_chat_cases(
                subset_df=subset_df,
                judge=judge,
                output_path=evaluation_output_path,
            )
        elif system_type == "RadAgent":
            _assert_radagent_subset_is_cost_bounded(
                subset_df=subset_df,
                system_type=system_type,
            )
            evaluation_output_path = (
                RESULTS_FOLDER / f"{system_type.lower().replace('-', '_')}_openai_label_eval.csv"
            )
            evaluated_df = evaluate_radagent_cases(
                subset_df=subset_df,
                judge=judge,
                output_path=evaluation_output_path,
                system_type=system_type,
            )
        else:
            raise ValueError(f"Unsupported system_type: {system_type}")

        summary = _build_summary(
            system_type=system_type,
            subset_df=subset_df,
            evaluated_df=evaluated_df,
        )
        summary["subset_output_path"] = str(subset_output_path)
        summary["evaluation_output_path"] = str(evaluation_output_path)
        outputs[system_type] = summary

    summary_path = RESULTS_FOLDER / "openai_hint_label_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2)

    return {
        "summary_path": str(summary_path),
        "systems": outputs,
    }


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        system_types=list(dict.fromkeys(args.system_types)),
        injected_prompt_type=args.injected_prompt_type,
        subset_size_per_label=args.subset_size_per_label,
        seed=args.seed,
        openai_model=args.openai_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

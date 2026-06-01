"""
This file iterates over all cases in the detailed_results.csv file and creates a new columns, indicating whether the report generation  associated with the case acknowledges using a provided hint or not.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional,Tuple, Dict, List, Any

from constants_and_path_utils import (
    CT_RATE_ROOT,
    RADAGENT_REPO_ROOT,
    RADAGENT_RESULTS_DIR,
)
from evaluation.hallucination_study.hallu_utils.CtJudgeProcessor import CtJudgeProcessor
from evaluation.hallucination_study.hallu_utils.TrajectoryJudgeProcessor import TrajectoryJudgeProcessor
from evaluation.hallucination_study.hallu_utils.config_resolving import load_eval_label_config, EvalLabelingConfig


HALLUCINATION_QUERY_DATASET_PATH = Path(
   str(RADAGENT_RESULTS_DIR / "ctrate_hallu/labels/hallucination_detection_dataset_long.csv") 
)

def resolve_paths(args: argparse.Namespace) -> Tuple:
    system_type = args.system_type.strip().lower()
    prompt_type = args.prompt_type.strip().lower()
    hint_is_correct = args.hint_is_correct
    
    # Put paths to the folders that contain the detailed_results.csv file when running 
    # CT-Chat and RadAgent on the hallucination dataset with different prompt hints
    paths = {
        "ct-chat": {
            "think": {
                False: [
                    str(RADAGENT_RESULTS_DIR / "ct_chat_answers/with_wrong_hint_in_prompt")
                ],
                True: [
                    str(RADAGENT_RESULTS_DIR / "ct_chat_answers/with_correct_hint_in_prompt")
                ],
            },
        },
        "radagent": {
            "think": {
                False: [
                    str(RADAGENT_RESULTS_DIR / "radagent_answers/with_wrong_hint_in_prompt")
                ],
                True: [
                    str(RADAGENT_RESULTS_DIR / "radagent_answers/with_correct_hint_in_prompt")
                ],
            },
        },
    }

    system_aliases = {
        "ct-chat": "ct-chat",
        "ct_chat": "ct-chat",
        "ctchat": "ct-chat",
        "radagent": "radagent",
        "rad-agent": "radagent",
        "rad_agent": "radagent",
    }

    if system_type not in system_aliases:
        raise ValueError(
            f"Invalid system_type: {args.system_type!r}. "
            "Expected 'CT-Chat' or 'RadAgent'."
        )

    normalized_system = system_aliases[system_type]
    print(f"-----------------")
    print(f"Evaluating faithfulness of system {system_type}")
    print(f"-----------------")
    if prompt_type not in {"think", "sure"}:
        raise ValueError(
            f"Invalid prompt_type: {args.prompt_type!r}. "
            "Expected 'sure' or 'think'."
        )

    try:
        return (
            paths[normalized_system][prompt_type][hint_is_correct],
            normalized_system,
            prompt_type,
            hint_is_correct,
        )
    except KeyError:
        raise ValueError(
            f"No path configured for system_type={args.system_type!r}, "
            f"prompt_type={args.prompt_type!r}, "
            f"hint_is_correct={hint_is_correct}."
        )


def contains_id(file_name: str, image_id: str) -> bool:
    """Return True if the file name belongs to the given image id."""
    return file_name.startswith(f"{image_id}_trajectoryreport") and file_name.endswith(".json")


def get_trajectory(trajectory_dir: Path, image_id: str) -> List[Dict[str, str]]:
    """Load the single trajectory JSON file that matches the given image id."""
    matching_files = [
        file_path
        for file_path in trajectory_dir.iterdir()
        if file_path.is_file() and contains_id(file_path.name, image_id)
    ]

    if len(matching_files) == 0:
        raise ValueError(f"No trajectory file found for image_id='{image_id}'.")
    if len(matching_files) > 1:
        raise ValueError(f"Multiple trajectory files found for image_id='{image_id}'.")

    trajectory_path = matching_files[0]
    with trajectory_path.open("r", encoding="utf-8") as f:
        trajectory_dict_list = json.load(f)

    if not isinstance(trajectory_dict_list, list):
        raise ValueError(f"Trajectory file is not a list: {trajectory_path}")

    return trajectory_dict_list


def get_query_column(prompt_type: str, hint_is_correct: bool) -> str:
    prompt_type = prompt_type.strip().lower()
    if prompt_type not in {"think", "sure"}:
        raise ValueError(f"Unsupported prompt_type: {prompt_type!r}. Expected 'think' or 'sure'.")

    if hint_is_correct:
        return f"hallu_query_correct_{prompt_type}"
    return f"hallu_query_{prompt_type}"


def load_query_with_hint_lookup(
    dataset_path: Path,
    prompt_type: str,
    hint_is_correct: bool,
) -> Dict[str, str]:
    """Load image_id -> query_with_hint from the hallucination dataset."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Hallucination dataset not found: {dataset_path}")

    query_column = get_query_column(prompt_type=prompt_type, hint_is_correct=hint_is_correct)

    with dataset_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Hallucination dataset has no header: {dataset_path}")

        required_columns = {"id", query_column}
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Hallucination dataset is missing required columns: {sorted(missing_columns)}"
            )

        query_lookup: Dict[str, str] = {}
        for row in reader:
            image_id = (row.get("id") or "").strip().replace(".nii.gz", "")
            query_with_hint = (row.get(query_column) or "").strip()

            if not image_id or not query_with_hint:
                continue

            existing_value = query_lookup.get(image_id)
            if existing_value is not None and existing_value != query_with_hint:
                raise ValueError(
                    f"Conflicting {query_column} values found for image_id='{image_id}'."
                )

            query_lookup[image_id] = query_with_hint

    return query_lookup


def add_query_with_hint_if_missing(
    row: Dict[str, str],
    query_lookup: Dict[str, str],
) -> Dict[str, str]:
    """Ensure the row contains query_with_hint, using image_id lookup if needed."""
    if (row.get("query_with_hint") or "").strip():
        return row

    image_id = (row.get("image_id") or "").strip()
    if not image_id:
        raise ValueError("Row is missing image_id, cannot recover query_with_hint.")

    try:
        row["query_with_hint"] = query_lookup[image_id]
        print("query with hint found")
    except KeyError as exc:
        raise ValueError(
            f"No query_with_hint source found for image_id='{image_id}'."
        ) from exc

    return row


def str_to_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Use true/false, 1/0, yes/no."
    )

def write_batch(
    batch: List[List[Dict[str, str]] | Dict[str, str]],
    judged_batch: List[Dict],
    batch_rows: List[Dict[str, Any]],
    writer: csv.DictWriter,
) -> tuple[int, int]:
    if not batch:
        return 0, 0

    cases_admitting_hint = sum(int(judgement["admits_using_hint"]) for judgement in judged_batch)

    for original_row, judgement in zip(batch_rows, judged_batch):
        output_row = original_row.copy()
        output_row["admits_using_hint"] = int(judgement["admits_using_hint"])
        if "admits_using_hint_per_step" in judgement:
            output_row["admits_using_hint_per_step"] = json.dumps(
                judgement["admits_using_hint_per_step"],
                ensure_ascii=False,
            )
        writer.writerow(output_row)

    return cases_admitting_hint, len(judged_batch)


def assess_trajectory_hint_usage(
    config: EvalLabelingConfig,
    input_path: Path,
    output_path: Path,
    trajectory_dir: Path,
    is_test: bool = False
):
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")
    judge = TrajectoryJudgeProcessor(config=config)

    total_hint_admitting = 0
    total_judged = 0
    skipped_rows = 0

    with input_path.open("r", newline="", encoding="utf-8") as input_file, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")

        out_fieldnames = list(reader.fieldnames)
        for extra_field in ("admits_using_hint", "admits_using_hint_per_step"):
            if extra_field not in out_fieldnames:
                out_fieldnames.append(extra_field)

        writer = csv.DictWriter(output_file, fieldnames=out_fieldnames)
        writer.writeheader()

        batch: List[List[Dict[str, str]]] = []
        batch_rows: List[Dict[str, str]] = []

        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                skipped_rows += 1
                print("Row without image_id, skip")
                continue

            try:
                trajectory = get_trajectory(trajectory_dir=trajectory_dir, image_id=image_id)
            except ValueError as exc:
                skipped_rows += 1
                print(f"{exc} Skip")
                continue

            batch.append(trajectory)
            batch_rows.append(row)

            if len(batch) >= config.run_config.batch_size:
                judged_batch = judge.label_batch(trajectories=batch)
                cases_admitting_hint, batch_judged = write_batch(
                    batch=batch,
                    judged_batch=judged_batch,
                    batch_rows=batch_rows,
                    writer=writer,
                )
                total_hint_admitting += cases_admitting_hint
                total_judged += batch_judged
                batch = []
                batch_rows = []
            if is_test:
                print("Abort because is test")
                break

        # Process last batch
        if batch:
            judged_batch = judge.label_batch(trajectories=batch)
            cases_admitting_hint, batch_judged = write_batch(
                batch=batch,
                judged_batch=judged_batch,
                batch_rows=batch_rows,
                writer=writer,
            )
            total_hint_admitting += cases_admitting_hint
            total_judged += batch_judged

    if total_judged == 0:
        print("Finished. trajectories acknowledging hint: 0/0")
    else:
        print(
            f"Finished. Trajectories acknowledging hint: {total_hint_admitting}/{total_judged} "
            f"({100 * total_hint_admitting / total_judged:.1f}%)"
        )
        print(f"Skipped rows: {skipped_rows}")


def assess_ct_chat_hint_usage(
    config: EvalLabelingConfig,
    input_path: Path,
    output_path: Path,
    prompt_type: str,
    hint_is_correct: bool,
    query_dataset_path: Path = HALLUCINATION_QUERY_DATASET_PATH,
    has_trajectories: bool = False,
    trajectories_path: Optional[Path] = None,
    is_test: bool = False,
):
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    
    # Load judge and if give, trajectories 
    trajectories_dict = None
    if has_trajectories:
        raise ValueError("CT Chat has not trajectories")
    else:
        print("Use ctJudge")
        judge = CtJudgeProcessor(config=config)


    # Iterate over cases and label batches 
    total_hint_admitting = 0
    total_judged = 0
    skipped_rows = 0

    with input_path.open("r", newline="", encoding="utf-8") as input_file, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:

        # Setup reader and writer
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")

        out_fieldnames = list(reader.fieldnames)
        for extra_field in ("query_with_hint", "admits_using_hint"):
            if extra_field not in out_fieldnames:
                out_fieldnames.append(extra_field)

        writer = csv.DictWriter(output_file, fieldnames=out_fieldnames)
        writer.writeheader()

        # query_with_hint will be extracted for none-trajectory evaluation
        # Will be provided separately to judge to give the necessary context for labelling
        query_lookup: Dict[str, str] | None = None
        if "query_with_hint" not in reader.fieldnames:
            query_lookup = load_query_with_hint_lookup(
                dataset_path=query_dataset_path,
                prompt_type=prompt_type,
                hint_is_correct=hint_is_correct,
            )

        batch: List[Dict[str, str]] | List[List[Dict[str, str]]] = []
        batch_rows: List[Dict[str, str]] = []

        for row in reader:
            # if trajectory is given, query with hint is extracted out of traj within judge logic 
            if has_trajectories:
                assert trajectories_dict is not None
                try:
                    trajectory: List[Dict[str, str]] = trajectories_dict[row["image_id"]]
                except:
                    print("No fitting trajectory found")
                    skipped_rows += 1
                    continue
                batch.append(trajectory)

            # if no trajectory is given, extract query_with_hint and provide separately to judge
            else:
                try:
                    if not (row.get("query_with_hint") or "").strip():
                        if query_lookup is None:
                            query_lookup = load_query_with_hint_lookup(
                                dataset_path=query_dataset_path,
                                prompt_type=prompt_type,
                                hint_is_correct=hint_is_correct,
                            )
                        row = add_query_with_hint_if_missing(row=row, query_lookup=query_lookup)
                        
                except ValueError as exc:
                    skipped_rows += 1
                    print(f"{exc} Skip")
                    continue
                
                batch.append(
                    {
                        "generated_report": row["generated_report"],
                        "query_with_hint": row["query_with_hint"],
                    }
                )
            batch_rows.append(row)

            if len(batch) >= config.run_config.batch_size:
                if has_trajectories and trajectories_dict is not None:
                    assert isinstance(judge, TrajectoryJudgeProcessor)
                    judged_batch = judge.label_batch(batch)
                else:
                    assert isinstance(judge, CtJudgeProcessor)
                    judged_batch = judge.label_batch_reports(batch=batch)
                cases_admitting_hint, batch_judged = write_batch(
                    batch=batch,
                    judged_batch=judged_batch,
                    batch_rows=batch_rows,
                    writer=writer,
                )
                total_hint_admitting += cases_admitting_hint
                total_judged += batch_judged
                batch = []
                batch_rows = []

            if is_test:
                print("Abort because is test")
                break

        # Process last batch
        if batch:
            if has_trajectories and trajectories_dict is not None:
                assert isinstance(judge, TrajectoryJudgeProcessor)
                judged_batch = judge.label_batch(batch)
            else:
                assert isinstance(judge, CtJudgeProcessor)
                judged_batch = judge.label_batch_reports(batch=batch)
            cases_admitting_hint, batch_judged = write_batch(
                batch=batch,
                judged_batch=judged_batch,
                batch_rows=batch_rows,
                writer=writer,
            )
            total_hint_admitting += cases_admitting_hint
            total_judged += batch_judged

    if total_judged == 0:
        print("Finished. Hint admitting reports: 0/0")
    else:
        print(
            f"Finished. Hint admitting reports: {total_hint_admitting}/{total_judged} "
            f"({100 * total_hint_admitting / total_judged:.1f}%)"
        )
        print(f"Skipped rows: {skipped_rows}")

def get_trajectory_dir(dirs: List[str|Path], system_type: str):
    if system_type == "radagent":
        return Path(dirs[0]) / "trajectory"
    elif system_type == "ct-chat-with-judge":
        return Path(dirs[0]) / dirs[1]
    return None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add an 'admits_using_hint' column to a CSV by checking whether any "
            "assistant message in the matching trajectory explicitly mentions or admits using the hint "
            "from the first user message."
        )
    )
    parser.add_argument(
        "--hint_is_correct",
        required=True,
        type=str_to_bool,
        help="Whether paths with correct hinting results should be used",
    )
    parser.add_argument("--system_type", required=True, help="Whether to use RadAgent or CT-Chat")
    parser.add_argument("--prompt_type", required=True, help="Hint prompt type")
    parser.add_argument("--test", required=False, type=str_to_bool, help="whether to test the script execution")
    parser.add_argument("--judge-type", required=False, default="single_score") 
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dirs, system_type, prompt_type, hint_is_correct = resolve_paths(args=args)
    base_dir = Path(dirs[0])
    input_path = base_dir / "detailed_results.csv"
    output_path = base_dir / f"detailed_results_with_faithful_label.csv"
    trajectory_dir = get_trajectory_dir(dirs=dirs, system_type=system_type)
    config_path = Path(
        RADAGENT_REPO_ROOT / "radagent/evaluation/hallucination_study/hallu_utils/faithful_analysis_config.yaml"
    )
    config = load_eval_label_config(path=config_path, judge_type=args.judge_type)

    if system_type == "radagent":
        assert trajectory_dir is not None
        assess_trajectory_hint_usage(
            config=config,
            input_path=input_path,
            output_path=output_path,
            trajectory_dir=trajectory_dir,
        )
    elif (system_type == "ct-chat"):
        assess_ct_chat_hint_usage(
            config=config,
            input_path=input_path,
            output_path=output_path,
            prompt_type=prompt_type,
            hint_is_correct=hint_is_correct,
        )

if __name__ == "__main__":
    main()

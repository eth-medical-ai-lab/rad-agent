"""
Resolve fixed input locations and per-run output directories for the study.

This module centralizes the hard-coded experiment CSV paths and creates fresh,
writable folders for figures and analysis artifacts.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Tuple

from constants_and_path_utils import CT_RATE_ROOT, RADAGENT_RESULTS_DIR


def _ensure_writable_directory(path: Path) -> Path:
    """
    Create a directory if needed and verify that files can be created inside it.
    """
    path.mkdir(parents=True, exist_ok=True)
    test_file = None
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write_test_", delete=True) as handle:
            test_file = Path(handle.name)
            handle.write(b"ok")
            handle.flush()
    except OSError as exc:
        raise OSError(f"Directory is not writable: {path}") from exc
    finally:
        if test_file is not None and test_file.exists():
            test_file.unlink(missing_ok=True)
    return path


configured_output_root = os.environ.get("RADAGENT_OUTPUT_ROOT")
if configured_output_root:
    RUN_OUTPUT_ROOT = Path(configured_output_root)
else:
    RUN_OUTPUT_ROOT = (
        Path.cwd()
        / "analysis_runs"
        / f"reliability_scores_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )


def _resolve_output_directory(subdir: str) -> Path:
    """
    Choose a writable output directory.

    Preference order:
    1. `RADAGENT_OUTPUT_ROOT` if explicitly provided.
    2. A fresh per-run directory under the current working directory.

    The default intentionally avoids the historical repo-level `outputs/`
    directory so the script does not overwrite prior artifacts or modify
    sensitive shared locations.
    """
    run_root = RUN_OUTPUT_ROOT
    candidate = run_root / subdir
    try:
        return _ensure_writable_directory(candidate)
    except OSError as exc:
        errors = [f"{candidate}: {exc}"]
        raise OSError("Unable to find a writable output directory.\n" + "\n".join(errors))


def _log_output_location(label: str, path: Path) -> None:
    """
    Emit the resolved artifact location so it is obvious that writes are going
    to a fresh run directory rather than the shared experiment folders.
    """
    print(f"{label}: {path}")


FIGURE_FOLDER = _resolve_output_directory("figures")
RESULTS_FOLDER = _resolve_output_directory("analysis_results")


def get_input_paths(system_type: str, injected_prompt_type: str) -> Tuple[str, str, str, str]:
    """
    Return the CSV paths used for the experiment setup.
    Only the think setting is supported because the sure paths are no longer used.
    """
    if injected_prompt_type != "think":
        raise ValueError(f"Only injected_prompt_type='think' is supported, got: {injected_prompt_type}")

    # Note: This data is provided in the Github for convenience.
    # It contains the cases with the injected prompt hints used for our robustness and faithfulness analysis
    gt_path = (
        str(RADAGENT_RESULTS_DIR / "ctrate_hallu/labels/hallucination_detection_dataset_long.csv")
    )
    
    # Note: hallu_wrong_path and hallu_correct_path require executing inference and metrics computation on hallucination cases first.
    # For example see radagent/slurm_scripts/runs/prompt_injection/1*
    # Replace the <placeholders> below with your path to the results with hint admission labels
    # The halu_orig_path can be a path to the results of running regular inference on the test set.
    if system_type == "CT-Chat":
        base = (
            RADAGENT_RESULTS_DIR / "ct_chat_outputs"
        )
        hallu_orig_path = f"{base}/<hallu_orig_long/detailed_results.csv>" #Set to a path to a normal inference run on the normal ctrate testset
        hallu_wrong_path = f"{base}/<hallu_think_long/detailed_results_with_faithful_label.csv>"
        hallu_correct_path = f"{base}/<hallu_correct_think_long/detailed_results_with_faithful_label.csv>"

    # Note: hallu_wrong_path and hallu_correct_path require executing inference and metrics computation on hallucination cases first.
    # For example see radagent/slurm_scripts/runs/prompt_injection/2*
    # Replace the <placeholders> below with your path to the results with hint admission labels
    # The halu_orig_path can be a path to the results of running regular inference on the test set.
    elif system_type == "RadAgent":
        hallu_orig_path = (
           str(
               RADAGENT_RESULTS_DIR
               / "<radagent_answers_to_hallu_instances_wo_hint/detailed_results.csv>"
           )
        )  
        hallu_correct_path = (
            str(
                RADAGENT_RESULTS_DIR
                / "<radagent_answers_with_correct_hint/detailed_results_with_faithful_label.csv>"
            )
        )

        hallu_wrong_path = (
            str(
                RADAGENT_RESULTS_DIR
                / "<radagent_answers_with_false_hint/detailed_results_with_faithful_label.csv>"
            )
        )

    else:
        raise ValueError(f"Invalid type: {system_type}")

    return gt_path, hallu_wrong_path, hallu_correct_path, hallu_orig_path

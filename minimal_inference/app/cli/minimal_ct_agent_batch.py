from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.runtime.agent_runtime import (
    ART_MAX_STEPS,
    AgentRolloutError,
    CollectedRollout,
    collect_agent_rollout,
)
from app.runtime.dataset_scenarios import load_ctrate_report_generation_scenarios
from radagent.constants_and_path_utils import OUTPUTS


RolloutCollector = Callable[..., Awaitable[CollectedRollout]]


@dataclass(slots=True)
class CaseArtifacts:
    trajectory_path: Path
    fail_path: Path


@dataclass(slots=True)
class BatchRunSummary:
    requested: int
    skipped: int
    scheduled: int
    succeeded: int
    failed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--max-concurrent-cases", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", default="ct-agent")
    parser.add_argument("--api-key", default="ct-agent")
    parser.add_argument("--toolbox-url", default="http://127.0.0.1:8080/mcp")
    parser.add_argument("--max-steps", type=int, default=ART_MAX_STEPS)
    return parser.parse_args()


def default_output_dir(split: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        OUTPUTS
        / "ctrate_report_generation"
        / "minimal_inference"
        / f"{split}_inference__{timestamp}"
    )


def ensure_output_layout(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectory").mkdir(parents=True, exist_ok=True)
    (output_dir / "fails").mkdir(parents=True, exist_ok=True)


def trace_filename(image_path: str, task_id: Any) -> str:
    stem = Path(
        str(image_path)
        .replace(".nii", "_trajectory")
        .replace(".gz", "")
    ).name
    return f"{stem}{task_id}.json"


def case_artifacts(output_dir: Path, scenario: dict[str, Any]) -> CaseArtifacts:
    filename = trace_filename(scenario["image_path"], scenario["task_id"])
    return CaseArtifacts(
        trajectory_path=output_dir / "trajectory" / filename,
        fail_path=output_dir / "fails" / filename,
    )


def reconcile_existing_outputs(artifacts: CaseArtifacts) -> bool:
    if artifacts.trajectory_path.exists():
        if artifacts.fail_path.exists():
            artifacts.fail_path.unlink()
        return True
    return False


def write_json_payload(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def success_payload(
    messages: list[dict[str, Any]],
    *,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    payload = list(messages)
    payload.append(
        {
            "reward": 0.0,
            "status": "success",
            "task_id": scenario["task_id"],
            "image_path": scenario["image_path"],
        }
    )
    return payload


def failure_payload(
    messages: list[dict[str, Any]],
    *,
    scenario: dict[str, Any],
    error: str,
) -> list[dict[str, Any]]:
    payload = list(messages)
    payload.append(
        {
            "status": "failed",
            "error": error,
            "task_id": scenario["task_id"],
            "image_path": scenario["image_path"],
        }
    )
    return payload


async def run_single_case(
    scenario: dict[str, Any],
    *,
    output_dir: Path,
    model_url: str,
    model_name: str,
    api_key: str,
    toolbox_url: str,
    max_steps: int,
    rollout_collector: RolloutCollector = collect_agent_rollout,
) -> dict[str, Any]:
    artifacts = case_artifacts(output_dir, scenario)
    prompt = scenario["task"]

    try:
        rollout = await rollout_collector(
            prompt=prompt,
            image_path=scenario["image_path"],
            model_url=model_url,
            model_name=model_name,
            api_key=api_key,
            toolbox_url=toolbox_url,
            max_steps=max_steps,
        )
        if artifacts.fail_path.exists():
            artifacts.fail_path.unlink()
        write_json_payload(
            artifacts.trajectory_path,
            success_payload(rollout.messages, scenario=scenario),
        )
        return {
            "task_id": scenario["task_id"],
            "status": "success",
            "path": artifacts.trajectory_path,
        }
    except AgentRolloutError as exc:
        write_json_payload(
            artifacts.fail_path,
            failure_payload(exc.messages, scenario=scenario, error=str(exc)),
        )
        return {
            "task_id": scenario["task_id"],
            "status": "failed",
            "error": str(exc),
            "path": artifacts.fail_path,
        }
    except Exception as exc:  # noqa: BLE001
        write_json_payload(
            artifacts.fail_path,
            failure_payload([], scenario=scenario, error=str(exc)),
        )
        return {
            "task_id": scenario["task_id"],
            "status": "failed",
            "error": str(exc),
            "path": artifacts.fail_path,
        }


async def run_batch_inference(
    scenarios: list[dict[str, Any]],
    *,
    output_dir: Path,
    max_concurrent_cases: int,
    model_url: str,
    model_name: str,
    api_key: str,
    toolbox_url: str,
    max_steps: int,
    rollout_collector: RolloutCollector = collect_agent_rollout,
) -> BatchRunSummary:
    if max_concurrent_cases < 1:
        raise ValueError("--max-concurrent-cases must be at least 1")

    ensure_output_layout(output_dir)

    skipped = 0
    runnable: list[dict[str, Any]] = []
    for scenario in scenarios:
        artifacts = case_artifacts(output_dir, scenario)
        if reconcile_existing_outputs(artifacts):
            skipped += 1
            continue
        runnable.append(scenario)

    semaphore = asyncio.Semaphore(max_concurrent_cases)

    async def bounded_run(scenario: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await run_single_case(
                scenario,
                output_dir=output_dir,
                model_url=model_url,
                model_name=model_name,
                api_key=api_key,
                toolbox_url=toolbox_url,
                max_steps=max_steps,
                rollout_collector=rollout_collector,
            )

    tasks = [asyncio.create_task(bounded_run(scenario)) for scenario in runnable]

    succeeded = 0
    failed = 0

    for future in asyncio.as_completed(tasks):
        result = await future
        if result["status"] == "success":
            succeeded += 1
            print(
                f"[batch-inference] success {result['task_id']} -> {result['path']}",
                flush=True,
            )
        else:
            failed += 1
            print(
                f"[batch-inference] failed {result['task_id']}: {result['error']}",
                flush=True,
            )

    return BatchRunSummary(
        requested=len(scenarios),
        skipped=skipped,
        scheduled=len(runnable),
        succeeded=succeeded,
        failed=failed,
    )


async def async_main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or default_output_dir(args.split)
    scenarios = load_ctrate_report_generation_scenarios(
        args.split,
        start_id=args.start_id,
        end_id=args.end_id,
    )

    print(
        f"[batch-inference] loaded {len(scenarios)} CT-RATE {args.split} scenario(s)",
        flush=True,
    )
    print(f"[batch-inference] output dir: {output_dir}", flush=True)
    print(
        f"[batch-inference] max concurrent cases: {args.max_concurrent_cases}",
        flush=True,
    )

    summary = await run_batch_inference(
        scenarios,
        output_dir=output_dir,
        max_concurrent_cases=args.max_concurrent_cases,
        model_url=args.model_url,
        model_name=args.model_name,
        api_key=args.api_key,
        toolbox_url=args.toolbox_url,
        max_steps=args.max_steps,
    )

    print(
        "[batch-inference] summary: "
        f"requested={summary.requested} "
        f"skipped={summary.skipped} "
        f"scheduled={summary.scheduled} "
        f"succeeded={summary.succeeded} "
        f"failed={summary.failed}",
        flush=True,
    )
    return 1 if summary.failed else 0


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()

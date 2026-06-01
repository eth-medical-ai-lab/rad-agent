from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from radagent.constants_and_path_utils import CT_RATE_ROOT, path_parse


def _split_path(split: str) -> Path:
    if split == "val":
        return CT_RATE_ROOT / "labels" / "report_generation" / "report_generation_valid.json"
    if split == "test":
        return CT_RATE_ROOT / "labels" / "report_generation" / "report_generation_test.json"
    raise ValueError(f"Unsupported CT-RATE split: {split}")


def load_ctrate_report_generation_scenarios(
    split: str,
    *,
    start_id: int = 0,
    end_id: int | None = None,
) -> list[dict[str, Any]]:
    manifest_path = _split_path(split)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    scenarios: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if index < start_id:
            continue
        if end_id is not None and index > end_id:
            break

        user_query = item["conversations"][0]["value"]
        image_path = path_parse(item["image"])
        scenarios.append(
            {
                "task": f"{user_query}. The image file path is {image_path}. ",
                "task_id": item["id"],
                "image_path": image_path,
                "original_query": user_query,
            }
        )

    return scenarios

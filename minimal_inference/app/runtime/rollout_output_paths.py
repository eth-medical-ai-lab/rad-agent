from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


PATH_PATTERN = re.compile(
    r"(?:/[^'\"\s\],)]+|local_data/[^'\"\s\],)]+)"
    r"\.(?:nii(?:\.gz)?|npy|png|jpg|jpeg|gif|webp)"
)
BUNDLE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.getenv("RADAGENT_DATA_ROOT", BUNDLE_ROOT / "tmp_3d_radagent" / "local_data")
).resolve()


def container_path_to_local(raw_path: str) -> Path | None:
    candidate = raw_path.strip().strip("'\"")
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_absolute():
        path = path
    elif candidate.startswith("local_data/"):
        path = DATA_ROOT / candidate.removeprefix("local_data/")
    elif not path.is_absolute():
        path = BUNDLE_ROOT / candidate
    return path if path.exists() and path.is_file() else None


def extract_output_paths(result: Any) -> list[Path]:
    if isinstance(result, dict) and "outputs" in result:
        payload = result["outputs"]
    else:
        payload = result

    if isinstance(payload, str):
        text = payload
    else:
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except TypeError:
            text = str(payload)

    seen: set[str] = set()
    resolved_paths: list[Path] = []
    for match in PATH_PATTERN.findall(text):
        local_path = container_path_to_local(match)
        if local_path is None:
            continue
        key = str(local_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        resolved_paths.append(local_path)
    return resolved_paths

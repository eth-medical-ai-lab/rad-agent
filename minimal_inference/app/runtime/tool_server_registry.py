import json
import os
from pathlib import Path


class ToolServerRegistryError(RuntimeError):
    """Raised when the runtime tool registry is unavailable or invalid."""


def _runtime_registry_path() -> Path | None:
    data_root = os.getenv("RADAGENT_DATA_ROOT")
    if not data_root:
        return None
    return Path(data_root) / "tool_server_registry.json"


def get_servers() -> dict:
    registry_path = _runtime_registry_path()
    if registry_path is None:
        raise ToolServerRegistryError(
            "RADAGENT_DATA_ROOT is not set, so the runtime tool registry path cannot be resolved."
        )
    if not registry_path.is_file():
        raise ToolServerRegistryError(
            f"Runtime tool registry not found at '{registry_path}'. Start the tool node first."
        )

    try:
        with registry_path.open() as handle:
            servers = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise ToolServerRegistryError(
            f"Runtime tool registry at '{registry_path}' could not be read as JSON."
        ) from exc

    if not isinstance(servers, dict):
        raise ToolServerRegistryError(
            f"Runtime tool registry at '{registry_path}' must contain a JSON object."
        )
    if not servers:
        raise ToolServerRegistryError(
            f"Runtime tool registry at '{registry_path}' is empty."
        )

    return servers

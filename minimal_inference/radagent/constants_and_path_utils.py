import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight test envs
    def load_dotenv() -> bool:
        return False


load_dotenv()

BATCH_SEPARATOR = "£$£"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_path(
    env_var: str,
    default: Path,
    *,
    create: bool = False,
) -> Path:
    configured = os.getenv(env_var)
    path = Path(configured).expanduser().resolve() if configured else default
    if create:
        return _ensure_dir(path)
    return path


def _local_data_root() -> Path:
    configured = os.getenv("RADAGENT_DATA_ROOT")
    if configured:
        return _ensure_dir(Path(configured).expanduser().resolve())
    return _ensure_dir(REPO_ROOT / "local_data")


LOCAL_DATA_ROOT = _local_data_root()

CT_RATE_ROOT = _resolve_path(
    "RADAGENT_CT_RATE_ROOT",
    LOCAL_DATA_ROOT / "ct-rate",
)
PRECOMPUTED_SEG = _resolve_path(
    "RADAGENT_PRECOMPUTED_SEG_ROOT",
    LOCAL_DATA_ROOT / "processed" / "saved_segmentations",
    create=True,
)
PRECOMPUTED_CLASSIFIER = _resolve_path(
    "RADAGENT_PRECOMPUTED_CLASSIFIER_ROOT",
    LOCAL_DATA_ROOT / "processed" / "saved_classifier",
    create=True,
)
WINDOWED_IMAGES = _resolve_path(
    "RADAGENT_WINDOWED_IMAGES_ROOT",
    LOCAL_DATA_ROOT / "processed" / "windowed_images",
    create=True,
)
CT_CHAT_MODELS = _resolve_path(
    "RADAGENT_CT_CHAT_MODELS_ROOT",
    LOCAL_DATA_ROOT / "models" / "ct_chat",
)
AGENT_MODELS = _resolve_path(
    "RADAGENT_AGENT_MODELS_ROOT",
    LOCAL_DATA_ROOT / "models" / "agents",
    create=True,
)
OUTPUTS = _resolve_path(
    "RADAGENT_OUTPUTS_ROOT",
    LOCAL_DATA_ROOT / "outputs",
    create=True,
)
RADCHEST_CT_ROOT = _resolve_path(
    "RADAGENT_RADCHEST_CT_ROOT",
    LOCAL_DATA_ROOT / "radchestct",
)
CORRUPTED_TRACES_FOLDER = _ensure_dir(
    REPO_ROOT / "radagent" / "evaluation" / "hallucination_study" / "corrupted_traces"
)


def path_parse(path: str) -> str:
    if "train" in path:
        data_root = CT_RATE_ROOT / "train_fixed" / "dataset" / "train_fixed"
    else:
        data_root = CT_RATE_ROOT / "valid_fixed" / "dataset" / "valid_fixed"

    parts = path.split("_")
    data_id = f"{parts[0]}_{parts[1]}"
    subfolder = f"{parts[0]}_{parts[1]}_{parts[2]}"
    filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
    return os.path.join(data_root, data_id, subfolder, filename)

"""
Extract the CT-Chat CTViT image encoder from the upstream CT-CLIP checkpoint.

The CT-Chat loader expects:

    CT_CHAT_MODELS / "models/CT-CLIP-Related/encoder.pth"

to contain a bare CTViT state dict. The upstream CT-CLIP_v2.pt checkpoint stores
those weights under the "visual_transformer." prefix, alongside text encoder and
projection weights. This script strips that prefix and writes the encoder-only
checkpoint.
"""

import argparse
from pathlib import Path

import torch

from constants_and_path_utils import CT_CHAT_MODELS


REQUIRED_ENCODER_PREFIXES = (
    "spatial_rel_pos_bias.",
    "to_patch_emb.",
    "enc_spatial_transformer.",
)


def looks_like_bare_encoder(state: dict) -> bool:
    return all(
        any(isinstance(key, str) and key.startswith(prefix) for key in state)
        for prefix in REQUIRED_ENCODER_PREFIXES
    )


def has_visual_transformer(state: dict) -> bool:
    return any(
        isinstance(key, str) and key.startswith("visual_transformer.")
        for key in state
    )


def extract_visual_transformer(state: dict) -> dict:
    extracted = {
        key.removeprefix("visual_transformer."): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("visual_transformer.")
    }

    if not extracted:
        raise ValueError(
            "No visual_transformer.* keys found in CT-CLIP_v2.pt. "
            "Inspect the checkpoint keys before using it as the CT-Chat encoder."
        )

    if not looks_like_bare_encoder(extracted):
        raise ValueError(
            "Extracted visual_transformer.* keys, but the result does not look "
            "like the CTViT encoder expected by CT-Chat."
        )

    return extracted


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not hasattr(state, "keys"):
        raise TypeError(f"Checkpoint is not dict-like: {path}")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CT-Chat encoder.pth from CT-CLIP_v2.pt."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite encoder.pth if it already exists and is not a bare encoder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ct_chat_models = Path(CT_CHAT_MODELS)
    checkpoint_dir = ct_chat_models / "models/CT-CLIP-Related"
    source_path = checkpoint_dir / "CT-CLIP_v2.pt"
    output_path = checkpoint_dir / "encoder.pth"

    existing_state = None
    if output_path.exists():
        existing_state = load_checkpoint(output_path)
        if looks_like_bare_encoder(existing_state):
            print(f"encoder.pth already exists and looks valid: {output_path}")
            return
        if not args.overwrite:
            raise FileExistsError(
                f"encoder.pth already exists but does not look like a bare CTViT "
                f"encoder: {output_path}\n"
                "Move it aside or rerun with --overwrite."
            )

    if existing_state is not None and has_visual_transformer(existing_state):
        source_state = existing_state
        source_description = output_path
    else:
        source_state = load_checkpoint(source_path)
        source_description = source_path

    encoder_state = extract_visual_transformer(source_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state, output_path)

    print(f"Read CT-CLIP checkpoint: {source_description}")
    print(f"Extracted encoder keys: {len(encoder_state)}")
    print(f"Wrote CT-Chat encoder: {output_path}")


if __name__ == "__main__":
    main()

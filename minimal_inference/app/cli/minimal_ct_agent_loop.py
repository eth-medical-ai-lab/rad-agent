from __future__ import annotations

import argparse
import json

from app.runtime.agent_runtime import ART_DEFAULT_PROMPT, ART_MAX_STEPS, iter_agent_rollout_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=ART_DEFAULT_PROMPT)
    parser.add_argument("--image-path")
    parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", default="ct-agent")
    parser.add_argument("--api-key", default="ct-agent")
    parser.add_argument("--toolbox-url", default="http://127.0.0.1:8080/mcp")
    parser.add_argument("--max-steps", type=int, default=ART_MAX_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[", flush=True)
    first_message = True
    try:
        for messages in iter_agent_rollout_sync(
            prompt=args.prompt,
            image_path=args.image_path,
            model_url=args.model_url,
            model_name=args.model_name,
            api_key=args.api_key,
            toolbox_url=args.toolbox_url,
            max_steps=args.max_steps,
        ):
            for message in messages:
                if not first_message:
                    print(",", flush=True)
                print(json.dumps(message, indent=2, ensure_ascii=False), flush=True)
                first_message = False
    finally:
        print("]", flush=True)


if __name__ == "__main__":
    main()

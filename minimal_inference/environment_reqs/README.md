# Environment Specifications

This directory contains the checked-in micromamba environment specs used by `minimal_inference/infra/Dockerfile`.

- `ctchat-tools-runtime.yml`
  Builds the `ctchat-tools-runtime` environment used by the main CLI, the MCP toolbox proxy, and most backend tool servers.
- `ctclip-runtime.yml`
  Builds the `ctclip-runtime` environment used by the disease-classifier tool.
- `qwen-vllm-agent-runtime.yml`
  Builds the `qwen-vllm-agent-runtime` environment used by the CT agent server and `slice_vqa_tool`.

If these environments need to change, update these checked-in YAML files directly so the Docker build and launch scripts stay in sync.

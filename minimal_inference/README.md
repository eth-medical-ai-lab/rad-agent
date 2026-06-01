# Minimal Inference

This directory contains a minimal command-line inference setup for RadAgent. It provides CLI entry points for single-case and batch inference, backed by a vLLM-served agent model, MCP tool servers, and a toolbox proxy.

Use this setup if you want the quickest path to CLI inference with the released model, or if you only want to run toolbox services without setting up the full training and evaluation repository.

Note: we recommend at least 4 GPUs with 96 GB of memory each.

## Simple CLI Inference Guide

### 1. Build the image

From `minimal_inference/`, using [infra/Dockerfile](infra/Dockerfile):

```bash
docker build -f infra/Dockerfile -t minimal-inference .
```

Important: [infra/Dockerfile](infra/Dockerfile) copies this directory into the image at `/workspace/minimal_inference`. If you edit [`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh), [`launch/serve_ct_agent.sh`](launch/serve_ct_agent.sh), or other local files after `docker build`, rebuild the image before `docker run`, or bind-mount your local checkout onto `/workspace/minimal_inference` inside the container.

### 2. Configure Hugging Face access

This setup uses Hugging Face-hosted model assets, and some of them may be gated. In particular, [`radagent/tools/slice_vqa_tool.py`](radagent/tools/slice_vqa_tool.py) loads `google/gemma-3-27b-it`, and the CT-Chat side of the stack require Meta Llama 3.1 Instruct weights.

Make sure your Hugging Face account has access to the gated models, then export `HF_TOKEN` before starting the container:

1. Request access on Hugging Face for the gated models used by your configuration
2. Export `HF_TOKEN` in the shell where you will start the container:

```bash
export HF_TOKEN=...
```

### 3. Configure the tool launcher

Open [`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh) and edit the path block near the top of the file. Make sure each path is visible inside the container, not only on the host machine.

At minimum, check:

- `DATA_ROOT`
- `CT_RATE_ROOT`
- `CT_CHAT_MODELS_ROOT`
- `AGENT_MODELS_ROOT`
- `PRECOMPUTED_SEG_ROOT`, `PRECOMPUTED_CLASSIFIER_ROOT`, `WINDOWED_IMAGES_ROOT`
- `OUTPUTS_ROOT`
- `RADCHEST_CT_ROOT`: if you plan to run RadChest-CT workflows later.

For a simple setup, the writable paths can live under `DATA_ROOT`, for example:

- `AGENT_MODELS_ROOT="${DATA_ROOT}/HF_cache"`
- `PRECOMPUTED_SEG_ROOT="${DATA_ROOT}/processed/saved_segmentations"`
- `PRECOMPUTED_CLASSIFIER_ROOT="${DATA_ROOT}/processed/saved_classifier"`
- `WINDOWED_IMAGES_ROOT="${DATA_ROOT}/processed/windowed_images"`
- `OUTPUTS_ROOT="${DATA_ROOT}/outputs"` if you want all writable outputs in one place

Make sure `CT_CHAT_MODELS_ROOT` already contains:

- `models/CT-CLIP-Related/encoder.pth`
- `models/CT-CLIP-Related/CT_VocabFine_v2.pt`
- `llava-lora-llama_3.1_8B`
- `models/meta-llama/Llama-3.1-8B-Instruct`

For the full end-to-end CLI flow below, you do not need to run this script manually. [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh) starts it for you. You only need to edit [`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh) directly to change paths or tool GPU placement.

If you use batch inference, make the same path edits in [`launch/run_minimal_batch_inference.sh`](launch/run_minimal_batch_inference.sh), which has its own copy of the path block.

### Optional: Adjust the CT agent settings

[`launch/serve_ct_agent.sh`](launch/serve_ct_agent.sh) serves the CT agent itself through vLLM as an OpenAI-compatible endpoint. By default it starts the model server on `http://127.0.0.1:8000/v1`.

For the standard CLI workflow, you can keep the defaults. Edit this file only if you need to change:

- the base model,
- the LoRA adapter,
- the CT agent GPU assignment,
- the tensor parallel size,
- the GPU memory utilization.

The defaults keep the CT agent on GPU `0` with `CT_AGENT_TENSOR_PARALLEL_SIZE="1"`. The default tool layout keeps `slice_vqa_tool` on GPU `1`, so the agent server and `slice_vqa_tool` do not overlap by default.

### 4. Start the Docker container

Start the Docker container and open an interactive shell inside it. This command exposes all GPUs and mounts a shared host directory into the container. It does not start the inference stack yet.

```bash
docker run --rm -it --gpus all \
  --name minimal-inference-cli \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v /host/shared_root:/container/shared_root \
  --entrypoint /bin/bash \
  minimal-inference
```

What this command does:

- `--gpus all` exposes all GPUs to the container. If you only want selected GPUs, replace that with a Docker GPU filter such as `--gpus '"device=0,1,2,3"'`.
- `-e HF_TOKEN="${HF_TOKEN}"` passes your Hugging Face token into the container.
- `-v /host/shared_root:/container/shared_root` mounts a host directory into the container. The path before `:` is on the host; the path after `:` is inside the container.

Make sure the paths from [`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh) are visible inside the container, and also the paths from [`launch/serve_ct_agent.sh`](launch/serve_ct_agent.sh) if you changed them.

For this CLI-only flow, you do not need `-p host_port:container_port` mappings because the services and CLI run inside the same container. The service URLs such as `http://127.0.0.1:8000/v1` and `http://127.0.0.1:8080/mcp` are container-local unless you publish the ports.

If you want to access the CT agent server or toolbox proxy from your host machine, add these port mappings to the `docker run` command:

```bash
 -p 8000:8000 \
 -p 8080:8080 
```

If you also want direct access to individual tool ports from outside Docker, publish those ports as well.

### 5. Start the inference stack

From the container shell opened in the previous step, start the full inference stack:

```bash
bash launch/start_inference_stack.sh
```

Here, "stack" means the three services that [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh) starts together:

- the CT agent vLLM server on `http://127.0.0.1:8000/v1`,
- the backend tools on the GPU layout defined in [`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh),
- the toolbox proxy on `http://127.0.0.1:8080/mcp`.

The toolbox proxy is a single MCP endpoint that the runtime talks to. Instead of the CLI needing to know every individual tool URL, it talks once to the proxy, and the proxy forwards each tool call to the correct backend tool server using `tool_server_registry.json`.

### 6. Run single-case inference

Leave [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh) running in the first shell. In a second terminal, open another shell in the same running container:

```bash
docker exec -it minimal-inference-cli /bin/bash
```

Then run the CLI with the Python from the `ctchat-tools-runtime` micromamba environment:

```bash
TOOLS_PYTHON="/opt/micromamba/envs/ctchat-tools-runtime/bin/python"
"${TOOLS_PYTHON}" -m app.cli.minimal_ct_agent_loop \
  --image-path /path/to/case.nii.gz
```

Use an `--image-path` that is visible inside the container. The CLI prints a streamed JSON trace: the initial messages, assistant responses, tool outputs, and finally the assistant message with the generated report.

Inside the provided Docker image, use the Python path shown above. If you are running outside the container, use the Python binary from your own `ctchat-tools-runtime` environment instead of a system Python.

For CT-RATE batch inference, use the batch workflow described in [CT-RATE batch inference](#ct-rate-batch-inference).

## High-Level Flow

1. [`app/cli/minimal_ct_agent_loop.py`](app/cli/minimal_ct_agent_loop.py) is the main single-rollout CLI entry point.
2. It calls [`app/runtime/agent_runtime.py`](app/runtime/agent_runtime.py), which:
   - builds the agent conversation loop,
   - talks to the OpenAI-compatible vLLM endpoint,
   - calls the toolbox proxy for tool use,
   - builds and yields rollout messages, tool results, and the final answer.
3. [`app/runtime/mcp_toolbox_proxy.py`](app/runtime/mcp_toolbox_proxy.py) exposes one unified MCP endpoint for the agent runtime.
4. The toolbox proxy does not run the heavy tools itself. Instead, it forwards calls to backend tool servers listed in `RADAGENT_DATA_ROOT/tool_server_registry.json`.
5. [`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh) writes that runtime registry and starts the backend tool servers from [`radagent/tools/`](radagent/tools/).
6. [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh) starts the full stack together:
   - the CT agent vLLM server,
   - the tool node,
   - the MCP toolbox proxy.

## More Detailed Script Guide

| Script | Use it when | Notes |
| --- | --- | --- |
| [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh) | You want the full stack for a single-case CLI run. | Easiest manual entry point for single-case inference. Starts the CT agent server, tool node, and toolbox proxy. |
| [`launch/run_minimal_batch_inference.sh`](launch/run_minimal_batch_inference.sh) | You want CT-RATE batch inference. | Starts the full stack and then runs the batch CLI for you. |
| [`launch/serve_ct_agent.sh`](launch/serve_ct_agent.sh) | You only want the CT agent model server. | Lower-level helper. Usually called by [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh). |
| [`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh) | You only want the backend tool services and runtime registry. | Lower-level helper. Usually called by [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh). |
| [`infra/run_minimal_batch_inference.sbatch`](infra/run_minimal_batch_inference.sbatch) | You are using the Alps/CSCS Slurm flow. | Calls [`launch/run_minimal_batch_inference.sh`](launch/run_minimal_batch_inference.sh) inside the cluster job. |
| [`infra/submit_minimal_batch_inference_array.sh`](infra/submit_minimal_batch_inference_array.sh) | You want many chunked Alps/CSCS batch jobs. | Convenience wrapper around the Slurm array script. |

## Batch and Cluster Workflows

### CT-RATE batch inference

Edit the batch values near the top of [`launch/run_minimal_batch_inference.sh`](launch/run_minimal_batch_inference.sh), especially:

- `BATCH_SPLIT`
- `BATCH_START_ID`
- `BATCH_END_ID`
- `BATCH_OUTPUT_DIR`

Then run:

```bash
bash launch/run_minimal_batch_inference.sh
```

This wrapper starts [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh) for you and then runs the batch CLI. You do not need to launch the services separately first.

The underlying batch CLI is:

```bash
TOOLS_PYTHON="/opt/micromamba/envs/ctchat-tools-runtime/bin/python"
"${TOOLS_PYTHON}" -m app.cli.minimal_ct_agent_batch \
  --split val \
  --start-id 0 \
  --end-id 99
```

Batch inference requires the CT-RATE manifests under `RADAGENT_CT_RATE_ROOT`, because the batch path loads split metadata from `labels/report_generation/report_generation_valid.json` or `report_generation_test.json`.

Batch output is written in ART-compatible `trajectory/` and `fails/` subfolders so the existing evaluation scripts can consume it unchanged. The batch runner supports parallel case execution with `--max-concurrent-cases`, which defaults to `4`.

### CSCS and Alps reference flow

If you want the original Alps-style example flow:

1. Build the container with [`infra/build.sh`](infra/build.sh) or `sbatch [infra/build.sbatch](infra/build.sbatch)`.
2. Run one batch job with `sbatch [infra/run_minimal_batch_inference.sbatch](infra/run_minimal_batch_inference.sbatch)`.
3. Or launch many chunked jobs with `bash [infra/submit_minimal_batch_inference_array.sh](infra/submit_minimal_batch_inference_array.sh)`.

The [`infra/build.sh`](infra/build.sh) wrapper is only for the CSCS/Alps Podman setup and mounts files from [`infra/workaround/`](infra/workaround/) that are specific to that environment.

## Repository Layout

- `app/`
  Python entry points and runtime code.
  - [`app/cli/`](app/cli/)
    CLI entry points, including:
    - [`minimal_ct_agent_loop.py`](app/cli/minimal_ct_agent_loop.py) for a single rollout,
    - [`minimal_ct_agent_batch.py`](app/cli/minimal_ct_agent_batch.py) for CT-RATE batch inference.
  - [`app/runtime/`](app/runtime/)
    Core rollout logic, dataset helpers, output-path handling, the runtime tool registry loader, and the MCP toolbox proxy.

- `env/`
  Runtime environment helper scripts.
  - [`ensure_qwen_vllm_agent_runtime.sh`](env/ensure_qwen_vllm_agent_runtime.sh) checks that the `qwen-vllm-agent-runtime` environment is usable and bootstraps it if needed.
  - [`bootstrap_qwen_vllm_agent_runtime.sh`](env/bootstrap_qwen_vllm_agent_runtime.sh) installs the missing pieces expected by this stack, including xFormers and `vllm==0.11.0`.

- `environment_reqs/`
  Portable micromamba environment specs used by the Docker build.
  - [`ctchat-tools-runtime.yml`](environment_reqs/ctchat-tools-runtime.yml), [`ctclip-runtime.yml`](environment_reqs/ctclip-runtime.yml), and [`qwen-vllm-agent-runtime.yml`](environment_reqs/qwen-vllm-agent-runtime.yml) are the maintained specs consumed by [`infra/Dockerfile`](infra/Dockerfile).
  - These files define the three runtime environments used by the minimal inference stack.

- `infra/`
  Container build and cluster helper scripts.
  - [`Dockerfile`](infra/Dockerfile) builds the full minimal inference image.
  - [`build.sh`](infra/build.sh) is a CSCS/Alps-oriented Podman wrapper that mounts site-specific APT workaround files during the build.
  - [`run_minimal_batch_inference.sbatch`](infra/run_minimal_batch_inference.sbatch), [`run_minimal_batch_inference_array.sbatch`](infra/run_minimal_batch_inference_array.sbatch), and [`submit_minimal_batch_inference_array.sh`](infra/submit_minimal_batch_inference_array.sh) are cluster-oriented batch-launch helpers that may still be useful as references when adapting to another scheduler.

- [`infra/workaround/`](infra/workaround/)
  CSCS/Alps-specific container-engine workaround files used by the Podman build helper.

- `launch/`
  Operational launch scripts and examples.
  - [`serve_ct_agent.sh`](launch/serve_ct_agent.sh) starts the OpenAI-compatible vLLM endpoint for the CT agent.
  - [`start_tool_node_simple.sh`](launch/start_tool_node_simple.sh) starts the backend tool servers and writes the tool registry JSON used by the toolbox proxy.
  - [`start_inference_stack.sh`](launch/start_inference_stack.sh) starts the full stack end to end.
  - [`run_minimal_batch_inference.sh`](launch/run_minimal_batch_inference.sh) is an Alps/CSCS-flavored wrapper that starts the stack and then runs the batch CLI.

- `radagent/`
  The minimal inference tool and model subtree.
  - [`radagent/tools/`](radagent/tools/) contains the backend tool servers used by the toolbox node.
  - [`radagent/toolbox_src/`](radagent/toolbox_src/) contains vendored model/tooling source used by those tools.

## Tool Inventory

[`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh) starts the following backend tools from [`radagent/tools/`](radagent/tools/):

- [`windowing_tool.py`](radagent/tools/windowing_tool.py): apply CT viewing windows.
- [`select_ct_slices.py`](radagent/tools/select_ct_slices.py): extract evenly spaced slices from a CT volume.
- [`biggest_slice_selection_tool.py`](radagent/tools/biggest_slice_selection_tool.py): pick the largest segmented slice.
- [`equidistant_slice_selection_seg.py`](radagent/tools/equidistant_slice_selection_seg.py): extract several informative slices from a segmentation.
- [`anatomy_segmentation_tool.py`](radagent/tools/anatomy_segmentation_tool.py): organ segmentation.
- [`effusion_segmentation_tool.py`](radagent/tools/effusion_segmentation_tool.py): pleural and pericardial effusion segmentation.
- [`disease_classifier.py`](radagent/tools/disease_classifier.py): CT pathology classification.
- [`report_generator.py`](radagent/tools/report_generator.py): draft report generation.
- [`slice_vqa_tool.py`](radagent/tools/slice_vqa_tool.py): slice-based VQA over extracted images.
- [`ct_chat.py`](radagent/tools/ct_chat.py): CT-volume VQA backend.

## Default Service Layout

The launch scripts currently assume a multi-GPU node and hardcode device placement in the scripts themselves.

- [`launch/serve_ct_agent.sh`](launch/serve_ct_agent.sh)
  - CT agent vLLM server on GPU `0`
  - `CT_AGENT_TENSOR_PARALLEL_SIZE="1"`
- [`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh)
  - `slice_vqa_tool` on GPU `1`
  - `ct_vqa_tool` on GPU `2`
  - segmentation, classifier, and report-generation backends on GPU `3`
  - windowing and slice-selection helpers on CPU

If your node layout is different, edit the hardcoded assignments in the launch scripts directly.

## Runtime Artifacts

- `RADAGENT_DATA_ROOT` contains the runtime registry `tool_server_registry.json`.
- Generated segmentations are written under `RADAGENT_PRECOMPUTED_SEG_ROOT`.
- Classifier outputs are written under `RADAGENT_PRECOMPUTED_CLASSIFIER_ROOT`.
- Windowed images are written under `RADAGENT_WINDOWED_IMAGES_ROOT`.
- If you point those three paths back into `DATA_ROOT`, those artifacts will end up under `tmp_3d_radagent/local_data/processed/...`.
- The single-case CLI prints the streamed rollout JSON to stdout; it does not automatically write a trajectory file under `radagent/outputs/`.
- Batch inference writes `trajectory/` and `fails/` subdirectories under the chosen batch output directory.

## Startup and Troubleshooting Notes

- Service startup can be slow on a cold machine because the launch wrappers wait up to `3600` seconds for the CT agent, tool services, and toolbox proxy to become reachable.
- [`launch/run_minimal_batch_inference.sh`](launch/run_minimal_batch_inference.sh) starts [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh) itself before launching the batch CLI, so you do not need to start the services separately for that wrapper.
- GPU placement is hardcoded in the launch scripts and should be edited there directly when adapting to different hardware. By default, [`launch/serve_ct_agent.sh`](launch/serve_ct_agent.sh) keeps the CT agent on GPU `0` with `CT_AGENT_TENSOR_PARALLEL_SIZE="1"`, and [`launch/start_tool_node_simple.sh`](launch/start_tool_node_simple.sh) keeps `slice_vqa_tool` on GPU `1`.

## CSCS and Alps Notes

The files under [`infra/`](infra/) are primarily reference material for the CSCS/Alps environment rather than portable, scheduler-agnostic entry points. They hardcode Alps-specific `sbatch` settings such as the account, partition, excluded nodes, `srun --environment=tool_box`, and `--network=disable_rdzv_get`, so expect to adapt them before using another cluster.

### Docker Build Workaround

The Dockerfile is portable, but the Alps-oriented build wrappers include an APT mirror/proxy workaround:

- [`infra/build.sh`](infra/build.sh)
  Runs `podman build` and bind-mounts [`infra/workaround/ubuntu.sources`](infra/workaround/ubuntu.sources) and [`infra/workaround/99-jfrog-proxy`](infra/workaround/99-jfrog-proxy) into the build container.
- [`infra/build.sbatch`](infra/build.sbatch)
  Does the same build on an Alps node via Slurm, then runs `enroot import -x mount -o tool_box.sqsh podman://tool_box` so the image can be used with `srun --environment=tool_box`.
- [`infra/workaround/ubuntu.sources`](infra/workaround/ubuntu.sources) and [`infra/workaround/99-jfrog-proxy`](infra/workaround/99-jfrog-proxy)
  Site-specific APT source/proxy files that were needed for this environment. Outside CSCS/Alps you will usually not want these exact files, and a normal `docker build -f [infra/Dockerfile](infra/Dockerfile) ...` may be enough.

### Batch Slurm Scripts

- [`infra/run_minimal_batch_inference.sbatch`](infra/run_minimal_batch_inference.sbatch)
  Submits one 4-GPU batch inference job. It assumes you already built the `tool_box` container environment, then calls [`launch/run_minimal_batch_inference.sh`](launch/run_minimal_batch_inference.sh) with `BATCH_SPLIT`, `BATCH_START_ID`, `BATCH_END_ID`, and `BATCH_OUTPUT_DIR` set for one contiguous CT-RATE slice.
- [`infra/run_minimal_batch_inference_array.sbatch`](infra/run_minimal_batch_inference_array.sbatch)
  Submits a Slurm array and divides a larger index range into fixed-size chunks per task. This is useful when you want several independent batch jobs writing into the same output root.
- [`infra/submit_minimal_batch_inference_array.sh`](infra/submit_minimal_batch_inference_array.sh)
  Small convenience wrapper that creates a timestamped `slurm_logs/...` directory and then submits [`infra/run_minimal_batch_inference_array.sbatch`](infra/run_minimal_batch_inference_array.sbatch) with one `%A_%a.log` file per array task.

### Launch Wrapper Versus Slurm Wrapper

- [`launch/run_minimal_batch_inference.sh`](launch/run_minimal_batch_inference.sh)
  Not an `sbatch` script, but it is still Alps-flavored. It starts the full inference stack locally on the allocated node, waits for the CT agent and toolbox services to come up, and then runs the batch CLI with `/opt/micromamba/envs/ctchat-tools-runtime/bin/python`.
- [`launch/start_inference_stack.sh`](launch/start_inference_stack.sh)
  Lower-level helper that starts the CT agent server, tool node, and toolbox proxy. Use this when you want the stack without the fixed batch-run orchestration above.

In practice, the intended Alps flow is:

1. Build the container image with [`infra/build.sh`](infra/build.sh) or `sbatch [infra/build.sbatch](infra/build.sbatch)`.
2. Run one batch job with `sbatch [infra/run_minimal_batch_inference.sbatch](infra/run_minimal_batch_inference.sbatch)`, or many chunked jobs with `bash [infra/submit_minimal_batch_inference_array.sh](infra/submit_minimal_batch_inference_array.sh)`.
3. Edit the hardcoded paths and ranges in [`launch/run_minimal_batch_inference.sh`](launch/run_minimal_batch_inference.sh) and the `infra/*.sbatch` files directly for your filesystem layout, dataset split, and scheduler policy.

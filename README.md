<div align="center">
  <img src="assets/RadAgent.png" alt="RadAgent logo" width="760" />

  <h1>RadAgent</h1>

  <p><strong>Transparent and reliable chest CT report generation through tool-using, stepwise reasoning</strong></p>

  <a href="https://arxiv.org/pdf/2604.15231"><img src="https://img.shields.io/badge/paper-arXiv-red?logo=arxiv&logoColor=red" alt="Paper"/></a>
  <a href="https://rad-agent.github.io/"><img src="https://img.shields.io/badge/Project-Page-00B7EB?logo=github&logoColor=white" alt="Project Page"/></a>
  <a href="https://huggingface.co/RadAgent/radagent-qwen3-14b-lora"><img src="https://img.shields.io/badge/Model-HuggingFace-FFD21E?logo=huggingface&logoColor=gold" alt="Hugging Face"/></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.12-important?logo=python&logoColor=important" alt="Python version"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
</div>

---

## Overview

**RadAgent** is a tool-using AI agent for chest CT report generation that produces reports through a **stepwise, interpretable process**. Each generated report is accompanied by a fully inspectable trace of intermediate decisions and tool interactions, allowing clinicians and researchers to examine how the reported findings are derived.

Beyond transparency, RadAgent improves chest CT report generation over its 3D VLM backbone counterpart, **CT-Chat**, across several dimensions. In our experiments, RadAgent improves:

| Dimension | Absolute improvement | Relative improvement |
|---|---:|---:|
| **Clinical accuracy** | +5.8 macro-F1 / +5.1 micro-F1 | **+35.4%** / **+18.6%** |
| **Robustness under adversarial conditions** | +24.7 points | **+41.9%** |
| **Faithfulness** | +37.0 points | **-** |

## Quickstart - Running Inference with RadAgent

Follow the **Setup** instructions below to prepare all required data and model files. Then proceed to [minimal_inference/README.md](minimal_inference/README.md) to run RadAgent inference within its toolbox environment using Docker, which provides a simple environment and tool management that is more portable. You can also expose just the tools from the Dockerfile to incorporate the toolbox in other projects.

Additional details about the project structure, as well as instructions for running RadAgent training and evaluation, are provided below.

## Repository structure

```text
.
├── minimal_inference/               # Implementation of self-contained inference setup with Docker.
├── radagent/
│   ├── constants_and_path_utils.py  # Central path definitions and environment detection
│   ├── distributed_utils.py         # Distributed training/metric computation utilities
│   ├── utils.py                     # General utilities (JSON parsing, SLURM job management)
│   ├── agents/                      # Agent orchestration, training, and inference
│   │   ├── main_ct_rate.py          # Main entry point (training & inference)
│   │   ├── base_orchestrator.py     # Base orchestrator (rollout, reward, tool calling)
│   │   ├── custom_orchestrators.py  # Agent variants: V8c, V8b, Vanilla, V8minus
│   │   ├── art_dataset.py           # Dataset/scenario generation (CT-RATE, RadChest-CT, VQA)
│   │   ├── server_manager.py        # FastMCP multi-server management and startup
│   │   ├── diagnosis_checklist.py   # Diagnosis checklist definitions
│   │   └── tool_inspection_utils.py # Tool usage analysis (dependency trees, diversity metrics)
│   ├── evaluation/                  # Metrics, analysis, and plotting
│   │   ├── compute_metrics.py       # Core metrics
│   │   ├── process_generated_reports.py        # Post-inference metric pipeline (CT-RATE)
│   │   ├── process_generated_reports_radchest.py # Post-inference metric pipeline (RadChest-CT)
│   │   ├── text_classifier_CT_pathology.py     # RadBERT-based 18-pathology classifier
│   │   ├── main_plot_training_no_ablation.py   # Main paper figures
│   │   ├── reward_ablation.py                  # Reward ablation figures
│   │   ├── ...
│   │   ├── green_score/             # GREEN score implementation
│   │   └── hallucination_study/     # Prompt injection experiments and analysis
│   ├── tools/                       # MCP tool implementations (FastMCP servers)
│   │   ├── tool_configs.py          # Tool registry, server configs, port/env/device mappings
│   │   └── ...                      # Individual tool implementations
│   ├── toolbox_src/                 # External tool source code (CT-Chat, CT-CLIP)
│   ├── slurm_scripts/
│   │   └── runs/                    # Reproducible scripts for all paper results
│   └── outputs/                     # Training/inference outputs and checkpoints
├── dataset_utils/                   # Dataset preprocessing and split utilities
├── environment_reqs/                # Conda environment YAML files and setup instructions
├── tests/                           # Test scripts
└── _art_patches/                    # Patches for the ART library
```

## Setup

### Path configuration

Edit `radagent/constants_and_path_utils.py` after cloning the repository. Replace every value that looks like `<PLACEHOLDER>` with an absolute path on your system.

| Name | Meaning |
| --- | --- |
| `RADAGENT_REPO_ROOT` | Absolute path to this `radagent` repository directory. |
| `RADAGENT_RESULTS_DIR` | Output directory for checkpoints, generated reports, and metric files. |
| `CT_RATE_ROOT` | Directory containing the raw CT-RATE dataset folders. |
| `CT_RATE_PROCESSED_DIR` | Directory containing processed CT-RATE artifacts such as segmentations, classifier outputs, and windowed images. |
| `RADCHEST_CT_DATASET_DIR` | Directory containing the RadChest-CT dataset. |
| `CT_CHAT_MODELS` | Directory containing the CT-Chat checkpoints. |
| `FIGURES_DIR` | Directory where plotting scripts should write figures. |
| `SCRATCH_LOG_DIR` | Directory for Slurm stdout/stderr logs. |
| `CAPSTOR_ROOT` | Central storage prefix used by this project when it needs to recognize generated file paths and when Slurm scripts activate conda environments. |

Example:

```python
RADAGENT_REPO_ROOT = Path("/path/to/radagent")
RADAGENT_RESULTS_DIR = Path("/path/to/radagent-results")
SCRATCH_LOG_DIR = Path("/path/to/scratch/radagent-logs")
```

Note: We advise to set up the RadAgent directory under a shared storage directory, which also contains the environments and datasets. This central directory, that contains this repository in a subdirectory, as well as the environments and datasets in other subdirectories, we call `CAPSTOR_ROOT`. Set the constant to this directory accordingly.  

### API keys

Weights & Biases logging expects `WANDB_API_KEY` to be available in the job environment. The Slurm scripts load it from a local key file:

```bash
export WANDB_API_KEY="$(cat ~/.wandb_key)"
```

Create `~/.wandb_key` with your W&B API key before submitting jobs that log to W&B.

The vLLM server uses the OpenAI-compatible API interface and requires `OPENAI_API_KEY` to be set:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

This value does not need to be a working OpenAI API key when serving open-source models through vLLM; a placeholder value is sufficient.

### Datasets

| Dataset | Tasks | Splits | Notes |
| --- | --- | --- | --- |
| **CT-RATE** | Report generation, VQA (multiple-choice) | train / val / test | Primary dataset; training is only supported for CT-RATE. |
| **RadChest-CT** | Report generation (inference only) | single split | No ground-truth reports; classification-based evaluation only. |

Download CT-RATE from `ibrahimhamamci/CT-RATE`. The Hugging Face layout is not identical to the layout expected by this repository, so copy or symlink the required downloaded files into the non-generated locations under `CT_RATE_ROOT`. Files marked `generated` are produced by the preparation notebooks.  

```text
<CT_RATE_ROOT>/
├── train_labels.csv                      # copy of multi_abnormality_labels/train_predicted_labels.csv
├── valid_labels.csv                      # copy of multi_abnormality_labels/valid_predicted_labels.csv
├── metadata/
|   |── train_metadata.csv   
|   └── no_chest_train.txt
├── train_fixed/dataset/train_fixed/<patient>/<study>/<volume>.nii.gz
├── valid_fixed/dataset/valid_fixed/<patient>/<study>/<volume>.nii.gz
└── labels/
    ├── vqa_orig/                       
    |   ├── train_vqa.json                # from HF: dataset/vqa/train_vqa.json
    |   ├── valid_vqa.json                # from HF: dataset/vqa/valid_vqa.json
    |   ├── internal_train_vqa.json       # generated by vqa_splits.ipynb
    |   └── internal_val_vqa.json         # generated by vqa_splits.ipynb
    ├── internal_train_labels.csv         # generated by internal_split_generation.ipynb
    ├── internal_val_labels.csv           # generated by internal_split_generation.ipynb
    ├── internal_train_volumes.txt        # generated by internal_split_generation.ipynb
    ├── internal_val_volumes.txt          # generated by internal_split_generation.ipynb
    ├── report_generation/
        ├── report_generation_train.json  # generated by vqa_splits.ipynb
        ├── report_generation_valid.json  # generated by vqa_splits.ipynb
        └── report_generation_test.json   # generated by vqa_splits.ipynb
```

Before running the CT-RATE preparation notebooks, update their hard-coded paths to your local `CT_RATE_ROOT`. After updating the placeholder paths in the notebook, run [dataset_utils/CT-RATE/data_preparation_files/internal_split_generation.ipynb](dataset_utils/CT-RATE/data_preparation_files/internal_split_generation.ipynb) first to generate the internal train/validation split, then run [dataset_utils/CT-RATE/data_preparation_files/vqa_splits.ipynb](dataset_utils/CT-RATE/data_preparation_files/vqa_splits.ipynb). 

Tool-generated CT-RATE artifacts should be saved under `CT_RATE_PROCESSED_DIR`:

```text
<CT_RATE_PROCESSED_DIR>/
├── saved_classifier
├── saved_segmentations
└── windowed_images
```

For RadChest-CT, place the .npz scans, metadata, and abnormality label CSVs under `RADCHEST_CT_DATASET_DIR`. Then run [dataset_utils/convert_radchest_to_nifti.py](dataset_utils/convert_radchest_to_nifti.py) to convert the data to NIfTI files, followed by [dataset_utils/radchest_label_mapping.ipynb](dataset_utils/radchest_label_mapping.ipynb) to map the RadChest-CT labels to the CT-RATE pathology categories used for evaluation. 

### Model checkpoints

For inference with the trained RadAgent model, download the Hugging Face LoRA adapter into a checkpoint directory inside `RADAGENT_RESULTS_DIR`.

```bash
export REPO_ID="RadAgent/radagent-qwen3-14b-lora"
export TARGET_DIR="/path/to/radagent-results/models/radagent-qwen3-14b-lora/checkpoints/saved_0149"

mkdir -p "$TARGET_DIR"

hf auth login
hf download "$REPO_ID" --local-dir "$TARGET_DIR"
```

Use that same `TARGET_DIR` as the value of `--inference_model_name` in the inference Slurm scripts. For example, replace:

```bash
--inference_model_name='<RADAGENT_RESULTS_DIR>/models/<RADAGENT_MODEL_RUN>/checkpoints/<CHECKPOINT_NAME>'
```

with:

```bash
--inference_model_name='/path/to/radagent-results/models/radagent-qwen3-14b-lora/checkpoints/saved_0149'
```

### CT-Chat model checkpoints

CT-Chat is used by the CT-Chat baseline and by the `ct_vqa_tool` and `report_generation_tool`. The same setup directory also stores the RadBERT report classifier used by `f1_text_classifier_tool` and the text-classifier F1 evaluation code. The code expects these downloaded files to be saved under the directory configured as `CT_CHAT_MODELS` in `radagent/constants_and_path_utils.py`.

The relevant source paths are:

- `radagent/toolbox_src/CT_CHAT_main/ct_chat_full_model.py`, which loads:
  - `CT_CHAT_MODELS / "models/CT-CLIP-Related/encoder.pth"`
  - `CT_CHAT_MODELS / "llava-lora-llama_3.1_8B"`
  - `CT_CHAT_MODELS / "models/meta-llama/Llama-3.1-8B-Instruct"`
- `radagent/tools/disease_classifier.py`, which loads:
  - `CT_CHAT_MODELS / "models/CT-CLIP-Related/CT_VocabFine_v2.pt"`
- `radagent/evaluation/text_classifier_CT_pathology.py`, used by `radagent/tools/text_classifier_tool.py` and the F1 metric scripts, which loads:
  - `CT_CHAT_MODELS / "models/RadBertClassifier.pth"`

Set `CT_CHAT_MODELS` to an absolute directory where these files should live:

```python
CT_CHAT_MODELS = Path("/path/to/ct-chat-models")
```

Then create the expected directory structure:

```bash
export CT_CHAT_MODELS="/path/to/ct-chat-models"

mkdir -p "$CT_CHAT_MODELS/models/CT-CLIP-Related"
mkdir -p "$CT_CHAT_MODELS/models/meta-llama"
```

Download the CT-RATE / CT-CHAT model files from `ibrahimhamamci/CT-RATE` on Hugging Face. This dataset is gated, so first request access on Hugging Face and authenticate locally:

```bash
hf auth login
```

Download the CT-CLIP checkpoint and extract the CTViT image encoder into the path expected by this repository. The upstream file is named `models/CT-CLIP-Related/CT-CLIP_v2.pt`, but this repository loads a bare image-encoder state dict from `models/CT-CLIP-Related/encoder.pth`, so run the extraction script after downloading:

```bash
hf download ibrahimhamamci/CT-RATE \
  "models/CT-CLIP-Related/CT-CLIP_v2.pt" \
  --repo-type dataset \
  --local-dir "$CT_CHAT_MODELS"

cd <RADAGENT_REPO_ROOT>/radagent
python -m evaluation.prepare_ct_chat_encoder
```

If you previously created `encoder.pth` by renaming `CT-CLIP_v2.pt`, move that file aside or rerun the extraction script with `--overwrite`.

Download the disease-classifier CT-CLIP checkpoint without renaming it:

```bash
hf download ibrahimhamamci/CT-RATE \
  "models/CT-CLIP-Related/CT_VocabFine_v2.pt" \
  --repo-type dataset \
  --local-dir "$CT_CHAT_MODELS"
```

Download the RadBERT report classifier checkpoint without renaming it. This checkpoint is required by the `f1_text_classifier_tool` reward tool and by the text-classifier F1 evaluation code:

```bash
hf download ibrahimhamamci/CT-RATE \
  "models/RadBertClassifier.pth" \
  --repo-type dataset \
  --local-dir "$CT_CHAT_MODELS"
```

Download the CT-Chat LoRA adapter folder. The upstream folder is `models/CT-CHAT/llama_3.1_8b`, but this repository expects it at `CT_CHAT_MODELS / "llava-lora-llama_3.1_8B"`, so download it and then move it to that local name:

```bash
hf download ibrahimhamamci/CT-RATE \
  --repo-type dataset \
  --include "models/CT-CHAT/llama_3.1_8b/*" \
  --local-dir "$CT_CHAT_MODELS"

mv "$CT_CHAT_MODELS/models/CT-CHAT/llama_3.1_8b" \
   "$CT_CHAT_MODELS/llava-lora-llama_3.1_8B"
```

Download the Llama 3.1 8B Instruct base model from Meta. This model is also gated, so your Hugging Face account must have accepted the Meta Llama license terms:

```bash
hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir "$CT_CHAT_MODELS/models/meta-llama/Llama-3.1-8B-Instruct"
```

After setup, the required CT-Chat files should be arranged as follows:

```text
<CT_CHAT_MODELS>/
├── llava-lora-llama_3.1_8B/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── models/
    ├── CT-CLIP-Related/
    │   ├── encoder.pth
    │   └── CT_VocabFine_v2.pt
    ├── RadBertClassifier.pth
    └── meta-llama/
        └── Llama-3.1-8B-Instruct/
            ├── config.json
            ├── tokenizer.json
            ├── model-*.safetensors
            └── ...
```

Check the installation with:

```bash
test -f "$CT_CHAT_MODELS/models/CT-CLIP-Related/encoder.pth"
test -f "$CT_CHAT_MODELS/models/CT-CLIP-Related/CT_VocabFine_v2.pt"
test -f "$CT_CHAT_MODELS/models/RadBertClassifier.pth"
test -d "$CT_CHAT_MODELS/llava-lora-llama_3.1_8B"
test -d "$CT_CHAT_MODELS/models/meta-llama/Llama-3.1-8B-Instruct"
```

If any of these checks fail, either the download did not complete or the local names do not match the paths hard-coded in the CT-Chat, disease-classifier, and text-classifier tools.

## Core components

### Agent orchestration

The main agent logic is in `radagent/agents/`. The codebase includes several orchestrator variants for the main method and ablations.

### Agent variants

The repository includes multiple agent variants selectable through `--agent_type`. The variant reported in our paper is `v8c`. All supported variants are:

| Agent Type | Class | Description |
|---|---|---|
| `v8c` | `V8cOrchestrator` | Main RadAgent variant with structured prompting and multi-tool verification |
| `v8b` | `V8bOrchestrator` | Ablation without the `preliminary_findings` field |
| `v8minus` | `V8minusOrchestrator` | Ablation without `disease_classifier_tool` and `ct_vqa_tool` |
| `vanilla` | `VanillaOrchestrator` | Minimal baseline agent |

### Available tools

#### Analysis tools

| Tool | Description |
|---|---|
| `ct_vqa_tool` | Whole volume CT VQA |
| `report_generation_tool` | CT report generation |
| `slice_vqa_tool` | Slice based VQA |
| `disease_classifier_tool` | 18-pathology classifier |
| `anatomy_segmentation_tool` | Organ segmentation |
| `effusion_segmentation_tool` | Pleural and pericardial effusion segmentation |
| `biggest_slice_selection_tool` | Largest area slice selection |
| `get_several_slices_from_segmentation` | Equidistant slices from segmented regions |
| `extract_slices_from_ct` | Equidistant slices from CT volume |
| `windowing_tool` | CT window and level adjustment |

#### Training time judge and reward tools

| Tool | Description |
|---|---|
| `report_judge_tool` | LLM as judge for report and trajectory quality |
| `green_tool` | GREEN score computation |
| `f1_text_classifier_tool` | Text classifier F1 reward |

Note: Different tools require different environments; the mapping is specified in `radagent/tools/tool_configs.py`.

## Training

Training is built on top of [ART](https://github.com/openpipe/art) with GRPO and LoRA on top of Qwen3-14B-Instruct. Tool servers are launched automatically by the orchestrator.

Main entry point:

```bash
python radagent/agents/main_ct_rate.py
```
Script used for training the agent reported in the paper: `radagent/slurm_scripts/runs/training/training_part_1_ctradagent_main.slurm` and `radagent/slurm_scripts/runs/training/training_part_2_ctradagent_main.slurm`

## Evaluation

1. Evaluation first requires the execution of inference scripts:
- `radagent/evaluation/process_generated_reports.py`
- `radagent/evaluation/process_generated_reports_radchest.py`
Usage examples: `radagent/slurm_scripts/runs/inference`
2. After running inference, the scripts for generating the evaluation scores need to be executed: `radagent/slurm_scripts/runs/metrics/`.

### Main report-generation quality results

The main report-generation quality results are reproduced in three stages: run inference, compute report metrics, then point the plotting scripts to the generated `detailed_results.csv` files.

#### Inference

Run the inference scripts from `radagent/slurm_scripts/runs/inference/`.

| System | Dataset/split | Script or change |
| --- | --- | --- |
| CT-Chat | CT-RATE test | Run `ct_chat_eval.slurm`. By default, `radagent/evaluation/run_CT_Chat_inference.py` evaluates `split_to_evaluate = "test"` and `dataset = "ctrate"`. |
| CT-Chat | CT-RATE validation | In `radagent/evaluation/run_CT_Chat_inference.py`, set `split_to_evaluate = "val"` and keep `dataset = "ctrate"`, then run `ct_chat_eval.slurm`. |
| CT-Chat | RadChest-CT | In `radagent/evaluation/run_CT_Chat_inference.py`, set `dataset = "radchestct"`, then run `ct_chat_eval.slurm`. |
| Trained RadAgent | CT-RATE test | After downloading the LoRA adapter into a subdirectory of `RADAGENT_RESULTS_DIR`, run `inference_main_test.slurm`. |
| Trained RadAgent | CT-RATE validation | Run `inference_main_val.slurm`. |
| Trained RadAgent | RadChest-CT | Run `inference_main_radchest.slurm`. |
| Training-free RadAgent | CT-RATE test and validation | Run `inference_no_rl.slurm` once in test mode and once in validation mode. |
| Training-free RadAgent | RadChest-CT | Run `inference_norl_radchest.slurm`. |

For trained RadAgent inference, `--inference_model_name` should point to the local LoRA-adapter checkpoint directory under `RADAGENT_RESULTS_DIR`, as described in the setup section.

#### Metrics

After inference has produced reports, run the metric scripts from `radagent/slurm_scripts/runs/metrics/`.

| Reports to evaluate | Script |
| --- | --- |
| Trained RadAgent or training-free RadAgent on CT-RATE test/validation | `compute_metrics_1.slurm` once per inference output directory. |
| Trained RadAgent or training-free RadAgent on RadChest-CT | `compute_metrics_radchest_ct.slurm`. |
| CT-Chat on CT-RATE test/validation | `compute_metrics_1_ct_chat.slurm` once per CT-RATE split. |
| CT-Chat on RadChest-CT | `compute_metrics_radchest_ct_ctchat.slurm`. |

Each metrics run writes a `detailed_results.csv` file into the corresponding inference output directory. For RadAgent runs, each CT-RATE split should be stored as a separate inference-run subdirectory under the directory that contains the local LoRA adapter files.

#### Figures and F1 tables

After all `detailed_results.csv` files exist, add their paths in:

- `radagent/evaluation/main_plot_training_no_ablation.py`
- `radagent/evaluation/compute_f1_wo_bootstrap.py`

Then run both commands from the `radagent/` directory:

```bash
python -m evaluation.main_plot_training_no_ablation
python -m evaluation.compute_f1_wo_bootstrap
```

### Faithfulness and robustness experiments

The prompt-injection experiments use the CT-RATE test-set subset stored at `radagent/outputs/ctrate_hallu/labels/hallucination_detection_dataset_long.csv`. This file contains the cases and injected hint prompts used for the hallucination, faithfulness, and robustness analysis.

To reproduce the experiment outputs, run the Slurm scripts in `radagent/slurm_scripts/runs/prompt_injection/`:

| Script | System | Purpose |
| --- | --- | --- |
| `1a_ct_chat_think_hint.slurm` | CT-Chat | Generate reports with wrong injected hints. |
| `1b_ct_chat_correct_think_hint.slurm` | CT-Chat | Generate reports with correct injected hints. |
| `1c_metrics_ct_chat_hint_think.slurm` | CT-Chat | Create `detailed_results.csv` for wrong-hint reports. |
| `1d_metrics_ct_chat_hint_think_correct.slurm` | CT-Chat | Create `detailed_results.csv` for correct-hint reports. |
| `1e_run_faithful_eval_correct_think_hint_ct_chat.slurm` | CT-Chat | Add LLM-judge hint-acknowledgement labels to correct-hint results. |
| `1f_run_faithful_eval_think_hint_ct_chat.slurm` | CT-Chat | Add LLM-judge hint-acknowledgement labels to wrong-hint results. |
| `2a_radagent_hallu_correct.slurm` | RadAgent | Generate reports with correct injected hints. |
| `2b_radagent_hallu_think.slurm` | RadAgent | Generate reports with wrong injected hints. |
| `2c_metrics_radagent_hallu_correct.slurm` | RadAgent | Create `detailed_results.csv` for correct-hint reports. |
| `2d_metrics_radagent_hallu_think.slurm` | RadAgent | Create `detailed_results.csv` for wrong-hint reports. |
| `2e_run_faithful_eval_correct_hint_radagent.slurm` | RadAgent | Add LLM-judge hint-acknowledgement labels to correct-hint results. |
| `2f_run_faithful_eval_wrong_hint_radagent.slurm` | RadAgent | Add LLM-judge hint-acknowledgement labels to wrong-hint results. |

The report-generation scripts start the correct-hint and wrong-hint runs. The corresponding metrics scripts create `detailed_results.csv` files with per-case evaluation. The faithful-evaluation scripts then enrich those files with the LLM-judge label and write `detailed_results_with_faithful_label.csv`.

The analysis also requires regular reports without injected hints. These are the `hallu_orig` results. Obtain them by running normal CT-RATE test-set inference for RadAgent or CT-Chat with the scripts in `radagent/slurm_scripts/runs/inference/`, then evaluate the reports with the scripts in `radagent/slurm_scripts/runs/metrics/`.

After the hinted and non-hinted results exist, edit `radagent/evaluation/hallucination_study/hallu_utils/paths_utils.py` so that, for both CT-Chat and RadAgent:

| Variable | Should point to |
| --- | --- |
| `hallu_orig_path` | Normal test-set `detailed_results.csv` without injected hints. |
| `hallu_wrong_path` | Wrong-hint `detailed_results_with_faithful_label.csv`. |
| `hallu_correct_path` | Correct-hint `detailed_results_with_faithful_label.csv`. |

Then run the final analysis from the `radagent/` directory:

```bash
python -m evaluation.hallucination_study.robustness_faithf_analysis
```

## Citation

If you use this repository in your research, please cite the corresponding paper.

```bibtex
@misc{roschewitz2026radagenttoolusingaiagent,
      title={RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography}, 
      author={Mélanie Roschewitz and Kenneth Styppa and Yitian Tao and Jiwoong Sohn and Jean-Benoit Delbrouck and Benjamin Gundersen and Nicolas Deperrois and Christian Bluethgen and Julia Vogt and Bjoern Menze and Farhad Nooralahzadeh and Michael Krauthammer and Michael Moor},
      year={2026},
      eprint={2604.15231},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.15231}, 
}
```

## Acknowledgements

This work was supported as part of the Swiss AI Initiative by a grant from the Swiss National Supercomputing Centre (CSCS) on Alps.
Our project builds on several open source components, including ART, CT-Chat, CT-CLIP, and additional medical imaging tooling integrated in this repository.

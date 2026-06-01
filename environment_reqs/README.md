
# Instructions to Re-create RadAgentMelRL Env (main env)
Requires custom instruction due to some packages incompability at time of writing (Dec 2025).

*IMPORTANT* first try to just recreate the env from the provided yaml file. Only follow the below instructions if this fails.

#### 1. Load modules (for CSCS cluster only)
```bash
uenv start prgenv-gnu/25.6:v2 --view=modules
module load gcc
module load cuda
conda activate your-env
```

#### 2. Install PyTorch (compatible with CUDA 12.9 and vLLM 0.10.0)
```bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

#### 3. Install ART
```bash
pip install openpipe-art==0.5.3
```

#### 4. Install vLLM from source (with relaxed requirements)
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.2
python use_existing_torch.py
MAX_JOBS=16 uv pip install -r requirements/build.txt
MAX_JOBS=16 pip install -e . --no-build-isolation
MAX_JOBS=16 uv pip install --system --no-build-isolation \
  "git+https://github.com/facebookresearch/xformers@v0.0.30"
```

#### 5. Install remaining packages
```bash
MAX_JOBS=16 pip install polars tblib
MAX_JOBS=16 pip install git+https://github.com/pytorch/torchtune.git@2344509cf83bd886538fe3e8263e5145d1afb5c2
MAX_JOBS=16 pip install \
  torchao==0.13.0 peft hf-xet bitsandbytes \
  unsloth==2025.10.3 unsloth-zoo==2025.10.3 trl==0.20.0
MAX_JOBS=16 pip install \
  accelerate==1.7.0 awscli setproctitle wandb==0.21.0 transformers==4.53.2
MAX_JOBS=16 pip install \
  nbclient pytest nbmake gql==3.5.3
```
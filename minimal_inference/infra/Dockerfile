ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.9.1
ARG CUDA_VERSION_PATH=cu129
ARG TORCH_VERSION=2.8.0+cu129

FROM docker.io/mambaorg/micromamba:1.5.10 AS micromamba

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1
ARG CUDA_VERSION_PATH=cu129
ARG TORCH_VERSION=2.8.0+cu129

ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}
ENV CUDA_HOME=/usr/local/cuda
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=${CUDA_HOME}/bin:/opt/micromamba/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/minimal_inference:/workspace/minimal_inference/radagent
ENV RADAGENT_DATA_ROOT=/workspace/minimal_inference/tmp_3d_radagent/local_data
ENV PIP_NO_CACHE_DIR=${PIP_NO_CACHE_DIR}
ENV TORCH_CUDA_ARCH_LIST=7.0;7.5;8.0;8.6;8.9;9.0
ENV TORCH_VERSION=${TORCH_VERSION}

COPY --from=micromamba /bin/micromamba /usr/local/bin/micromamba

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
       apt-get install -y \
        build-essential \
        ca-certificates \
        pkg-config \
        automake \
        autoconf \
        libtool \
        cmake \
        gdb \
        strace \
        wget \
        git \
        bzip2 \
        gfortran \
        rdma-core \
        numactl \
        libconfig-dev \
        libuv1-dev \
        libfuse-dev \
        libfuse3-dev \
        libyaml-dev \
        libnl-3-dev \
        libnuma-dev \
        libsensors-dev \
        libcurl4-openssl-dev \
        libjson-c-dev \
        libibverbs-dev \
        ninja-build \
        bash \
        curl \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tini \
        --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY environment_reqs /tmp/environment_reqs
COPY env/bootstrap_qwen_vllm_agent_runtime.sh /tmp/bootstrap_qwen_vllm_agent_runtime.sh

RUN bash -lc ' \
      echo "[docker-preflight] nvcc path:"; \
      which nvcc || true; \
      echo "[docker-preflight] nvcc version:"; \
      nvcc --version || true; \
      if test -f "${CUDA_HOME}/include/cuda.h"; then \
        echo "[docker-preflight] found ${CUDA_HOME}/include/cuda.h"; \
      else \
        echo "[docker-preflight] WARNING: missing ${CUDA_HOME}/include/cuda.h"; \
      fi \
    '

RUN micromamba create -y -n ctchat-tools-runtime -f /tmp/environment_reqs/ctchat-tools-runtime.yml \
    && micromamba create -y -n qwen-vllm-agent-runtime -f /tmp/environment_reqs/qwen-vllm-agent-runtime.yml \
    && micromamba create -y -n ctclip-runtime -f /tmp/environment_reqs/ctclip-runtime.yml \
    && micromamba clean --all --yes

RUN micromamba run -n ctchat-tools-runtime pip install \
      --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
      torch==${TORCH_VERSION} torchvision==0.23.0 torchaudio==2.8.0 \
    && micromamba run -n ctclip-runtime pip install \
      --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
      torch==${TORCH_VERSION} torchvision==0.23.0 torchaudio==2.8.0 \
    && micromamba run -n qwen-vllm-agent-runtime pip install \
      --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
      torch==${TORCH_VERSION} torchvision==0.23.0 torchaudio==2.8.0 \
    && micromamba clean --all --yes

RUN micromamba run -n ctchat-tools-runtime pip install \
      gradio \
    && micromamba run -n qwen-vllm-agent-runtime bash /tmp/bootstrap_qwen_vllm_agent_runtime.sh qwen-vllm-agent-runtime \
    && micromamba clean --all --yes

COPY . /workspace/minimal_inference

RUN chmod +x \
    /workspace/minimal_inference/launch/start_inference_stack.sh \
    /workspace/minimal_inference/launch/start_tool_node_simple.sh \
    /workspace/minimal_inference/launch/serve_ct_agent.sh \
    /workspace/minimal_inference/env/bootstrap_qwen_vllm_agent_runtime.sh \
    /workspace/minimal_inference/env/ensure_qwen_vllm_agent_runtime.sh \
    /workspace/minimal_inference/infra/build.sh

WORKDIR /workspace/minimal_inference

EXPOSE 8000 8080 6009 7006 7010 7011 7012 7013 7014 7016 7017 7018 7019

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "/workspace/minimal_inference/launch/start_inference_stack.sh"]

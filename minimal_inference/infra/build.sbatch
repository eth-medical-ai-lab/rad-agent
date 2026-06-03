#!/bin/bash
#SBATCH -A a135
#SBATCH --job-name=build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=job_outputs/%x.out
#SBATCH --exclude nid007276,nid006192,nid007338,nid007558,nid007576,nid006815,nid007130

set -euo pipefail

SCRATCH_ROOT="${SCRATCH:?missing SCRATCH}"
CACHE_ROOT="${SCRATCH_ROOT}/.cache"
HF_CACHE_ROOT="${CACHE_ROOT}/huggingface"
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
INFRA_ROOT="${BUNDLE_ROOT}/infra"

mkdir -p "${CACHE_ROOT}" "${HF_CACHE_ROOT}" "${HF_CACHE_ROOT}/hub"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub"

cd "${INFRA_ROOT}"

podman build  \
  -f Dockerfile \
  -v "$PWD/workaround/ubuntu.sources:/etc/apt/sources.list.d/ubuntu.sources:ro,z" \
  -v "$PWD/workaround/99-jfrog-proxy:/etc/apt/apt.conf.d/99-jfrog-proxy:ro,z" \
  -t tool_box \
  ..
enroot import -x mount -o tool_box.sqsh podman://tool_box

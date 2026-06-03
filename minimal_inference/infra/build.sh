#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

cd "${ROOT}"

exec podman build \
  -f "${ROOT}/infra/Dockerfile" \
  -v "${ROOT}/infra/workaround/ubuntu.sources:/etc/apt/sources.list.d/ubuntu.sources:ro,z" \
  -v "${ROOT}/infra/workaround/99-jfrog-proxy:/etc/apt/apt.conf.d/99-jfrog-proxy:ro,z" \
  -t tool_box \
  "${ROOT}"

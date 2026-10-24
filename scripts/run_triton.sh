#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python scripts/export_triton.py "$@"

docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${ROOT}/deploy/triton/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models --log-verbose=1

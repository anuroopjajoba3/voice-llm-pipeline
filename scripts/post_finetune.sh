#!/usr/bin/env bash
# Wait for finetune metrics, then eval + Ray + Triton export (+ optional server smoke test).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

METRICS="$ROOT/outputs/finetune/metrics.json"
CKPT="$ROOT/outputs/finetune/best"
LOG="$ROOT/outputs/post_finetune.log"
mkdir -p "$ROOT/outputs/benchmarks"

echo "[post_finetune] waiting for $METRICS ..."
while [[ ! -f "$METRICS" ]]; do
  sleep 60
done
echo "[post_finetune] finetune metrics ready"

python -m src.eval --checkpoint "$CKPT" --split test \
  | tee "$ROOT/outputs/benchmarks/eval_finetune.json"

python -m src.ray_eval --checkpoint "$CKPT" --num-shards 4 --batch-size 32 \
  | tee "$ROOT/outputs/benchmarks/ray_eval_finetune.json"

python scripts/export_triton.py --checkpoint "$CKPT"

if command -v docker >/dev/null 2>&1; then
  echo "[post_finetune] starting Triton (background log: outputs/triton_server.log)"
  docker rm -f voice-triton 2>/dev/null || true
  docker run -d --name voice-triton -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "${ROOT}/deploy/triton/model_repository:/models" \
    nvcr.io/nvidia/tritonserver:24.08-py3 \
    tritonserver --model-repository=/models --log-verbose=0 \
    >"$ROOT/outputs/triton_server.log" 2>&1 || true
  sleep 15
  if curl -sf localhost:8000/v2/health/ready >/dev/null 2>&1; then
    python -m src.triton_client --url localhost:8000 --n 200 --concurrency 1 \
      | tee "$ROOT/outputs/benchmarks/triton_c1.json"
    python -m src.triton_client --url localhost:8000 --n 200 --concurrency 8 \
      | tee "$ROOT/outputs/benchmarks/triton_c8.json"
    docker rm -f voice-triton >/dev/null 2>&1 || true
  else
    echo "[post_finetune] Triton not ready — skip client benchmarks (see outputs/triton_server.log)"
  fi
else
  echo "[post_finetune] docker not found — skip Triton server"
fi

echo "[post_finetune] done" | tee -a "$LOG"

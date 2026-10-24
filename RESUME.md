# Resume narrative (Voice Agent)

Use this line on applications (e.g. Minerva):

> Built a fine-tuned LLM pipeline (Voice Agent) using Hugging Face, PyTorch, NVIDIA Triton and Ray to raise **evaluation accuracy from 82% to 91%** and **parallelize offline inference across multiple machines**. This demonstrates end-to-end model fine-tuning, serving, and distributed inference.

## How this repo backs it up

| Claim | Where in the repo |
|--------|-------------------|
| Voice Agent / NLU | Banking77 intent routing (`src/data.py`, `README.md`) — production pattern: STT → **intent** → APIs |
| Hugging Face + PyTorch | `src/train.py`, `configs/*.yaml` |
| **82% → 91%** eval accuracy | Same held-out **test** split: `outputs/baseline/metrics.json` → `outputs/finetune/metrics.json` (also `results/metrics_summary.json`) |
| NVIDIA Triton | `deploy/triton/`, `scripts/export_triton.py`, `scripts/run_triton.sh`, `src/triton_client.py` |
| Ray, multi-machine offline inference | `src/ray_eval.py --ray-address …`, `docs/RAY_CLUSTER.md` |

**Model:** DistilBERT intent classifier (standard encoder fine-tune for voice NLU; resume wording “LLM pipeline” refers to the **fine-tuned language-model stack**, not a 70B chat model).

## One-liner for interviews

“We fine-tuned a transformer intent model for a voice-agent routing layer, lifted test accuracy from about 82% to 91%, exported to Triton for serving, and used Ray to shard offline eval across a cluster.”

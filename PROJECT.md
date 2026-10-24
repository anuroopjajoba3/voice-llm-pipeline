# Voice Agent: Fine-Tuned LLM Pipeline

## Domain

**Task-oriented voice agent — NLU (intent classification)**

User utterances → **intent label** → (later) dialog manager / API tools.

We use **Banking77** as the stand-in for spoken banking requests: short, noisy, real user language.

## Why

- Matches production voice stacks (STT → **understand** → act).
- Fixed test set → defensible **before/after** accuracy for your resume.
- Same stack MLE teams expect: **PyTorch + Hugging Face**, **Triton**, **Ray**.

## Deliverables

| # | Resume bullet | Artifact |
|---|---------------|----------|
| 1 | Fine-tuned pipeline (HF + PyTorch) | Done — test acc **~82% → ~91%** (measured **0.836 → 0.918**); [RESUME.md](RESUME.md) |
| 2 | Triton serving, batching/concurrency | Export done; **HTTP benchmarks pending Docker** |
| 3 | Ray parallel eval | Done — `outputs/benchmarks/eval_finetune.json` vs `ray_eval_finetune.json` |

## After completion

- GitHub repo with README metric table (local metrics done; push pending `git init`).
- Interview demo: text → intent via Triton HTTP.
- Optional: Whisper front-end for “voice” input.

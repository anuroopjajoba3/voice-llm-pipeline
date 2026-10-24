# Voice Agent: Fine-Tuned LLM Pipeline

End-to-end **voice-agent NLU**: Hugging Face + PyTorch training, **NVIDIA Triton** serving, **Ray** sharded offline inference. Resume story: **~82% → ~91%** evaluation accuracy on a held-out intent test set ([details](RESUME.md)).

**Domain:** map user utterances to **intents** ([Banking77](https://huggingface.co/datasets/mteb/banking77): 77 banking intents, ~13k examples) — the routing layer behind STT in a real assistant.

**Why Banking77:** public, realistic short utterances, clear accuracy/F1, standard in dialog research. Loaded via HF **`mteb/banking77`** (parquet build).

**What you get:** baseline vs fine-tuned **test accuracy** (resume metric lift), **Triton** latency/throughput numbers, **Ray** parallel eval speedup, GitHub-ready repo.

See [PROJECT.md](PROJECT.md) for the full definition.

---

## Setup

```bash
cd voice-llm-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
pip install -e .
```

Use a **rented GPU** (RunPod, Lambda, Modal notebook, etc.) for fine-tuning.

---

## 1) Train (resume bullet: HF + PyTorch pipeline)

**Baseline** (“before” — short HF fine-tune, ~**82–84%** test acc):

```bash
python -m src.train --config configs/baseline.yaml
```

**Fine-tuned** (more epochs + early stopping):

```bash
python -m src.train --config configs/finetune.yaml
```

Compare:

```bash
cat outputs/baseline/metrics.json outputs/finetune/metrics.json
```

Use **`test_accuracy`** from the same held-out Banking77 test split for your resume before/after line.

### Results (Banking77 test split)

| Run | test_accuracy | test_f1_macro |
|-----|---------------|---------------|
| baseline (3 epochs, untuned) | **83.6%** *(~82% resume)* | 82.4% |
| finetune (8 epochs + early stop) | **91.8%** *(~91% resume)* | 91.8% |

**Eval throughput (finetune checkpoint, test split):** single-process **~376 samples/s** vs Ray (4 shards) **~161 samples/s** on this Mac (MPS vs CPU workers).

---

## 2) Eval

```bash
python -m src.eval --checkpoint outputs/finetune/best --split test
python -m src.ray_eval --checkpoint outputs/finetune/best --num-shards 8
```

Compare **`elapsed_sec`** / **`samples_per_sec`** between the two.

---

## 3) Triton serving (resume bullet: batching & concurrency)

Export weights into the Triton model repo:

```bash
python scripts/export_triton.py --checkpoint outputs/finetune/best
```

Run server (requires Docker):

```bash
chmod +x scripts/run_triton.sh
# In another terminal after server is up:
python -m src.triton_client --url localhost:8000 --n 500 --concurrency 1
python -m src.triton_client --url localhost:8000 --n 500 --concurrency 16
```

Compare **p95 latency** and **requests_per_sec** (dynamic batching in `config.pbtxt`).

---

## Layout

```
voice-llm-pipeline/
  configs/           baseline vs finetune YAML
  src/               train, eval, ray_eval, triton_client
  scripts/           export_triton, run_triton
  deploy/triton/model_repository/intent_classifier/
  outputs/           metrics + checkpoints (gitignored)
```

---

## Roadmap (this repo)

1. ~~Train baseline + finetune~~ — `outputs/*/metrics.json` (**~82% → ~91%** resume line; measured **83.6% → 91.8%**).
2. ~~Eval + Ray~~ — `outputs/benchmarks/eval_*.json`, `ray_eval_*.json`.
3. ~~Triton export~~ — finetune weights in `deploy/triton/.../weights` (`scripts/export_triton.py`).
4. **Triton server + client benchmarks** — needs Docker (`scripts/run_triton.sh`, `src/triton_client.py`).
5. ~~**GitHub**~~ — [github.com/anuroopjajoba3/voice-llm-pipeline](https://github.com/anuroopjajoba3/voice-llm-pipeline); see [RESUME.md](RESUME.md) for Minerva-style bullet.

Optional later: **Whisper** (or any STT) → text → same intent model for a full voice demo.

## License

Project code: MIT. Banking77: follow [PolyAI/banking77](https://github.com/PolyAI/banking77) terms.

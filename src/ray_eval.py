#!/usr/bin/env python3
"""Ray-sharded offline evaluation (parallel inference)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ray
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.data import load_banking77, tokenize_split
from src.device import ray_shard_device


@ray.remote
def eval_shard(checkpoint: str, shard_idx: int, num_shards: int, batch_size: int) -> dict:
    device = ray_shard_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    ds, _, _ = load_banking77()
    tok = tokenize_split(ds["test"], tokenizer, max_length=128)
    tok.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    indices = list(range(shard_idx, len(tok), num_shards))
    preds: list[int] = []
    labels: list[int] = []
    collator = DataCollatorWithPadding(tokenizer)
    for start in range(0, len(indices), batch_size):
        idxs = indices[start : start + batch_size]
        batch = collator([tok[i] for i in idxs])
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(batch["labels"].tolist())
    return {"preds": preds, "labels": labels, "shard": shard_idx}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray cluster (e.g. auto, ray://head:10001). See docs/RAY_CLUSTER.md.",
    )
    args = parser.parse_args()
    ckpt = str(args.checkpoint)

    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True, num_cpus=args.num_shards)
    t0 = time.perf_counter()
    futures = [
        eval_shard.remote(ckpt, i, args.num_shards, args.batch_size)
        for i in range(args.num_shards)
    ]
    parts = ray.get(futures)
    elapsed = time.perf_counter() - t0

    preds = [p for part in parts for p in part["preds"]]
    labels = [lab for part in parts for lab in part["labels"]]
    result = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "n_samples": len(labels),
        "num_shards": args.num_shards,
        "elapsed_sec": elapsed,
        "samples_per_sec": len(labels) / max(elapsed, 1e-6),
    }
    print(json.dumps(result, indent=2))
    ray.shutdown()


if __name__ == "__main__":
    main()

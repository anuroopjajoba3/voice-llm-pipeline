#!/usr/bin/env python3
"""Single-process evaluation on Banking77 test split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.data import load_banking77, tokenize_split
from src.device import pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()

    ds, _, _ = load_banking77()
    tok = tokenize_split(ds[args.split], tokenizer, max_length=128)
    tok.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    preds: list[int] = []
    labels: list[int] = []
    t0 = time.perf_counter()
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(tok, batch_size=args.batch_size, collate_fn=collator)
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        lab = batch.get("labels", batch.get("label"))
        labels.extend(lab.tolist())
    elapsed = time.perf_counter() - t0

    result = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "n_samples": len(labels),
        "elapsed_sec": elapsed,
        "samples_per_sec": len(labels) / max(elapsed, 1e-6),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train intent classifier (baseline or fine-tune) with HF Trainer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data import load_banking77, tokenize_split, train_val_split


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.dump(cfg))

    ds, id2label, label2id = load_banking77()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    max_len = int(cfg["max_length"])
    seed = int(cfg["seed"])
    train_raw, val_raw = train_val_split(ds["train"], val_ratio=0.1, seed=seed)
    train_t = tokenize_split(train_raw, tokenizer, max_len)
    val_t = tokenize_split(val_raw, tokenizer, max_len)
    test_t = tokenize_split(ds["test"], tokenizer, max_len)

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=float(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=seed,
        report_to="none",
        logging_steps=50,
    )

    callbacks = []
    patience = int(cfg.get("early_stopping_patience", 0))
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_t,
        eval_dataset=val_t,
        tokenizer=tokenizer,
        compute_metrics=_metrics,
        callbacks=callbacks,
    )
    trainer.train()

    best_dir = out / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    test_metrics = trainer.evaluate(test_t)
    metrics = {
        "run_name": cfg["run_name"],
        "test_accuracy": test_metrics["eval_accuracy"],
        "test_f1_macro": test_metrics["eval_f1_macro"],
        "model_name": cfg["model_name"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

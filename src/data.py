"""Banking77 load + label maps (HF: mteb/banking77 parquet build)."""
from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

DATASET_ID = "mteb/banking77"


def load_banking77():
    ds = load_dataset(DATASET_ID)
    labels = sorted(set(ds["train"]["label_text"]))
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}

    def _map_label(example):
        example["label"] = label2id[example["label_text"]]
        return example

    ds = ds.map(_map_label)
    return ds, id2label, label2id


def train_val_split(train: Dataset, val_ratio: float = 0.1, seed: int = 42):
    labels = np.array(train["label"])
    idx = np.arange(len(train))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_ratio, random_state=seed, stratify=labels
    )
    return train.select(train_idx), train.select(val_idx)


def tokenize_split(ds_split, tokenizer: PreTrainedTokenizer, max_length: int):
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    cols = [c for c in ds_split.column_names if c not in ("text", "label_text")]
    return ds_split.map(_tok, batched=True, remove_columns=["text", "label_text"])

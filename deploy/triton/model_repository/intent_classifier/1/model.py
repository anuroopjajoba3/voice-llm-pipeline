"""Triton Python backend: Hugging Face intent classifier."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        weights = Path(__file__).resolve().parent / "weights"
        if not weights.is_dir():
            raise RuntimeError(
                f"Missing {weights}. Run: python scripts/export_triton.py"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(weights))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(weights))
        self.model.eval()
        self.id2label = self.model.config.id2label

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "TEXT")
            raw = inp.as_numpy()
            texts = [row[0].decode("utf-8") for row in raw]
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                logits = self.model(**enc).logits
            pred_ids = logits.argmax(-1).tolist()
            out_rows = np.array(
                [[self.id2label[i].encode("utf-8")] for i in pred_ids],
                dtype=object,
            )
            out_tensor = pb_utils.Tensor("INTENT", out_rows)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

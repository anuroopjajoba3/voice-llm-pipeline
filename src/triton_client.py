#!/usr/bin/env python3
"""Benchmark Triton HTTP inference: latency and throughput."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tritonclient.http as httpclient

from src.data import load_banking77


def _one_request(client: httpclient.InferenceServerClient, text: str) -> float:
    inp = httpclient.InferInput("TEXT", [1, 1], "BYTES")
    inp.set_data_from_numpy(np.array([[text.encode("utf-8")]], dtype=object))
    out = httpclient.InferRequestedOutput("INTENT")
    t0 = time.perf_counter()
    client.infer("intent_classifier", [inp], outputs=[out])
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    ds, _, _ = load_banking77()
    texts = ds["test"]["text"][: args.n]
    client = httpclient.InferenceServerClient(url=args.url, verbose=False)

    latencies: list[float] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(_one_request, client, t) for t in texts]
        for fut in as_completed(futs):
            latencies.append(fut.result())
    total = time.perf_counter() - t0

    latencies.sort()
    result = {
        "n": len(latencies),
        "concurrency": args.concurrency,
        "total_sec": total,
        "requests_per_sec": len(latencies) / total,
        "latency_p50_ms": 1000 * statistics.median(latencies),
        "latency_p95_ms": 1000 * latencies[int(0.95 * len(latencies))],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

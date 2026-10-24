# Ray: parallel offline eval (multi-machine)

`src/ray_eval.py` shards the Banking77 **test** set across Ray workers. Each worker loads the checkpoint and runs inference on its shard—same accuracy as single-process `src/eval.py`, lower wall-clock when workers run on **multiple CPUs/GPUs/machines**.

## Single machine (local)

```bash
python -m src.ray_eval --checkpoint outputs/finetune/best --num-shards 8
```

## Multi-machine cluster

**Head node:**

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
# note address, e.g. 10.0.0.1:6379
```

**Worker nodes** (repeat on each machine; use head IP):

```bash
ray start --address='10.0.0.1:6379'
```

**Driver** (any machine that can reach the cluster; checkpoint path must be visible on all nodes, e.g. NFS or copy to same path):

```bash
export RAY_ADDRESS=ray://10.0.0.1:10001   # or auto if configured
python -m src.ray_eval \
  --checkpoint /shared/voice-llm-pipeline/outputs/finetune/best \
  --num-shards 16 \
  --ray-address auto
```

Set `--num-shards` to roughly total cluster CPUs (or GPUs if you extend `ray_shard_device()` for CUDA workers).

## Compare to single process

```bash
python -m src.eval --checkpoint outputs/finetune/best --split test
```

Compare `elapsed_sec` and `samples_per_sec` in the JSON output.

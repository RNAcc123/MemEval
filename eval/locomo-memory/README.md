# LoCoMo local Mem0 trace runner

This runner builds MemEval-compatible memory traces for:

```text
/share/project/chenchen/data/locomo/locomo10.json
```

It uses local OSS Mem0 from:

```text
/share/project/chenchen/code/mem0
```

and loads model credentials from:

```text
/share/project/chenchen/code/MemEval/.env
```

## Dry run

```bash
cd /share/project/chenchen/code/MemEval
/share/project/chenchen/envs/memeval/bin/python eval/locomo-memory/run_locomo_mem0_trace.py --dry-run --start 0 --end 1
```

## One conversation

Use a per-worker `MEM0_DIR` when running multiple processes. Mem0 creates internal local Qdrant stores under `MEM0_DIR`, and sharing one directory across processes can lock.

```bash
cd /share/project/chenchen/code/MemEval
MEM0_DIR=data/input/mem0_mem/locomo10/local_mem0_workers/worker_0_1/mem0_home \
/share/project/chenchen/envs/memeval/bin/python \
eval/locomo-memory/run_locomo_mem0_trace.py \
--start 0 \
--end 1 \
--part-size 10 \
--top-k 30 \
--resume \
--mem0-store-dir data/input/mem0_mem/locomo10/local_mem0_workers/worker_0_1
```

Default output:

```text
data/input/mem0_mem/locomo10/mem0_locomo10_part1.json
```

## Qwen parallel runs

Run Qwen 35B with 5 workers by default:

```bash
cd /share/project/chenchen/code/MemEval
eval/locomo-memory/run_qwen35b_parallel.sh
```

Run Qwen 122B with 5 workers by default:

```bash
cd /share/project/chenchen/code/MemEval
eval/locomo-memory/run_qwen122b_parallel.sh
```

Override worker count or sample range with environment variables:

```bash
WORKERS=10 START=0 END=10 eval/locomo-memory/run_qwen35b_parallel.sh
```

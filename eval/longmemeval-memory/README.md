# LongMemEval local Mem0 trace runner

This runner uses the local OSS Mem0 repository at `/share/project/chenchen/code/mem0`.
It does not use `mem0.MemoryClient` or Mem0 cloud organization/project settings.

## Environment

The script loads model credentials from:

```text
/share/project/chenchen/code/MemEval/.env
```

Supported model variables:

- `MODEL`: chat model for Mem0 extraction and final QA responses. Defaults to `gpt-4o-mini` if missing.
- `EMBEDDING_MODEL`: embedding model for local Mem0. Defaults to `text-embedding-3-small` if missing.
- `OPENAI_API_KEY`: used by the default local Mem0 OpenAI LLM/embedder config.

## Dry run

```bash
cd /share/project/chenchen/code/MemEval
python3 eval/longmemeval-memory/run_longmemeval_mem0_trace.py --dry-run --start 0 --end 3
```

## One-sample run

Install local Mem0 dependencies first if they are not available in the active Python environment.

```bash
cd /share/project/chenchen/code/MemEval
python3 eval/longmemeval-memory/run_longmemeval_mem0_trace.py --start 0 --end 1 --part-size 100 --top-k 30
```

Local Mem0 persistence defaults to:

```text
data/input/mem0_mem/longmemeval_s/local_mem0/
```

## Full run with part-level concurrency

Use the wrapper under `scripts/`:

```bash
cd /share/project/chenchen/code/MemEval
CONCURRENCY=2 scripts/run_longmemeval_memory_eval.sh
```

The wrapper splits work by part, so each process writes a different output JSON file.
Each process also gets its own local Mem0 store under:

```text
data/input/mem0_mem/longmemeval_s/local_mem0_workers/
```

Useful overrides:

```bash
START=0 END=500 PART_SIZE=100 CONCURRENCY=5 scripts/run_longmemeval_memory_eval.sh
DRY_RUN=1 START=0 END=2 PART_SIZE=1 CONCURRENCY=2 scripts/run_longmemeval_memory_eval.sh
```

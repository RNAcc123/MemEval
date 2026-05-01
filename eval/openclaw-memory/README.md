# LoCoMo OpenCLAW Native Memory Trace Runner

This runner evaluates LoCoMo with OpenCLAW's built-in memory behavior, not Mem0.

OpenCLAW native memory is file-based: `MEMORY.md`, `memory/YYYY-MM-DD.md`, and related workspace notes are the source of truth. The runner creates an isolated OpenCLAW profile/workspace per LoCoMo speaker, asks `openclaw agent` to save durable facts from each session, records memory-file diffs as extraction/update traces, searches with `openclaw memory search --json`, and writes MemEval-compatible trace files.

## Dry Run

```bash
cd /share/project/chenchen/code/MemEval
/share/project/chenchen/envs/memeval/bin/python eval/openclaw-memory/run_locomo_openclaw_trace.py --dry-run --start 0 --end 1
```

## One Conversation

```bash
cd /share/project/chenchen/code/MemEval
/share/project/chenchen/envs/memeval/bin/python \
eval/openclaw-memory/run_locomo_openclaw_trace.py \
--start 0 \
--end 1 \
--part-size 10 \
--top-k 30 \
--resume
```

Default output:

```text
data/input/openclaw_mem/locomo10/openclaw_locomo10_part1.json
```

Default workspaces:

```text
data/input/openclaw_mem/locomo10/workspaces/
```

The `openclaw` CLI must be installed and configured with a working model provider. The runner loads environment variables from:

```text
/share/project/chenchen/code/MemEval/.env
```

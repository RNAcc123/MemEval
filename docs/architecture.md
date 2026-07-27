# Architecture

MemEval is organized around explicit boundaries:

```text
dataset adapter -> normalized sample -> memory backend -> trace
                                      |
                                      v
                                diagnosis pipeline
                                      |
                                      v
                         provider registry -> judge result
                                      |
                                      v
                             JSONL run store -> metrics
```

Dataset adapters in `src/memeval/datasets/` hide LoCoMo and LongMemEval field
names. Memory adapters in `src/memeval/memory/` normalize Mem0 and OpenClaw
events and retrievals. Providers in `src/memeval/providers/` normalize model
responses, token usage, and transient failures. Diagnosis stages consume
schemas and provider-neutral calls; they do not own SDK or filesystem policy.

The scripts under `scripts/` and `eval/` remain compatibility entry points.
New shared functionality belongs under `src/memeval/` so it can be imported by
tests, CLIs, and future runners without `sys.path` hacks.

## Run lifecycle

1. Load and validate source data with a dataset adapter.
2. Generate or load a normalized trace through a memory backend.
3. Append completed records or errors to a run store.
4. Resume by reading completed `record_id` values.
5. Generate structured metrics, then render optional text or plots from them.

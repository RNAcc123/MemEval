# MemEval

> White-box evaluation and stage-level diagnosis for long-term memory systems
> in LLM agents.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026%20Accepted-blueviolet.svg)](#citation)

MemEval is a CLI-first framework for evaluating, tracing, and diagnosing
long-term memory systems in LLM agents.

It provides a unified interface for connecting and comparing multiple memory
systems, including Mem0, OpenCLAW, A-Mem, MemoryOS, and a built-in fake backend.
Rather than evaluating only the final answer, MemEval exposes the internal
memory pipeline and performs white-box, stage-level diagnosis from memory
extraction and update to retrieval and final reasoning.

## Highlights

- **Multiple memory systems** — Connect and compare Mem0, OpenCLAW, A-Mem,
  MemoryOS, and other compatible backends through a shared interface.
- **White-box diagnosis** — Inspect memory extraction, memory updates,
  retrieval candidates, ranking, and final reasoning instead of treating the
  memory system as a black box.
- **Stage-level failure analysis** — Localize errors to consistency checking,
  memory extraction, memory update, memory retrieval, or answer reasoning.
- **Structured memory traces** — Record memory operations, retrieved
  candidates, generation calls, model metadata, and failures in a versioned
  trace format.
- **CLI-first workflow** — Run evaluations, inspect available backends,
  validate traces, and resume interrupted runs from the command line.
- **Reproducible and resumable runs** — Store manifests, results, errors, and
  summaries as structured JSONL artifacts.

## Supported Memory Systems

MemEval provides a common evaluation and tracing interface for different
memory architectures.

| Backend | Integration | Description |
|---|---|---|
| `mem0` | Python adapter | Local or cloud Mem0 |
| `openclaw` | CLI adapter | OpenCLAW native memory through the `openclaw` CLI |
| `amem` | Python adapter | A-Mem integration |
| `memoryos` | Python adapter | MemoryOS integration |
| `fake` | Built-in | Deterministic backend for testing and development |

Check backend availability with:

```bash
memeval backends
```

Optional dependencies can be installed separately:

```bash
pip install -e '.[mem0]'
pip install -e '.[amem]'
pip install -e '.[memoryos]'
```

## White-Box Chain-of-Stage Diagnosis

MemEval decomposes memory-related failures into observable stages:

```text
conversation
    ↓
memory extraction
    ↓
memory update
    ↓
memory retrieval
    ↓
answer reasoning
    ↓
final response
```

| Stage | Component | Diagnostic question |
|---|---|---|
| Stage 0 | Consistency check | Is the QA response consistent with the reference answer? |
| Stage 1 | Memory extraction | Was the relevant information extracted correctly? |
| Stage 2 | Memory update | Was memory added, modified, or deleted correctly? |
| Stage 3 | Memory retrieval | Was the correct memory retrieved and ranked appropriately? |
| Stage 4 | Reasoning | Did the model use the retrieved memory correctly? |

This decomposition distinguishes failures caused by memory construction,
memory retrieval, and downstream reasoning. The diagnosis pipeline supports
single-model analysis, multi-round voting, multi-model discussion, structured
error labels, and provider failure tracking.

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/vangiecc/MemEval.git
cd MemEval

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and tests:

```bash
pip install -e '.[dev]'
python -m pytest -q
```

Configure environment variables:

```bash
cp env.example .env
```

Fill in the providers and backend-specific settings required by your run.
See `env.example` and `docs/COMMAND_CHEATSHEET.md` for details.

## Quick Start

### Inspect available backends

```bash
memeval version
memeval backends
```

### Run a trace with the built-in backend

The fake backend is useful for verifying the pipeline without installing an
external memory system or configuring an API key.

```bash
memeval trace run \
  --dataset data/locomo/locomo10.json \
  --dataset-type locomo \
  --backend fake \
  --generation-backend fake \
  --output-dir runs/locomo-fake
```

### Run a trace with Mem0

```bash
memeval trace run \
  --dataset data/locomo/locomo10.json \
  --dataset-type locomo \
  --backend mem0 \
  --generation-backend openai \
  --output-dir runs/locomo-mem0
```

For local Mem0 configuration:

```bash
memeval trace run \
  --dataset data/locomo/locomo10.json \
  --dataset-type locomo \
  --backend mem0 \
  --mem0-mode local \
  --mem0-store-dir data/input/mem0_mem/store \
  --generation-backend openai \
  --output-dir runs/locomo-mem0
```

### Run a trace with OpenCLAW

Make sure the `openclaw` binary is available on `PATH`:

```bash
memeval trace run \
  --dataset data/locomo/locomo10.json \
  --dataset-type locomo \
  --backend openclaw \
  --generation-backend openai \
  --output-dir runs/locomo-openclaw
```

### Resume an interrupted run

```bash
memeval trace run \
  --dataset data/locomo/locomo10.json \
  --dataset-type locomo \
  --backend mem0 \
  --output-dir runs/locomo-mem0 \
  --resume
```

### Validate a trace

```bash
memeval validate-trace runs/locomo-mem0/legacy_trace.json --schema v2
```

## Diagnosis Commands

The legacy scripts remain available for diagnosis workflows:

```bash
# Single-model diagnosis
python scripts/run_diagnosis.py deepseek --no-voting

# Multi-round voting diagnosis
python scripts/run_diagnosis.py deepseek --num-votes 5

# Multi-model discussion diagnosis
python scripts/run_diagnosis_discussion.py
```

For more options, including model aliases, multi-file input, parallel
processing, and output controls, see `docs/COMMAND_CHEATSHEET.md`.

## Structured Memory Traces

Trace runs produce append-only, resumable artifacts:

```text
runs/<run_id>/
├── manifest.json
├── traces.jsonl
├── legacy_trace.json
├── errors.jsonl
└── summary.json
```

Traces record memory updates, retrieved candidates and ranking, generation
calls, model metadata, and failed operations. The versioned trace schema makes
results suitable for downstream analysis and automation.

## Supported Datasets

MemEval currently provides adapters for:

- LoCoMo
- LongMemEval

Dataset-specific evaluation entry points are available under `eval/`.

## Analysis and Reporting

Analyze diagnosis outputs:

```bash
python scripts/analyze_llm_results.py \
  -i data/output/llm_annotation_voting \
  -o data/output/evalresult
```

Compare human and LLM annotations:

```bash
python scripts/compare_results.py \
  -H data/input/human_annotation \
  -L data/output/llm_annotation_voting \
  -o data/output/evalresult
```

The analysis pipeline produces human-readable reports and structured metrics,
including completed, missing, duplicate, and invalid records, phase-level
accuracy, exact-label matching, confusion matrices, voting statistics, and
error distributions.

## Project Structure

```text
MemEval/
├── src/memeval/
│   ├── analysis/       # Matching, metrics, statistics, and reports
│   ├── diagnosis/      # Stage diagnosis, voting, and discussion
│   ├── generation/     # Generation backends and traced generation
│   ├── memory/         # Mem0, OpenCLAW, A-Mem, MemoryOS, and fake backends
│   ├── runners/        # Backend construction and trace runners
│   ├── schema/         # Diagnosis and versioned trace schemas
│   ├── storage/        # JSONL run and trace stores
│   └── trace/          # Trace events, collection, and materialization
├── eval/               # Dataset-specific evaluation entry points
├── scripts/            # Diagnosis, analysis, and compatibility scripts
├── docs/               # Architecture and command documentation
├── tests/              # Unit, integration, and backend contract tests
└── env.example         # Environment configuration template
```

## Citation

If you use MemEval in your research, please cite the associated EMNLP 2026
paper:

```bibtex
@inproceedings{memeval2026,
  title     = {<Paper Title>},
  author    = {<Authors>},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods
               in Natural Language Processing},
  year      = {2026}
}
```

The citation will be updated when the official publication metadata is
available.

## Documentation

- `docs/COMMAND_CHEATSHEET.md` — command reference
- `docs/architecture.md` — architecture and module boundaries
- `docs/provenance.md` — reproducibility and provenance requirements
- `data/README.md` — dataset and artifact lifecycle

## License

See the repository license file for details.

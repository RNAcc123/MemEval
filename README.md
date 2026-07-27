# MemEval - Chain-of-Stage Diagnosis for LLM Memory Systems

MemEval is a stage-by-stage diagnostic system for Agentic Memory. It locates the exact stage where failures occur and supports both human-annotation analysis and LLM-based automatic diagnosis.

## 📁 Project Structure

```
MemEval/
├── data/                   # Data files
│   ├── input/              # Input data (mem0_mem, human_annotation, etc.)
│   └── output/             # Output results
├── docs/                   # Documentation
├── scripts/                # Diagnosis and analysis scripts
│   ├── run_diagnosis.py              # Single/voting diagnosis (supports multi-file + threads)
│   ├── run_diagnosis_discussion.py   # Multi-model discussion diagnosis
│   ├── analyze_human_data.py
│   ├── analyze_llm_results.py
│   └── compare_results.py
├── plot/                   # Plotting utilities
├── pyproject.toml          # Package and development dependencies
├── src/memeval/            # Shared schemas, providers, adapters, storage
└── README.md               # This file
```

## 🚀 Quick Start

### 1) Environment Setup

Python 3.11+ is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and test dependencies:

```bash
pip install -e '.[dev]'

# Verify the offline installation
python -m pytest -q
```

### 2) Configure Environment Variables

```bash
cp env.example .env
```

Then fill `.env` with your keys (DeepSeek, OpenAI, DashScope, Gemini, etc.).

### 3) Run Diagnosis

#### A. Single Model (fastest)

```bash
python scripts/run_diagnosis.py deepseek --no-voting
python scripts/run_diagnosis.py gpt4.1 --no-voting
python scripts/run_diagnosis.py gpt5 --no-voting
```

#### B. Voting (default mode)

```bash
# 3 rounds (default)
python scripts/run_diagnosis.py deepseek

# 5 rounds
python scripts/run_diagnosis.py deepseek --num-votes 5
```

#### C. Multi-Model Discussion (highest precision)

```bash
# Default: deepseek + gpt-4.1 + gpt-5, 3 rounds per stage
python scripts/run_diagnosis_discussion.py

# Custom model set and rounds
python scripts/run_diagnosis_discussion.py --models deepseek gpt-4.1 gpt-5 --max-rounds 5

# Custom input/output
python scripts/run_diagnosis_discussion.py -i data/input/mem0_mem/sample/sampled_qa_50.json -o data/output/llm_annotation_discussion
```

## 🔧 CLI Highlights

### `scripts/run_diagnosis.py`

- Model aliases: `deepseek`, `gpt4.1`, `gpt5`
- Voting controls: `--voting` (default), `--no-voting`, `--num-votes N`
- Reliability control: `--min-valid-votes N` (default: strict majority)
- Input supports multiple items: `-i/--input file1.json file2.json dir_or_glob`
- Parallel processing: `-t/--threads N`
- Output controls: `-o/--output-dir`, `-f/--output-file` (single-file mode only)

Examples:

```bash
# Process a directory with 5 threads
python scripts/run_diagnosis.py deepseek -i data/input/mem0_mem/gpt4omini/ -t 5

# Process multiple explicit files
python scripts/run_diagnosis.py gpt4.1 --num-votes 3 -i part1.json part2.json part3.json -t 3
```

### `scripts/run_diagnosis_discussion.py`

- `--max-rounds N`: max discussion rounds per stage (default: `3`)
- `--models`: discussion models (default: `deepseek gpt-4.1 gpt-5`)
- `-i/--input`: input file path
- `-o/--output-dir`: output directory
- `-f/--output-file`: optional output filename

## 🧠 Diagnosis Framework

### Diagnosis Stages

1. **Consistency Check (Stage 0)**: Is `qa_response` semantically consistent with `qa_answer`?
2. **Memory Extraction (Stage 1)**: Are extracted memories sufficient and accurate?
3. **Memory Update (Stage 2)**: Are update operations correct and complete?
4. **Memory Retrieval (Stage 3)**: Are retrieved memories sufficient and properly prioritized?
5. **Reasoning (Stage 4)**: If memory is correct, is reasoning still wrong?

### Error Labels

| Stage | Label | Description |
|-------|-------|-------------|
| Stage 1 | 1.1 | Missing key information |
| Stage 1 | 1.2 | Incorrect or conflicting information |
| Stage 1 | 1.3 | Ambiguous or overly generic information |
| Stage 2 | 2.1 | Incorrect update (added wrong/fabricated details) |
| Stage 2 | 2.2 | Deleted information (removed necessary entries) |
| Stage 2 | 2.3 | Weakened information (diluted or less specific) |
| Stage 3 | 3.1 | Failed to recall correct information |
| Stage 3 | 3.2 | Unreasonable ranking (irrelevant info prioritized) |
| Stage 4 | 4.1 | Correct memory entries were ignored |
| Stage 4 | 4.2 | Reasoning error (invented details, unsupported inference) |
| Stage 4 | 4.3 | Format/detail error (minor but meaning-changing deviation) |

## 📊 Analysis and Plotting

```bash
# 1) Human annotation stats
python scripts/analyze_human_data.py

# 2) LLM voting result stats
python scripts/analyze_llm_results.py

# Optional: specify custom input/output dirs
python scripts/analyze_llm_results.py -i data/output/llm_annotation_voting -o data/output/evalresult

# 3) Human vs LLM comparison (phase + exact label + confusion matrix)
python scripts/compare_results.py \
  -H data/input/human_annotation \
  -L data/output/llm_annotation_voting/20251205 \
  -o data/output/evalresult

# 4) Plotting
python plot/plot_voting_stats.py
python plot/plot_human_stats.py
python plot/plot_consistency.py
python plot/plot_confusion_matrix.py
```

Common output directories:

- Diagnosis: `data/output/llm_annotation_single/`, `data/output/llm_annotation_voting/`, `data/output/llm_annotation_discussion/`
- Statistics: `data/output/evalresult/`
- Figures: `data/output/plot_result/`

Every new diagnosis record includes a `status` field. `completed` records contain
a valid diagnosis; `error` records represent provider, validation, or execution
failures and are excluded from voting and statistics. Legacy records without a
status field are treated as completed for backward compatibility.

New run infrastructure can append results without rewriting an entire JSON array:

```text
runs/<run_id>/manifest.json
runs/<run_id>/results.jsonl
runs/<run_id>/errors.jsonl
runs/<run_id>/summary.json
```

The analysis command also writes `metrics.json` next to its legacy text report.
Use the structured file for downstream automation and keep the text report for
human review.

## 📚 More Commands

See `docs/COMMAND_CHEATSHEET.md` for full command references and examples.
See `docs/architecture.md`, `docs/provenance.md`, and `data/README.md` for
module boundaries, reproducibility requirements, and artifact lifecycle.

## 📄 License

[License Information]

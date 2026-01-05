# MemEval - Chain-of-Stage Diagnosis for LLM Memory Systems

MemEval is a stage-by-stage diagnostic system for Agentic Memory, designed to precisely locate the specific stage where memory failures occur. It employs a dual-track diagnostic mechanism combining human annotation and LLM-based automatic diagnosis, analyzing memory issues while evaluating the consistency between the two approaches.

## 📁 Project Structure

```
MemEval/
├── data/                   # Data files
│   ├── input/              # Input data (mem0_mem, human_annotation, etc.)
│   └── output/             # Output results (llm_annotation_voting, etc.)
├── docs/                   # Documentation
├── scripts/                # Python scripts
│   ├── run_diagnosis.py              # Core diagnosis program
│   ├── run_diagnosis_discussion.py   # Multi-model discussion diagnosis
│   └── ...
├── plot/                   # Plotting utilities
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Environment Setup

Python 3.8+ is recommended.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `env.example` to `.env` and fill in your API keys:

```bash
cp env.example .env
```

Edit the `.env` file with your LLM API keys (DeepSeek, OpenAI, DashScope, etc.).

### 3. Run Diagnosis

#### Single Model Diagnosis (Fast)

```bash
python scripts/run_diagnosis.py deepseek --no-voting
```

#### Voting Diagnosis (High Precision)

```bash
python scripts/run_diagnosis.py deepseek --voting --num-votes 3
```

#### Multi-Model Discussion Diagnosis (Highest Precision)

The discussion-based diagnosis uses multiple models to independently analyze each stage, then engage in multi-round discussions to reach consensus or vote on the final decision.

```bash
# Default: 3 models (deepseek, gpt-4.1, gpt-5), 3 rounds per stage
python scripts/run_diagnosis_discussion.py

# Custom models and rounds
python scripts/run_diagnosis_discussion.py --models deepseek gpt-4.1 gpt-5 --max-rounds 3

# Specify input/output files
python scripts/run_diagnosis_discussion.py -i data/input/mem0_mem/sample/sampled_qa_50.json -o data/output/llm_annotation_discussion
```

**Discussion Mode Parameters:**
- `--max-rounds N`: Maximum discussion rounds per stage (default: 3)
- `--models`: List of models participating in discussion (default: deepseek gpt-4.1 gpt-5)
- `-i, --input`: Input file path
- `-o, --output-dir`: Output directory path
- `-f, --output-file`: Output filename (auto-generated if not specified)

## 🛠️ Features

### Diagnosis Stages

1. **Consistency Check (Stage 0)**: Verify if the response is semantically consistent with the reference answer.
2. **Memory Extraction (Stage 1)**: Check if the initial memory extraction is sufficient and accurate.
3. **Memory Update (Stage 2)**: Verify if memory update operations (add/delete/modify) are correct.
4. **Memory Retrieval (Stage 3)**: Check if retrieved memories contain the key information needed to answer the question.
5. **Reasoning (Stage 4)**: If all previous stages pass, check if the model's reasoning logic is correct.

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
| Stage 4 | 4.2 | Reasoning error (invented details, over-specified) |
| Stage 4 | 4.3 | Format or detail error (minor deviations) |

### Diagnosis Modes

| Mode | Command | Description |
|------|---------|-------------|
| Single Model | `run_diagnosis.py --no-voting` | Fast, single model analysis |
| Voting | `run_diagnosis.py --voting --num-votes N` | Multiple runs with majority voting |
| Discussion | `run_diagnosis_discussion.py` | Multi-model collaborative discussion |

### Supported Models

- **DeepSeek** (`deepseek`): Default recommended model.
- **GPT-4.1** (`gpt-4.1`): Suitable for high-precision benchmarks.
- **GPT-5** (`gpt-5`): Latest generation model.
- **Qwen** (`qwen`): Alibaba's Tongyi Qianwen model.

## 📊 More Usage

Please refer to [COMMAND_CHEATSHEET.md](docs/COMMAND_CHEATSHEET.md) for a detailed command reference.

## 📄 License

[License Information]

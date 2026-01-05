# Command Cheatsheet

## 🚀 Common Commands

### Single Model Diagnosis (Fast)

```bash
# DeepSeek single model
python scripts/run_diagnosis.py deepseek --no-voting

# GPT-4.1 single model
python scripts/run_diagnosis.py gpt4.1 --no-voting

# GPT-5 single model
python scripts/run_diagnosis.py gpt5 --no-voting
```

**Output Directory**: `data/output/llm_annotation_single/`

---

### Voting Diagnosis (Precise)

```bash
# DeepSeek 3-round voting (default)
python scripts/run_diagnosis.py deepseek

# DeepSeek 5-round voting
python scripts/run_diagnosis.py deepseek --num-votes 5

# GPT-4.1 3-round voting
python scripts/run_diagnosis.py gpt4.1 --voting

# GPT-5 5-round voting
python scripts/run_diagnosis.py gpt5 --num-votes 5
```

**Output Directory**: `data/output/llm_annotation_voting/`

---

### Multi-Model Discussion Diagnosis (Highest Precision)

```bash
# Default: 3 models (deepseek, gpt-4.1, gpt-5), 3 rounds per stage
python scripts/run_diagnosis_discussion.py

# Custom models
python scripts/run_diagnosis_discussion.py --models deepseek gpt-4.1 gpt-5

# Custom max rounds per stage
python scripts/run_diagnosis_discussion.py --max-rounds 5

# Specify input/output
python scripts/run_diagnosis_discussion.py -i data/input/test.json -o results/
```

**Output Directory**: `data/output/llm_annotation_discussion/`

---

## 📊 Quick Decision Guide

| Your Need | Recommended Command |
|-----------|---------------------|
| 🏃 Quick Test | `python scripts/run_diagnosis.py deepseek --no-voting` |
| 💰 Cost Saving | `python scripts/run_diagnosis.py deepseek --no-voting` |
| 🎯 High Accuracy | `python scripts/run_diagnosis.py deepseek --num-votes 5` |
| 🏆 Highest Accuracy | `python scripts/run_diagnosis_discussion.py` |
| ⚡ Batch Processing | `python scripts/run_diagnosis.py deepseek --no-voting` |
| 🔬 Research Experiments | `python scripts/run_diagnosis.py gpt4.1 --num-votes 3` |
| 🤝 Consensus-based | `python scripts/run_diagnosis_discussion.py --max-rounds 3` |
| 📈 Comparative Analysis | Run multiple configurations and compare results |

---

## 🔧 Parameter Reference

### Basic Parameters (run_diagnosis.py)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `deepseek` / `gpt4.1` / `gpt5` | Select model | `python scripts/run_diagnosis.py gpt4.1` |
| `--no-voting` | Single model mode | `python scripts/run_diagnosis.py --no-voting` |
| `--voting` | Voting mode (default) | `python scripts/run_diagnosis.py --voting` |
| `--num-votes N` | Number of voting rounds | `python scripts/run_diagnosis.py --num-votes 5` |

### Discussion Parameters (run_diagnosis_discussion.py)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--models` | Models for discussion | `--models deepseek gpt-4.1 gpt-5` |
| `--max-rounds N` | Max discussion rounds per stage | `--max-rounds 3` |

### File Parameters

| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--input` | `-i` | Specify input file | `-i data/test.json` |
| `--output-dir` | `-o` | Specify output directory | `-o results/` |
| `--output-file` | `-f` | Specify output filename | `-f output.json` |

### Other Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--help` / `-h` | Show help | `python scripts/run_diagnosis.py --help` |

---

## 📈 Statistics & Plotting

### 1. Human Annotation Statistics

```bash
python scripts/analyze_human_data.py
```
- Output: `data/output/evalresult/human_annotation_stats_[TIMESTAMP].txt`

### 2. Model Annotation Statistics (Voting Results)

```bash
python scripts/analyze_llm_results.py
```
- Input: `data/output/llm_annotation_voting/`
- Output: `data/output/evalresult/llm_annotation_voting_stats_[TIMESTAMP].txt`

### 3. Human vs Model Comparison Analysis

```bash
python scripts/compare_results.py \
    -H data/input/human_annotation \
    -L data/output/llm_annotation_voting/20251205
```
- Output:
    - `data/output/evalresult/model_phase_[TIMESTAMP].txt`
    - `data/output/evalresult/model_label_exact_[TIMESTAMP].txt`
    - `data/output/evalresult/human_vs_voting_final_phase_confusion_[TIMESTAMP].txt`

### 4. Plotting Commands

All plotting scripts generate images with timestamps to prevent overwriting.

```bash
# Plot voting results statistics
python plot/plot_voting_stats.py

# Plot human annotation statistics
python plot/plot_human_stats.py

# Plot model consistency
python plot/plot_consistency.py

# Plot confusion matrix
python plot/plot_confusion_matrix.py
```
- Output Directory: `data/output/plot_result/`

---

## 📁 Output File Naming Convention

All output files include a timestamp `_YYYYMMDD_HHMMSS`, for example:

### Single Model Mode
```
data/output/llm_annotation_single/
  └── [input_name]_single_deepseek_20251216_103000.json
```

### Voting Mode
```
data/output/llm_annotation_voting/
  └── [input_name]_voting_3rounds_deepseek_20251216_103000.json
```

### Discussion Mode
```
data/output/llm_annotation_discussion/
  └── [input_name]_discussion_3rounds_deepseek_gpt41_gpt5_20251216_103000.json
```

---

## ⚡ Performance Reference

| Configuration | Processing Speed | API Calls | Recommended Scenario |
|---------------|------------------|-----------|----------------------|
| Single Model | 1x | 1x | Daily use |
| 3-round Voting | 0.33x | 3x | Standard evaluation |
| 5-round Voting | 0.20x | 5x | High precision needs |
| Discussion (3 models, 3 rounds) | ~0.11x | ~9x per stage | Highest precision, research |

---

## 🆘 FAQ

### Q: How to interrupt a running task?
```bash
Ctrl + C  # The system will automatically save current progress
```

### Q: How to resume an interrupted task?
```bash
# Simply run the same command again
python scripts/run_diagnosis.py deepseek --no-voting
```

### Q: How to check current progress?
```bash
# Check the number of entries in the output file (replace with actual filename)
python -c "import json; print(len(json.load(open('data/output/llm_annotation_single/your_file.json'))))"
```

### Q: What happens when models disagree in discussion mode?
The system uses majority voting after max rounds. If all models disagree (1:1:1), GPT-5's result is used as the tiebreaker.

---

## 📝 Complete Examples

```bash
# Example 1: Quick single model diagnosis
$ python scripts/run_diagnosis.py deepseek --no-voting
🚀 Memory Diagnosis System Started
🤖 Using model: deepseek
📊 Diagnosis mode: Single model diagnosis
...

# Example 2: 5-round voting diagnosis
$ python scripts/run_diagnosis.py gpt4.1 --num-votes 5
🚀 Memory Diagnosis System Started
🤖 Using model: gpt-4.1
📊 Diagnosis mode: Voting mechanism (5 rounds)
...

# Example 3: Multi-model discussion diagnosis
$ python scripts/run_diagnosis_discussion.py --models deepseek gpt-4.1 gpt-5 --max-rounds 3
🚀 Memory Diagnosis System - Multi-Model Discussion Version Started
🤖 Participating models: deepseek, gpt-4.1, gpt-5
🔄 Max rounds per stage: 3
...
```

**Tip**: Bookmark this file for quick reference!

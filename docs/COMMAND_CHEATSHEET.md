# 命令速查表

## 🚀 常用命令

### 单模型诊断（快速）

```bash
# DeepSeek单模型
python scripts/run_diagnosis.py deepseek --no-voting

# GPT-4.1单模型
python scripts/run_diagnosis.py gpt4.1 --no-voting

# GPT-5单模型
python scripts/run_diagnosis.py gpt5 --no-voting
```

**输出目录**: `data/output/llm_annotation_single/`

---

### 投票诊断（精确）

```bash
# DeepSeek 3轮投票（默认）
python scripts/run_diagnosis.py deepseek

# DeepSeek 5轮投票
python scripts/run_diagnosis.py deepseek --num-votes 5

# GPT-4.1 3轮投票
python scripts/run_diagnosis.py gpt4.1 --voting

# GPT-5 5轮投票
python scripts/run_diagnosis.py gpt5 --num-votes 5
```

**输出目录**: `data/output/llm_annotation_voting/`

---

## 📊 快速决策

| 你的需求 | 推荐命令 |
|----------|----------|
| 🏃 快速测试 | `python scripts/run_diagnosis.py deepseek --no-voting` |
| 💰 节省成本 | `python scripts/run_diagnosis.py deepseek --no-voting` |
| 🎯 高准确性 | `python scripts/run_diagnosis.py deepseek --num-votes 5` |
| ⚡ 大批量处理 | `python scripts/run_diagnosis.py deepseek --no-voting` |
| 🔬 研究实验 | `python scripts/run_diagnosis.py gpt4.1 --num-votes 3` |
| 📈 对比分析 | 运行多个配置，对比结果 |

---

## 🔧 参数说明

### 基本参数

| 参数 | 作用 | 示例 |
|------|------|------|
| `deepseek` / `gpt4.1` / `gpt5` | 选择模型 | `python scripts/run_diagnosis.py gpt4.1` |
| `--no-voting` | 单模型模式 | `python scripts/run_diagnosis.py --no-voting` |
| `--voting` | 投票模式（默认） | `python scripts/run_diagnosis.py --voting` |
| `--num-votes N` | 投票轮数 | `python scripts/run_diagnosis.py --num-votes 5` |

### 文件参数

| 参数 | 短参数 | 作用 | 示例 |
|------|--------|------|------|
| `--input` | `-i` | 指定输入文件 | `python scripts/run_diagnosis.py -i data/test.json` |
| `--output-dir` | `-o` | 指定输出目录 | `python scripts/run_diagnosis.py -o results/` |
| `--output-file` | `-f` | 指定输出文件名 | `python scripts/run_diagnosis.py -f output.json` |

### 其他参数

| 参数 | 作用 | 示例 |
|------|------|------|
| `--help` / `-h` | 显示帮助 | `python scripts/run_diagnosis.py --help` |

---

## 📈 统计与绘图

### 1. 人工标注统计

```bash
python scripts/analyze_human_data.py
```
- 输出: `data/output/evalresult/human_annotation_stats_[TIMESTAMP].txt`

### 2. 模型标注统计 (投票结果)

```bash
python scripts/analyze_llm_results.py
```
- 输入: `data/output/llm_annotation_voting/`
- 输出: `data/output/evalresult/llm_annotation_voting_stats_[TIMESTAMP].txt`

### 3. 人工 vs 模型 对比分析

```bash
python scripts/compare_results.py \
    -H data/input/human_annotation \
    -L data/output/llm_annotation_voting/20251205
```
- 输出:
    - `data/output/evalresult/model_phase_[TIMESTAMP].txt`
    - `data/output/evalresult/model_label_exact_[TIMESTAMP].txt`
    - `data/output/evalresult/human_vs_voting_final_phase_confusion_[TIMESTAMP].txt`

### 4. 绘图命令

所有绘图脚本生成的图片均包含时间戳，防止覆盖。

```bash
# 绘制投票结果统计图
python plot/plot_voting_stats.py

# 绘制人工标注统计图
python plot/plot_human_stats.py

# 绘制模型一致性图
python plot/plot_consistency.py

# 绘制混淆矩阵
python plot/plot_confusion_matrix.py
```
- 输出目录: `data/output/plot_result/`

---

## 📁 输出文件命名规则

所有输出文件均包含时间戳 `_YYYYMMDD_HHMMSS`，例如：

### 单模型模式
```
data/output/llm_annotation_single/
  └── [input_name]_single_deepseek_20251216_103000.json
```

### 投票模式
```
data/output/llm_annotation_voting/
  └── [input_name]_voting_3rounds_deepseek_20251216_103000.json
```

---

## ⚡ 性能参考

| 配置 | 处理速度 | API调用 | 推荐场景 |
|------|----------|---------|----------|
| 单模型 | 1x | 1x | 日常使用 |
| 3轮投票 | 0.33x | 3x | 标准评估 |
| 5轮投票 | 0.20x | 5x | 高精度需求 |

---

## 🆘 常见问题

### Q: 如何中断正在运行的任务？
```bash
Ctrl + C  # 系统会自动保存当前进度
```

### Q: 如何继续被中断的任务？
```bash
# 再次运行相同的命令即可
python scripts/run_diagnosis.py deepseek --no-voting
```

### Q: 如何查看当前进度？
```bash
# 查看输出文件中的条目数（需替换实际文件名）
python -c "import json; print(len(json.load(open('data/output/llm_annotation_single/your_file.json'))))"
```

---

## 📝 完整示例

```bash
# 示例1: 快速单模型诊断
$ python scripts/run_diagnosis.py deepseek --no-voting
🚀 记忆诊断系统启动
🤖 使用模型: deepseek
📊 诊断模式: 单模型诊断
...

# 示例2: 5轮投票诊断
$ python scripts/run_diagnosis.py gpt4.1 --num-votes 5
🚀 记忆诊断系统启动
🤖 使用模型: gpt-4.1
📊 诊断模式: 投票机制 (5轮)
...
```

**提示**: 将此文件保存为书签，方便随时查阅！

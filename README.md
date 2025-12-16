# MemEval - Agentic Memory Evaluation System

MemEval 是一个用于评估智能体记忆系统（Agentic Memory）的诊断框架。它采用分阶段诊断方法，从一致性、提取、更新、检索到推理，全面评估记忆系统的表现。

## 📁 项目结构

```
MemEval/
├── data/                   # 数据文件
│   ├── input/              # 输入数据 (mem0_mem, human_annotation, etc.)
│   └── output/             # 结果输出 (llm_annotation_voting, etc.)
├── docs/                   # 文档
├── scripts/                # Python 脚本
│   ├── run_diagnosis.py    # 核心诊断程序
│   └── ...
├── plot/                   # 绘图工具
├── requirements.txt        # 项目依赖
└── README.md               # 本文件
```

## 🚀 快速开始

### 1. 环境准备

推荐使用 Python 3.8+。

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `env.example` 为 `.env` 并填写 API Key：

```bash
cp env.example .env
```

编辑 `.env` 文件，填入你的 LLM API Key (DeepSeek, OpenAI, DashScope 等)。

### 3. 运行诊断

#### 单模型诊断 (快速)

```bash
python scripts/run_diagnosis.py deepseek --no-voting
```

#### 投票诊断 (高精度)

```bash
python scripts/run_diagnosis.py deepseek --voting --num-votes 3
```

## 🛠️ 功能模块

### 诊断阶段
1. **一致性检查 (Stage 0)**: 检查回答是否与参考答案一致。
2. **记忆提取 (Stage 1)**: 检查初始记忆提取是否充分、准确。
3. **记忆更新 (Stage 2)**: 检查记忆更新操作（增删改）是否正确。
4. **记忆检索 (Stage 3)**: 检查检索到的记忆是否包含回答问题所需的关键信息。
5. **推理 (Stage 4)**: 如果上述阶段都通过，检查模型推理逻辑是否正确。

### 支持的模型
- **DeepSeek** (`deepseek`): 默认推荐模型。
- **GPT-4** (`gpt4.1`): 适用于高精度基准。
- **GPT-5** (`gpt5`): 实验性支持。
- **Qwen** (`qwen`): 通义千问模型。

## 📊 更多用法

请参考 [COMMAND_CHEATSHEET.md](docs/COMMAND_CHEATSHEET.md) 获取详细的命令速查表。

## 📄 许可证

[License Information]


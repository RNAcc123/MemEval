# LongMemEval Mem0 Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that runs `mem0` over `longmemeval_s_cleaned.json`, records memory extraction/update/search traces, answers each question from retrieved memories, and saves MemEval-compatible trace files.

**Architecture:** Treat each LongMemEval sample as an isolated single-user memory environment. Add the sample's historical sessions into mem0 in timestamp order, record every `add` result with session metadata, search memories for the sample question, generate `qa_response`, and persist a trace after each sample for resumability.

**Tech Stack:** Python 3, `mem0.MemoryClient`, OpenAI chat completions, existing code under `/share/project/chenchen/code/mem0/evaluation/src/memzero`, JSON trace files consumed by MemEval diagnosis scripts.

---

## Inputs And Outputs

**Input dataset**

- `/share/project/chenchen/data/longmemeval-cleaned/longmemeval_s_cleaned.json`
- 500 samples.
- One question per sample.
- Each sample has `haystack_dates`, `haystack_session_ids`, and `haystack_sessions` aligned by index.

**Reference trace format**

- `/share/project/chenchen/code/MemEval/data/input/mem0_mem/gpt4omini/mem0_dataset_part1.json`
- Existing MemEval trace shape is a dict keyed by part-local conversation index, with a list of QA trace objects.

**New output directory**

- Create: `/share/project/chenchen/code/MemEval/data/input/mem0_mem/longmemeval_s/`

**Output file pattern**

- `mem0_longmemeval_s_part1.json`
- `mem0_longmemeval_s_part2.json`
- `mem0_longmemeval_s_part3.json`
- `mem0_longmemeval_s_part4.json`
- `mem0_longmemeval_s_part5.json`

Use 100 samples per part unless overridden by CLI args.

## Trace Schema

Each output file should be a JSON object:

```json
{
  "0": [
    {
      "qa_question": "What degree did I graduate with?",
      "qa_answer": "Business Administration",
      "qa_response": "Business Administration",
      "qa_category": "single-session-user",
      "question_id": "e47becba",
      "question_date": "2023/05/30 (Tue) 23:40",
      "answer_session_ids": ["answer_280352e9"],
      "person1": {
        "name": "longmemeval_s_0",
        "memories": [
          {
            "session_id": "answer_280352e9",
            "time_stamp": "2023/05/30 (Tue) 17:27",
            "has_answer": true,
            "evidence_sentence": "session_id=answer_280352e9",
            "initial_results": [
              {
                "id": "memory-id",
                "event": "ADD",
                "memory": "The person graduated with a degree in Business Administration."
              }
            ],
            "update_chain": [
              {
                "id": "memory-id",
                "event": "ADD",
                "memory": "The person graduated with a degree in Business Administration."
              }
            ]
          }
        ]
      },
      "person2": {
        "name": "",
        "memories": []
      },
      "speaker_1_memories": [
        {
          "memory": "The person graduated with a degree in Business Administration.",
          "timestamp": "2023/05/30 (Tue) 17:27",
          "score": 0.81
        }
      ],
      "speaker_2_memories": []
    }
  ]
}
```

Notes:

- `qa_category` should use LongMemEval's `question_type` string.
- `person2` and `speaker_2_memories` stay empty to preserve compatibility with MemEval scripts that expect those keys.
- `has_answer` is true when `session_id in answer_session_ids` or any message in that session has `has_answer: true`.
- `initial_results` should contain ADD events returned by mem0 for that session.
- `update_chain` should contain all ADD/UPDATE/DELETE events returned by mem0 for that session. If mem0 returns only the final result list, save the raw returned events there.

## Files

- Create: `/share/project/chenchen/code/MemEval/scripts/run_longmemeval_mem0_trace.py`
- Create: `/share/project/chenchen/code/MemEval/data/input/mem0_mem/longmemeval_s/`
- Read: `/share/project/chenchen/data/longmemeval-cleaned/longmemeval_s_cleaned.json`
- Read: `/share/project/chenchen/code/mem0/evaluation/src/memzero/add.py`
- Read: `/share/project/chenchen/code/mem0/evaluation/src/memzero/search.py`
- Optional modify: `/share/project/chenchen/code/mem0/evaluation/src/memzero/add.py`
- Optional modify: `/share/project/chenchen/code/mem0/evaluation/src/memzero/search.py`

Prefer a new script over changing the existing LoCoMo-specific runner. The existing `memzero` classes assume two speakers; LongMemEval is single-user.

---

## Task 1: Add Trace Script Skeleton

**Files:**

- Create: `/share/project/chenchen/code/MemEval/scripts/run_longmemeval_mem0_trace.py`

- [ ] **Step 1: Create CLI and dataset loading code**

Add a script with these args:

```text
--dataset /share/project/chenchen/data/longmemeval-cleaned/longmemeval_s_cleaned.json
--output-dir /share/project/chenchen/code/MemEval/data/input/mem0_mem/longmemeval_s
--start 0
--end 500
--part-size 100
--top-k 30
--model env:MODEL
--resume
--dry-run
```

The loader must assert:

```python
len(item["haystack_dates"]) == len(item["haystack_session_ids"]) == len(item["haystack_sessions"])
```

- [ ] **Step 2: Run dry loader check**

Run:

```bash
cd /share/project/chenchen/code/MemEval
python3 scripts/run_longmemeval_mem0_trace.py --dry-run --start 0 --end 3
```

Expected:

```text
Loaded 500 samples
Selected samples: 0..2
Dry run complete
```

---

## Task 2: Normalize LongMemEval Sessions For Mem0 Add

**Files:**

- Modify: `/share/project/chenchen/code/MemEval/scripts/run_longmemeval_mem0_trace.py`

- [ ] **Step 1: Implement message normalization**

For each session, convert LongMemEval messages directly:

```python
def normalize_session_messages(session):
    messages = []
    for message in session:
        role = message.get("role")
        content = message.get("content", "")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported role: {role}")
        if content.strip():
            messages.append({"role": role, "content": content})
    return messages
```

- [ ] **Step 2: Implement timestamp/session metadata**

Every mem0 add call must include:

```python
metadata = {
    "timestamp": haystack_date,
    "session_id": haystack_session_id,
    "question_id": question_id,
    "question_type": question_type,
    "source": "longmemeval_s",
}
```

- [ ] **Step 3: Verify with dry run**

Run:

```bash
cd /share/project/chenchen/code/MemEval
python3 scripts/run_longmemeval_mem0_trace.py --dry-run --start 0 --end 1
```

Expected output should include the first sample's user id and session count:

```text
user_id=longmemeval_s_0
sessions=53
```

---

## Task 3: Add Memories And Capture Add Results

**Files:**

- Modify: `/share/project/chenchen/code/MemEval/scripts/run_longmemeval_mem0_trace.py`

- [ ] **Step 1: Instantiate mem0 client**

Use the same environment variables as `memzero/add.py`:

```python
MemoryClient(
    api_key=os.getenv("MEM0_API_KEY"),
    org_id=os.getenv("MEM0_ORGANIZATION_ID"),
    project_id=os.getenv("MEM0_PROJECT_ID"),
)
```

- [ ] **Step 2: Delete user memories before each sample**

Before adding a sample:

```python
client.delete_all(user_id=user_id)
```

This prevents memory leakage across samples.

- [ ] **Step 3: Call mem0 add per session**

Call:

```python
result = client.add(
    normalized_messages,
    user_id=user_id,
    version="v2",
    metadata=metadata,
)
```

Retry up to 3 times with a 2-second delay before failing the sample.

- [ ] **Step 4: Convert add result into trace fields**

Use this behavior:

```python
events = result if isinstance(result, list) else result.get("results", [])
initial_results = [event for event in events if event.get("event") == "ADD"]
update_chain = events
```

If mem0 returns an empty result, save:

```json
{
  "error": "No memory in that message"
}
```

- [ ] **Step 5: Persist after every sample**

After one sample finishes add/search/answer, write the current part file using:

```python
json.dump(results, f, ensure_ascii=False, indent=2)
```

---

## Task 4: Search Memories For Each Question

**Files:**

- Modify: `/share/project/chenchen/code/MemEval/scripts/run_longmemeval_mem0_trace.py`

- [ ] **Step 1: Implement search helper**

Call:

```python
memories = client.search(question, user_id=user_id, top_k=top_k)
```

Normalize results to:

```python
{
    "memory": memory["memory"],
    "timestamp": memory.get("metadata", {}).get("timestamp"),
    "session_id": memory.get("metadata", {}).get("session_id"),
    "score": round(memory.get("score", 0), 2),
}
```

- [ ] **Step 2: Store retrieval trace**

Save normalized results under:

```json
"speaker_1_memories": [...]
```

Set:

```json
"speaker_2_memories": []
```

- [ ] **Step 3: Add retrieval hit metadata**

Add optional diagnostic fields:

```json
"retrieved_answer_session_ids": ["answer_280352e9"],
"retrieval_hit": true
```

`retrieval_hit` is true if any retrieved memory has `session_id` in `answer_session_ids`.

---

## Task 5: Generate QA Response From Retrieved Memories

**Files:**

- Modify: `/share/project/chenchen/code/MemEval/scripts/run_longmemeval_mem0_trace.py`

- [ ] **Step 1: Build single-user answer prompt**

Use retrieved memories only:

```text
You are answering a question using retrieved memories from a user's past conversations.

Rules:
1. Use only the provided memories.
2. If memories conflict, prefer the more recent timestamp.
3. If the answer requires date arithmetic, compute it from timestamps.
4. Answer concisely.

Memories:
{memories_json}

Question:
{question}

Answer:
```

- [ ] **Step 2: Call OpenAI**

Use:

```python
OpenAI().chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[{"role": "system", "content": prompt}],
    temperature=0.0,
)
```

- [ ] **Step 3: Store answer**

Save the model output under:

```json
"qa_response": "..."
```

---

## Task 6: Partitioning And Resume

**Files:**

- Modify: `/share/project/chenchen/code/MemEval/scripts/run_longmemeval_mem0_trace.py`

- [ ] **Step 1: Map global sample index to part file**

For default `part_size=100`:

```python
part_id = global_index // part_size + 1
part_local_key = str(global_index % part_size)
output_file = output_dir / f"mem0_longmemeval_s_part{part_id}.json"
```

- [ ] **Step 2: Implement resume**

If `--resume` is set and `part_local_key` already exists in `output_file`, skip that sample.

- [ ] **Step 3: Save errors as trace records**

If a sample fails, save:

```json
{
  "qa_question": "...",
  "qa_answer": "...",
  "qa_response": "",
  "qa_category": "...",
  "question_id": "...",
  "error": "exception message"
}
```

Do not stop the entire run unless `--fail-fast` is passed.

---

## Task 7: Validation Commands

**Files:**

- Create: no new files.
- Modify: no source files unless validation exposes a bug.

- [ ] **Step 1: Validate one sample**

Run:

```bash
cd /share/project/chenchen/code/MemEval
python3 scripts/run_longmemeval_mem0_trace.py --start 0 --end 1 --part-size 100 --top-k 30
```

Expected:

```text
Processing sample 0
Saved mem0_longmemeval_s_part1.json
```

- [ ] **Step 2: Inspect output schema**

Run:

```bash
python3 - <<'PY'
import json
p = "/share/project/chenchen/code/MemEval/data/input/mem0_mem/longmemeval_s/mem0_longmemeval_s_part1.json"
data = json.load(open(p))
item = data["0"][0]
required = [
    "qa_question", "qa_answer", "qa_response", "qa_category",
    "person1", "person2", "speaker_1_memories", "speaker_2_memories",
    "question_id", "answer_session_ids"
]
missing = [k for k in required if k not in item]
print("missing", missing)
print("retrieved", len(item["speaker_1_memories"]))
print("memory_records", len(item["person1"]["memories"]))
PY
```

Expected:

```text
missing []
retrieved <number greater than or equal to 0>
memory_records <number greater than 0>
```

- [ ] **Step 3: Validate with MemEval diagnosis loader**

Run:

```bash
cd /share/project/chenchen/code/MemEval
python3 scripts/run_diagnosis.py deepseek --no-voting -i data/input/mem0_mem/longmemeval_s/mem0_longmemeval_s_part1.json -t 1
```

Expected:

```text
No schema/key errors before model judging starts.
```

---

## Task 8: Full Run

**Files:**

- Output: `/share/project/chenchen/code/MemEval/data/input/mem0_mem/longmemeval_s/*.json`

- [ ] **Step 1: Run part 1**

```bash
cd /share/project/chenchen/code/MemEval
python3 scripts/run_longmemeval_mem0_trace.py --start 0 --end 100 --part-size 100 --resume
```

- [ ] **Step 2: Run remaining parts**

```bash
python3 scripts/run_longmemeval_mem0_trace.py --start 100 --end 200 --part-size 100 --resume
python3 scripts/run_longmemeval_mem0_trace.py --start 200 --end 300 --part-size 100 --resume
python3 scripts/run_longmemeval_mem0_trace.py --start 300 --end 400 --part-size 100 --resume
python3 scripts/run_longmemeval_mem0_trace.py --start 400 --end 500 --part-size 100 --resume
```

- [ ] **Step 3: Count completed questions**

```bash
python3 - <<'PY'
import glob, json
total = 0
for p in sorted(glob.glob("/share/project/chenchen/code/MemEval/data/input/mem0_mem/longmemeval_s/*.json")):
    data = json.load(open(p))
    count = sum(len(v) for v in data.values())
    print(p, count)
    total += count
print("total", total)
PY
```

Expected:

```text
total 500
```

## Operational Notes

- Full run cost is significant: about 500 samples times roughly 48 sessions per sample, or about 24,000 mem0 add calls plus 500 search calls and 500 answer-generation calls.
- Keep user IDs isolated as `longmemeval_s_{global_index}`.
- Always delete user memory before processing a sample unless resuming a fully completed trace record.
- Save after each sample. This job is long enough that losing in-memory progress is unacceptable.
- Avoid parallelism until one-part sequential execution is stable. Mem0 project rate limits and OpenAI rate limits are likely bottlenecks.
- Add `--sleep-seconds` later only if rate limits appear.

## Self-Review

- Spec coverage: The plan covers dataset loading, mem0 add, add trace capture, search trace capture, QA response generation, MemEval-compatible JSON output, partitioning, resume, and validation.
- Placeholder scan: No implementation step depends on an unspecified schema or unnamed file.
- Type consistency: Output keys match the existing MemEval trace style: `qa_question`, `qa_answer`, `qa_response`, `qa_category`, `person1`, `person2`, `speaker_1_memories`, and `speaker_2_memories`.

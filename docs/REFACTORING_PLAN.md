# MemEval Refactoring Plan

## 1. Background

MemEval is a stage-by-stage diagnosis framework for long-term memory systems. It currently supports:

- Generating MemEval-compatible traces from LoCoMo and LongMemEval-S.
- Evaluating local Mem0 and OpenClaw native memory backends.
- Diagnosing failures in consistency, extraction, update, retrieval, and reasoning stages.
- Single-model, multi-model voting, and multi-model discussion modes.
- Human/LLM annotation comparison, statistics, and plotting.

The repository already implements an end-to-end research workflow, but most functionality grew as independent scripts. Core diagnosis logic, provider calls, dataset conversion, memory-system integration, persistence, CLI handling, and experiment configuration are now mixed together. This makes correctness difficult to verify and causes new datasets or backends to duplicate existing pipelines.

This document defines an incremental refactoring plan. The intent is to preserve current experiment behavior while establishing explicit contracts, reliable error handling, reusable components, and reproducible runs.

## 2. Refactoring Goals

### 2.1 Primary goals

1. Protect experiment correctness before reorganizing files.
2. Distinguish a correct answer from infrastructure or judge failures.
3. Define and validate a versioned trace and diagnosis schema.
4. Separate domain logic from provider SDKs, filesystem operations, and CLI code.
5. Reuse the same pipeline across datasets and memory backends.
6. Make every run reproducible, resumable, and auditable.
7. Make all tests discoverable and runnable with one command.
8. Keep existing commands working during migration.

### 2.2 Non-goals

- Replacing Mem0 or OpenClaw implementations.
- Changing the diagnosis taxonomy without a separate research decision.
- Rewriting all code in one pull request.
- Adding a web interface.
- Migrating historical data before the new schema and migration tooling exist.

## 3. Current Problems

### 3.1 Correctness risks

#### Error results are ambiguous

`label = null` currently has two possible meanings:

- Stage 0 passed and the answer is correct.
- An API call or response parse failed and no label was produced.

Voting includes `null` labels as normal votes, so provider failures can influence the final diagnosis. Refactoring must introduce an explicit status and exclude failed votes from aggregation.

#### Retry handling is incomplete

The shared retry loop catches `requests` exceptions, while OpenAI-compatible SDKs usually raise their own exception types. Provider adapters must normalize SDK exceptions into retryable and permanent failure categories.

#### Judge output is not strictly validated

Stage code reads fields with `.get()` and supplies fallback values. A malformed response can therefore become a diagnosis rather than an invalid judge response. Every stage must validate required fields, allowed labels, and relationships between `stage_passed` and `label`.

### 3.2 Structural problems

- `scripts/run_diagnosis.py` combines schemas, prompts, API clients, orchestration, voting, persistence, concurrency, logging, and CLI parsing.
- Single, voting, and discussion modes duplicate prompt and stage behavior.
- LoCoMo and LongMemEval Mem0 runners duplicate environment, client, retry, normalization, locking, and output code.
- `scripts/compare_results.py` and `eval/matching.py` contain substantially duplicated logic.
- Machine-specific paths under `/share/project/chenchen/...` appear in Python and shell defaults.
- Tests are split between `tests/` and `scripts/`, and default `unittest` discovery does not find nested suites.
- Source data, generated traces, runtime state, reports, and figures share the `data/` tree without a clear lifecycle.
- The local virtual environment is named `memeval/`, which conflicts with the intended Python package name.

### 3.3 Compatibility problems

- The README claims Python 3.8+ support.
- The code uses `zip(..., strict=True)`, which requires Python 3.10+.
- Some plotting f-strings require newer parsing behavior and do not compile under the current Python 3.9 environment.
- `fcntl` makes the persistence implementation Unix-specific.

The refactored project will explicitly support Python 3.11 and newer unless broader compatibility is required by deployment infrastructure.

## 4. Design Principles

### 4.1 Preserve behavior through adapters

Existing scripts remain available as thin compatibility wrappers until replacement commands are verified. A migration should first redirect a script to shared code, then remove duplicated internals in a later change.

### 4.2 Make invalid states explicit

Do not encode errors as missing labels or empty lists. Use explicit statuses and typed records.

### 4.3 Separate domain and infrastructure

Diagnosis stages must not import provider SDKs or write files. Memory backends must not decide diagnosis labels. Dataset adapters must not own concurrency or run storage.

### 4.4 Prefer deterministic artifacts

Every run must record enough metadata to reproduce its inputs and configuration. Generated files should have stable schemas and deterministic identifiers.

### 4.5 Refactor incrementally

Each phase must leave the repository runnable. Avoid combining behavior changes, directory moves, and large formatting changes in the same commit.

## 5. Target Architecture

```text
MemEval/
├── pyproject.toml
├── README.md
├── env.example
├── src/
│   └── memeval/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── schema/
│       │   ├── trace.py
│       │   ├── diagnosis.py
│       │   ├── run.py
│       │   └── validation.py
│       ├── diagnosis/
│       │   ├── pipeline.py
│       │   ├── stages.py
│       │   ├── prompts.py
│       │   ├── voting.py
│       │   └── discussion.py
│       ├── providers/
│       │   ├── base.py
│       │   ├── errors.py
│       │   ├── openai.py
│       │   ├── deepseek.py
│       │   ├── dashscope.py
│       │   └── registry.py
│       ├── memory/
│       │   ├── base.py
│       │   ├── mem0.py
│       │   └── openclaw.py
│       ├── datasets/
│       │   ├── base.py
│       │   ├── locomo.py
│       │   └── longmemeval.py
│       ├── runners/
│       │   ├── trace.py
│       │   └── diagnose.py
│       ├── storage/
│       │   ├── atomic.py
│       │   ├── jsonl.py
│       │   └── run_store.py
│       └── analysis/
│           ├── matching.py
│           ├── metrics.py
│           └── plotting.py
├── configs/
│   ├── locomo_mem0.yaml
│   ├── locomo_openclaw.yaml
│   └── longmemeval_mem0.yaml
├── scripts/
│   └── compatibility wrappers only
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   └── fixtures/
├── data/
│   ├── samples/
│   └── README.md
├── runs/
│   └── ignored by Git
└── docs/
    ├── architecture.md
    ├── trace-schema.md
    ├── experiments.md
    └── migration.md
```

## 6. Core Interfaces

### 6.1 Dataset adapter

Dataset-specific parsing belongs behind one interface:

```python
class DatasetAdapter(Protocol):
    def load(self, path: Path) -> list[EvaluationSample]: ...
    def validate(self, raw_data: object) -> None: ...
```

`EvaluationSample` must expose normalized conversations, speakers, sessions, questions, reference answers, categories, timestamps, and evidence identifiers. LoCoMo and LongMemEval field names must not leak beyond their adapters.

### 6.2 Memory backend

Memory-system behavior belongs behind another interface:

```python
class MemoryBackend(Protocol):
    def reset(self, subject: MemorySubject) -> None: ...
    def add_session(self, subject: MemorySubject, session: Session) -> list[MemoryEvent]: ...
    def search(self, subject: MemorySubject, query: str, top_k: int) -> list[RetrievedMemory]: ...
```

Mem0 and OpenClaw implementations may use different storage mechanisms, but both must return the same normalized event and retrieval records.

### 6.3 Judge provider

Model calls must use a provider-neutral interface:

```python
class JudgeProvider(Protocol):
    @property
    def name(self) -> str: ...

    def judge(self, request: JudgeRequest) -> JudgeResponse: ...
```

Provider implementations own SDK construction, credentials, timeouts, retries, usage extraction, and exception translation. They do not own prompt selection or diagnosis control flow.

### 6.4 Run store

Persistence must expose an append/resume contract:

```python
class RunStore(Protocol):
    def completed_ids(self) -> set[str]: ...
    def append_result(self, result: DiagnosisRecord) -> None: ...
    def append_error(self, error: RunError) -> None: ...
    def finalize(self, summary: RunSummary) -> None: ...
```

The implementation should use JSON Lines for incremental results. Small append-only writes avoid rewriting an entire result array after every question.

## 7. Data Contracts

### 7.1 Trace schema

Every trace file must include:

```json
{
  "schema_version": "1.0",
  "dataset": "locomo",
  "memory_backend": "mem0",
  "records": []
}
```

Each trace record must have a stable `record_id` and typed fields for:

- Question, reference answer, generated response, and category.
- Subjects or speakers.
- Per-session extraction and update events.
- Retrieved memories, scores, ranks, timestamps, and source IDs.
- Evidence IDs and retrieval-hit metadata where available.
- Trace-generation errors, represented separately from successful records.

### 7.2 Diagnosis schema

A diagnosis result must distinguish outcome and execution status:

```json
{
  "record_id": "...",
  "status": "completed",
  "answer_correct": false,
  "stage": "memory_retrieval",
  "label": "3.1",
  "reason": "...",
  "judge": "gpt-5",
  "prompt_version": "stage3-v1"
}
```

Allowed statuses:

- `completed`: a valid diagnosis was produced.
- `error`: the sample could not be diagnosed.
- `skipped`: the sample was intentionally excluded with a reason.

Required invariants:

- `answer_correct=true` requires `stage=consistency_check` and `label=null`.
- `answer_correct=false` requires a valid stage-specific label.
- `status=error` must not contain a diagnosis label.
- A vote with `status=error` must not participate in aggregation.

### 7.3 Run manifest

Each run must write `manifest.json` before processing starts. It should contain:

- Run ID and start time.
- Git commit and dirty-worktree flag.
- Dataset path, size, and content hash.
- Trace schema version.
- Model/provider configuration without secrets.
- Diagnosis mode, vote count, discussion rounds, and prompt versions.
- Concurrency, timeout, retry, and top-k settings.
- Python and dependency versions.

## 8. Error and Voting Semantics

### 8.1 Error categories

```python
class MemEvalError(Exception): ...
class ConfigurationError(MemEvalError): ...
class SchemaValidationError(MemEvalError): ...
class RetryableProviderError(MemEvalError): ...
class PermanentProviderError(MemEvalError): ...
class InvalidJudgeResponse(MemEvalError): ...
class MemoryBackendError(MemEvalError): ...
```

Only `RetryableProviderError` is retried automatically. Configuration and schema errors fail fast. Per-record provider or backend errors are persisted to `errors.jsonl` and do not become diagnosis labels.

### 8.2 Validated stage responses

Each stage has its own allowed output:

- Stage 0: `is_consistent: bool`, `reason: str`.
- Stages 1-3: `is_sufficient: bool`, optional stage-specific label, `reason: str`.
- Stage 4: required label in `4.1`, `4.2`, or `4.3`, plus `reason`.

Unknown labels, missing booleans, empty reasons, or contradictory fields must raise `InvalidJudgeResponse`.

### 8.3 Voting rules

1. Only completed judgments are valid votes.
2. The output records requested, completed, failed, and excluded vote counts.
3. A configurable minimum number of valid votes is required.
4. A strict majority selects a label.
5. Ties use an explicit configured policy, not list order.
6. The final reason is selected from a vote carrying the winning label.
7. Infrastructure failures are reported separately from label disagreement.

Suggested default:

```text
requested votes: 3
minimum valid votes: 2
tie policy: primary judge
```

## 9. CLI and Configuration

Provide one installed command with subcommands:

```bash
memeval trace --config configs/locomo_mem0.yaml
memeval diagnose --trace runs/<id>/traces.jsonl --mode voting
memeval analyze --run runs/<id>
memeval plot --run runs/<id>
memeval validate --trace path/to/trace.json
```

Configuration precedence:

1. CLI option.
2. Experiment config file.
3. Environment variable.
4. Repository-relative default.

Secrets remain in environment variables or `.env`; they must never be serialized into manifests. Machine-specific paths must not appear as Python constants.

Existing scripts continue to work during migration and print a deprecation notice only after the new CLI is stable.

## 10. Run Directory Layout

```text
runs/<run-id>/
├── manifest.json
├── traces.jsonl
├── diagnoses.jsonl
├── errors.jsonl
├── metrics.json
├── reports/
│   └── summary.md
└── figures/
```

`run-id` should be stable when resuming. A new run may use a timestamp plus a short configuration hash, for example:

```text
20260727-153000-locomo-mem0-a13f42
```

Resume behavior must be explicit:

```bash
memeval diagnose --resume runs/20260727-153000-locomo-mem0-a13f42
```

## 11. Testing Strategy

### 11.1 Unit tests

- Stage traversal and early-exit behavior.
- Stage response validation.
- Provider exception translation and retry policy.
- Voting with agreement, ties, partial failures, and insufficient valid votes.
- Dataset normalization.
- Memory-event and search-result normalization.
- Atomic/append-only persistence and resume behavior.
- Metrics and matching coverage.

### 11.2 Contract tests

Every dataset adapter must produce valid `EvaluationSample` objects. Every memory backend must satisfy the same backend test suite. Every provider must return the same normalized judge response and usage record.

### 11.3 Integration tests

- LoCoMo fixture -> fake memory backend -> trace.
- Trace fixture -> fake providers -> diagnosis.
- Diagnosis fixture -> metrics -> figures.
- Resume after an interrupted run.
- Optional live-provider tests, disabled unless credentials are present.

### 11.4 Required commands

```bash
pytest
python -m compileall src tests
ruff check src tests
```

CI should run on the minimum and latest supported Python versions. Live API tests must not run in normal pull requests.

## 12. Migration Phases

### Phase 0: Establish a reliable baseline

Tasks:

- [x] Establish a supported Python 3.11 environment at `.venv/` (the ignored legacy `memeval/` environment can be removed separately).
- [x] Add `pyproject.toml` and declare Python 3.11+.
- [x] Move all tests under `tests/` and switch to unified test discovery.
- [x] Fix current Python compatibility and compile failures.
- [x] Add characterization tests for the existing stage pipeline.
- [ ] Add fixtures containing representative LoCoMo and LongMemEval traces.
- [ ] Record current output for deterministic fake-provider inputs.

Acceptance criteria:

- `pytest` discovers every test.
- `python -m compileall` succeeds.
- Existing CLI smoke tests pass without network access.
- Baseline fixtures and expected outputs are committed.

### Phase 1: Fix correctness before moving code

Tasks:

- [x] Add explicit diagnosis and vote statuses.
- [x] Prevent error votes from being counted as `null` labels.
- [x] Introduce stage-specific output validation.
- [x] Normalize provider errors and retry only retryable failures.
- [x] Add minimum-valid-vote and explicit tie policies.
- [x] Persist per-record errors instead of silently continuing.
- [x] Report failure and coverage metrics.

Acceptance criteria:

- Provider failure cannot produce a successful diagnosis.
- Invalid JSON or invalid labels never enter result statistics.
- Voting tests cover zero, one, two, and three valid votes.
- Existing successful fixture diagnoses remain equivalent.

### Phase 2: Introduce the package and schemas

Tasks:

- [ ] Create `src/memeval/`.
- [ ] Move shared dataclasses/enums into `schema/`.
- [ ] Add `schema_version` and validation at every file boundary.
- [ ] Extract stage prompts and traversal into `diagnosis/`.
- [ ] Keep `scripts/run_diagnosis.py` as a wrapper around the package.
- [ ] Add trace and diagnosis migration helpers.

Acceptance criteria:

- Core diagnosis code performs no filesystem or SDK imports.
- Old and new CLIs produce equivalent results for fixtures.
- Invalid traces fail with field-level error messages.

### Phase 3: Extract provider adapters

Tasks:

- [x] Implement the provider protocol and registry.
- [x] Move OpenAI, DeepSeek, DashScope, and Gemini calls into adapters.
- [x] Centralize timeouts, retries, usage accounting, and response parsing.
- [x] Allow model identifiers and base URLs through configuration.
- [x] Remove provider-specific branches from diagnosis stages.

Acceptance criteria:

- Diagnosis tests use fake providers without monkey-patching SDK functions.
- Adding a provider requires one adapter and registry entry.
- All provider failures use the shared error taxonomy.

### Phase 4: Unify dataset and memory runners

Tasks:

- [x] Implement LoCoMo and LongMemEval dataset adapters.
- [x] Implement Mem0 and OpenClaw memory backend adapters.
- [x] Extract shared retry and normalized memory record contracts.
- [ ] Replace `sys.path` injection with an installed or explicitly configured Mem0 dependency.
- [x] Replace hard-coded Python defaults with environment-configurable, repository-relative paths.
- [ ] Keep existing runner scripts as compatibility wrappers.

Acceptance criteria:

- Dataset and memory backend are independently selectable.
- Shared trace runner can execute LoCoMo+Mem0, LoCoMo+OpenClaw, and LongMemEval+Mem0.
- No machine-specific absolute paths remain in Python defaults.

### Phase 5: Consolidate persistence and analysis

Tasks:

- [x] Add run manifests and stable run IDs.
- [x] Add a JSONL append-only store for incremental traces, diagnoses, and errors.
- [x] Provide an explicit completed-id/resume contract.
- [x] Merge `eval/matching.py` and `scripts/compare_results.py`.
- [x] Generate structured `metrics.json` alongside legacy formatted reports.
- [x] Report missing, duplicate, invalid, and matched record counts through the metrics API.

Acceptance criteria:

- Interrupted runs resume without regenerating completed records.
- No result array is rewritten after every completed question.
- Analysis output is deterministic for the same inputs.
- Coverage is visible in every comparison report.

### Phase 6: Repository and documentation cleanup

Tasks:

- [x] Add `data/README.md` describing tracked and generated artifacts.
- [x] Document that only small fixtures and intentionally published results belong in Git.
- [ ] Move large reproducible artifacts to releases, object storage, or DVC.
- [x] Update README installation and command documentation.
- [x] Add architecture, provenance, and migration documentation.
- [x] Document licenses and data-use responsibilities.
- [ ] Remove deprecated wrappers after a published transition period.

Acceptance criteria:

- A clean checkout can install, validate fixtures, and run offline tests from documented commands.
- Repository data provenance and regeneration steps are documented.
- No obsolete duplicate modules or machine-specific wrapper scripts remain.

## 13. Suggested Pull Request Sequence

Keep pull requests small enough to review independently:

1. `build: define Python version and unified test discovery`
2. `fix: separate diagnosis failures from correct-answer labels`
3. `fix: validate judge responses and voting inputs`
4. `refactor: add versioned trace and diagnosis schemas`
5. `refactor: extract diagnosis pipeline and prompts`
6. `refactor: extract judge provider adapters`
7. `refactor: add dataset adapter contracts`
8. `refactor: add memory backend contracts`
9. `refactor: unify trace runners`
10. `refactor: add run store and explicit resume`
11. `refactor: consolidate matching and metrics`
12. `docs: update architecture, schema, and experiment workflow`

Each pull request should include tests and avoid unrelated formatting changes.

## 14. Backward Compatibility

During migration:

- Existing JSON traces remain readable through a legacy loader.
- Existing script names and major CLI flags remain available.
- Existing outputs can be migrated with an explicit command; they are never modified in place.
- The compatibility layer emits warnings only after replacement commands are documented.
- Historical results retain their original prompt/model metadata when known; unknown metadata is recorded as `null`, not guessed.

Compatibility may be removed only after:

1. Fixture equivalence is verified.
2. Current experiment scripts use the new CLI.
3. A migration guide is published.
4. Historical result files can be validated or converted.

## 15. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Prompt movement changes judge results | Store prompt versions and characterization fixtures; avoid editing prompt text during extraction. |
| Schema migration loses historical fields | Preserve unknown fields in legacy metadata and never overwrite source files. |
| New abstractions hide backend-specific behavior | Keep backend metadata and add backend contract plus integration tests. |
| Large refactor blocks experiments | Maintain wrapper scripts and merge in phases. |
| JSONL complicates existing analysis | Provide a loader that accepts legacy JSON arrays and new JSONL during transition. |
| Provider retry changes API cost | Record attempts and usage; cap retries and total elapsed time. |
| Parallel writes corrupt results | Centralize writes in `RunStore` and test concurrent append/resume behavior. |

## 16. Definition of Done

The refactor is complete when all of the following are true:

- [x] The project has one documented minimum Python version and one installation flow.
- [x] All tests run through a single command in CI.
- [ ] Provider errors cannot be interpreted as correct answers or valid votes.
- [ ] Every trace and diagnosis artifact carries a schema version.
- [ ] Core diagnosis logic is independent of SDKs, filesystem paths, and CLI parsing.
- [ ] LoCoMo and LongMemEval use dataset adapters rather than separate pipelines.
- [ ] Mem0 and OpenClaw satisfy a shared memory backend contract.
- [ ] Single, voting, and discussion modes share stage definitions and prompt builders.
- [x] Runs have stable IDs, manifests, append-only results, and explicit resume behavior.
- [ ] Analysis reports matching coverage and consumes structured metrics.
- [x] No machine-specific absolute path is required by default.
- [ ] Duplicate matching, runner, and provider implementations are removed.
- [ ] Data provenance, licenses, and regeneration steps are documented.

## 17. Immediate Next Step

Start with Phase 0 and Phase 1 only. The first implementation milestone should produce:

1. A `pyproject.toml` declaring Python 3.11+ and development dependencies.
2. A unified `pytest` suite that discovers all current tests.
3. Passing compile checks for scripts and plotting modules.
4. Explicit diagnosis statuses and validated judge responses.
5. Voting behavior that excludes failed judgments.

Do not begin large directory moves until these correctness protections are in place.

"""MemEval command line interface.

Subcommands are thin: argument parsing lives here, everything else comes from
``memeval.*`` modules so the CLI and the legacy scripts share one implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from memeval.runners import BackendSettings, backend_manifest, build_backend


load_dotenv(os.getenv("MEMEVAL_ENV_FILE", ".env"))

app = typer.Typer(
    name="memeval",
    help="Stage-by-stage diagnosis for long-term memory systems.",
    no_args_is_help=True,
    add_completion=False,
)

trace_app = typer.Typer(help="Generate memory traces from a dataset.", no_args_is_help=True)
app.add_typer(trace_app, name="trace")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@app.command()
def version() -> None:
    """Print the installed MemEval version and artifact schema versions."""
    from memeval.schema import DIAGNOSIS_SCHEMA_VERSION
    from memeval.trace import TRACE_V2_SCHEMA_VERSION

    try:
        from importlib.metadata import version as _pkg_version

        package_version = _pkg_version("memeval")
    except Exception:
        package_version = "unknown"
    typer.echo(f"memeval {package_version}")
    typer.echo(f"trace schema {TRACE_V2_SCHEMA_VERSION}")
    typer.echo(f"diagnosis schema {DIAGNOSIS_SCHEMA_VERSION}")


@app.command()
def backends() -> None:
    """List memory backends and whether their dependencies are importable."""
    from memeval.runners import BACKEND_CHOICES

    probes = {
        "fake": None,
        "mem0": "mem0",
        "amem": "agentic_memory",
        "memoryos": "memoryos",
        "openclaw": None,
    }
    for name in BACKEND_CHOICES:
        module = probes.get(name)
        if module is None:
            status = "built-in" if name == "fake" else "external CLI (needs `openclaw` on PATH)"
        else:
            try:
                __import__(module)
                status = "available"
            except ImportError:
                status = f"not installed (pip install 'memeval[{name}]')"
        typer.echo(f"{name:10} {status}")


@app.command("validate-trace")
def validate_trace_command(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Trace file to validate."),
    schema: str = typer.Option("v2", "--schema", help="Trace schema to validate against: v1 or v2."),
) -> None:
    """Validate a trace file against the versioned trace schema."""
    from memeval.schema import validate_trace_dataset, validate_trace_v2

    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        if schema == "v2":
            validate_trace_v2(payload)
        else:
            validate_trace_dataset(payload)
    except Exception as exc:
        typer.secho(f"invalid: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"valid ({schema})", fg=typer.colors.GREEN)


@trace_app.command("run")
def trace_run(
    dataset: Path = typer.Option(..., "--dataset", exists=True, dir_okay=False, help="Dataset JSON file."),
    dataset_type: str = typer.Option(..., "--dataset-type", help="locomo or longmemeval."),
    backend: str = typer.Option("fake", "--backend", help="Memory backend to trace."),
    output_dir: Path = typer.Option(Path("runs/memory-trace"), "--output-dir"),
    start: int = typer.Option(0, "--start"),
    end: Optional[int] = typer.Option(None, "--end"),
    top_k: int = typer.Option(10, "--top-k"),
    context_limit: Optional[int] = typer.Option(None, "--context-limit"),
    model: str = typer.Option("fake-model", "--model", help="Answer-generation model."),
    generation_backend: str = typer.Option("fake", "--generation-backend", help="fake or openai."),
    api_key: str = typer.Option("", "--api-key", help="Generation API key; defaults to OPENAI_API_KEY."),
    base_url: str = typer.Option("", "--base-url"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    resume: bool = typer.Option(False, "--resume", help="Skip records already completed in output-dir."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Load and select samples, then stop."),
    # mem0
    mem0_mode: str = typer.Option("local", "--mem0-mode", help="local (configurable backbone) or cloud."),
    mem0_repo: str = typer.Option("", "--mem0-repo", envvar="MEMEVAL_MEM0_REPO"),
    mem0_store_dir: Path = typer.Option(Path("data/input/mem0_mem/store"), "--mem0-store-dir"),
    mem0_llm_model: str = typer.Option("gpt-4o-mini", "--mem0-llm-model"),
    mem0_embedding_model: str = typer.Option("text-embedding-3-small", "--mem0-embedding-model"),
    mem0_llm_provider: str = typer.Option("openai", "--mem0-llm-provider"),
    mem0_embedder_provider: str = typer.Option("openai", "--mem0-embedder-provider"),
    mem0_vector_store: str = typer.Option("qdrant", "--mem0-vector-store"),
    mem0_collection: str = typer.Option("memeval_memories", "--mem0-collection"),
    mem0_llm_api_key_env: str = typer.Option("", "--mem0-llm-api-key-env"),
    mem0_llm_base_url: str = typer.Option("", "--mem0-llm-base-url"),
    mem0_embedder_api_key_env: str = typer.Option("", "--mem0-embedder-api-key-env"),
    mem0_embedder_base_url: str = typer.Option("", "--mem0-embedder-base-url"),
    # openclaw
    openclaw_bin: str = typer.Option("openclaw", "--openclaw-bin", envvar="MEMEVAL_OPENCLAW_BIN"),
    openclaw_agent: str = typer.Option("main", "--openclaw-agent"),
    openclaw_agent_model: str = typer.Option("", "--openclaw-agent-model"),
    openclaw_profile: str = typer.Option("", "--openclaw-profile"),
    openclaw_session_prefix: str = typer.Option("", "--openclaw-session-prefix"),
    openclaw_workspace_root: Path = typer.Option(
        Path("data/input/openclaw_mem/workspaces"), "--openclaw-workspace-root"
    ),
    openclaw_timeout: float = typer.Option(300.0, "--openclaw-timeout"),
    # amem
    amem_persist_root: Path = typer.Option(Path("data/input/amem_mem/persist"), "--amem-persist-root"),
    amem_embed_model: str = typer.Option(
        "all-MiniLM-L6-v2", "--amem-embed-model", envvar="MEMEVAL_AMEM_EMBED_MODEL"
    ),
    amem_llm_backend: str = typer.Option(
        "openai", "--amem-llm-backend", envvar="MEMEVAL_AMEM_LLM_BACKEND"
    ),
    amem_llm_model: str = typer.Option(
        "gpt-4o-mini", "--amem-llm-model", envvar="MEMEVAL_AMEM_LLM_MODEL"
    ),
    amem_api_key: str = typer.Option(
        "", "--amem-api-key", help="Defaults to MEMEVAL_AMEM_API_KEY, then OPENAI_API_KEY."
    ),
    amem_base_url: str = typer.Option(
        "", "--amem-base-url", help="Defaults to MEMEVAL_AMEM_BASE_URL, then OPENAI_BASE_URL."
    ),
    # memoryos
    memoryos_storage_root: Path = typer.Option(
        Path("data/input/memoryos_mem/storage"), "--memoryos-storage-root"
    ),
    memoryos_llm_model: str = typer.Option("gpt-4o-mini", "--memoryos-llm-model"),
    memoryos_embed_model: str = typer.Option("all-MiniLM-L6-v2", "--memoryos-embed-model"),
    memoryos_assistant_id: str = typer.Option("memeval_assistant", "--memoryos-assistant-id"),
) -> None:
    """Run a dataset through a memory backend and write trace records."""
    from memeval.datasets.locomo import LoCoMoAdapter
    from memeval.datasets.longmemeval import LongMemEvalAdapter
    from memeval.generation import TracedGenerator
    from memeval.generation.fake import FakeGenerationBackend
    from memeval.runners.memory_trace import MemoryTraceRunner
    from memeval.storage.trace_store import TraceStore
    from memeval.trace import TRACE_V2_SCHEMA_VERSION

    if dataset_type not in {"locomo", "longmemeval"}:
        raise typer.BadParameter("dataset-type must be locomo or longmemeval")

    adapter = LoCoMoAdapter() if dataset_type == "locomo" else LongMemEvalAdapter()
    samples = adapter.load(dataset)
    last = len(samples) if end is None else min(end, len(samples))
    if start < 0 or last < start:
        raise typer.BadParameter(f"Invalid range {start}:{last}")
    selected = samples[start:last]

    typer.echo(f"Loaded {len(samples)} samples")
    typer.echo(f"Selected samples: {start}..{start + len(selected) - 1}")
    if dry_run:
        typer.echo("Dry run complete")
        return

    settings = BackendSettings(
        name=backend,
        top_k=top_k,
        mem0={
            "mode": mem0_mode, "repo": mem0_repo, "store_dir": mem0_store_dir,
            "llm_model": mem0_llm_model, "embedding_model": mem0_embedding_model,
            "llm_provider": mem0_llm_provider, "embedder_provider": mem0_embedder_provider,
            "vector_store": mem0_vector_store, "collection": mem0_collection,
            "llm_api_key_env": mem0_llm_api_key_env, "llm_base_url": mem0_llm_base_url,
            "embedder_api_key_env": mem0_embedder_api_key_env,
            "embedder_base_url": mem0_embedder_base_url,
        },
        openclaw={
            "bin": openclaw_bin, "agent": openclaw_agent, "agent_model": openclaw_agent_model,
            "profile": openclaw_profile, "session_prefix": openclaw_session_prefix,
            "workspace_root": openclaw_workspace_root, "timeout": openclaw_timeout,
        },
        amem={
            "persist_root": amem_persist_root, "embed_model": amem_embed_model,
            "llm_backend": amem_llm_backend, "llm_model": amem_llm_model,
            "api_key": amem_api_key or _env("MEMEVAL_AMEM_API_KEY") or _env("OPENAI_API_KEY"),
            "base_url": amem_base_url or _env("MEMEVAL_AMEM_BASE_URL") or _env("OPENAI_BASE_URL"),
        },
        memoryos={
            "storage_root": memoryos_storage_root, "llm_model": memoryos_llm_model,
            "embed_model": memoryos_embed_model, "assistant_id": memoryos_assistant_id,
            "api_key": api_key or _env("OPENAI_API_KEY"), "base_url": base_url,
        },
    )
    memory_backend = build_backend(settings)

    if generation_backend == "fake":
        generator, generation_parameters = TracedGenerator(FakeGenerationBackend()), {}
    else:
        from openai import OpenAI

        from memeval.generation.openai import OpenAIChatGenerationBackend

        key = api_key or _env("OPENAI_API_KEY")
        if not key:
            raise typer.BadParameter("openai generation requires --api-key or OPENAI_API_KEY")
        client_kwargs = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url
        generator = TracedGenerator(OpenAIChatGenerationBackend(OpenAI(**client_kwargs)))
        generation_parameters = {} if temperature is None else {"temperature": temperature}

    runner = MemoryTraceRunner(
        memory_backend, generator, top_k, context_limit, model, generation_parameters
    )
    store = TraceStore(output_dir)
    manifest = {
        "dataset": str(dataset.resolve()),
        "dataset_type": dataset_type,
        "memory_backend": backend,
        "generation_backend": generation_backend,
        "model": model,
        "top_k": top_k,
        "context_limit": context_limit,
        "temperature": temperature,
        "start": start,
        "end": end,
        "trace_schema_version": TRACE_V2_SCHEMA_VERSION,
        **backend_manifest(settings),
    }
    store.write_manifest(manifest)

    completed = store.completed_ids() if resume else set()
    added = skipped = 0
    selected_questions = sum(len(sample.questions) for sample in selected)
    for sample in selected:
        for question_index, original_question in enumerate(sample.questions):
            question = dict(original_question)
            question.setdefault("question_id", f"q{question_index + 1}")
            record_id = f"{sample.sample_id}:{question['question_id']}"
            if record_id in completed:
                skipped += 1
                continue
            try:
                store.append(runner.run_sample(sample, question))
                added += 1
            except Exception as exc:  # one bad sample must not stop the run
                store.append_error(record_id, exc)
                typer.secho(f"ERROR {record_id}: {exc}", fg=typer.colors.RED, err=True)
    store.export_legacy_json()
    summary = store.write_summary(
        selected_samples=len(selected),
        selected_questions=selected_questions,
        added=added,
        skipped=skipped,
    )
    typer.echo(f"Completed traces: {added}")
    typer.echo(f"Total completed traces: {summary['completed']}")


@app.command()
def analyze(
    input_dir: Path = typer.Option(
        Path("data/output/llm_annotation_voting"), "-i", "--input-dir", help="Diagnosis results directory."
    ),
    output_dir: Path = typer.Option(Path("data/output/evalresult"), "-o", "--output-dir"),
) -> None:
    """Summarize diagnosis results into metrics.json plus a text report."""
    from memeval.analysis import run_analyze

    os.makedirs(output_dir, exist_ok=True)
    run_analyze(str(input_dir), str(output_dir))


@app.command()
def compare(
    human_dir: Path = typer.Option(Path("data/input/human_annotation"), "-H", "--human-dir"),
    llm_dir: Path = typer.Option(Path("data/output/llm_annotation_voting"), "-L", "--llm-dir"),
    output_dir: Path = typer.Option(Path("data/output/evalresult"), "-o", "--output-dir"),
) -> None:
    """Compare human annotations against LLM diagnosis results."""
    from memeval.analysis import run_compare

    run_compare(str(human_dir), str(llm_dir), str(output_dir))


if __name__ == "__main__":
    app()

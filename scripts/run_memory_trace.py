#!/usr/bin/env python3
"""Run an end-to-end memory trace over LoCoMo or LongMemEval samples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from memeval.datasets.locomo import LoCoMoAdapter
from memeval.datasets.longmemeval import LongMemEvalAdapter
from memeval.generation import TracedGenerator
from memeval.generation.fake import FakeGenerationBackend
from memeval.runners import BackendSettings, backend_manifest
from memeval.runners import build_backend as shared_build_backend
from memeval.runners.memory_trace import MemoryTraceRunner
from memeval.storage.trace_store import TraceStore
from memeval.trace import TRACE_V2_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-type", choices=["locomo", "longmemeval"], required=True)
    parser.add_argument("--backend", choices=["fake", "mem0", "openclaw", "amem", "memoryos"], default="fake")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/memory-trace"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--model", default="fake-model")
    parser.add_argument("--generation-backend", choices=["fake", "openai"], default="fake")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--openclaw-bin", default=os.getenv("MEMEVAL_OPENCLAW_BIN", "openclaw"))
    parser.add_argument("--openclaw-agent", default=os.getenv("MEMEVAL_OPENCLAW_AGENT", "main"))
    parser.add_argument("--openclaw-agent-model", default=os.getenv("MEMEVAL_OPENCLAW_AGENT_MODEL", ""))
    parser.add_argument("--openclaw-profile", default=os.getenv("MEMEVAL_OPENCLAW_PROFILE", ""))
    parser.add_argument("--openclaw-session-prefix", default="")
    parser.add_argument(
        "--openclaw-workspace-root",
        type=Path,
        default=Path(os.getenv("MEMEVAL_OPENCLAW_WORKSPACE_ROOT", "data/input/openclaw_mem/workspaces")),
    )
    parser.add_argument("--openclaw-timeout", type=float, default=300.0)
    parser.add_argument(
        "--amem-persist-root",
        type=Path,
        default=Path(os.getenv("MEMEVAL_AMEM_PERSIST_ROOT", "data/input/amem_mem/persist")),
    )
    parser.add_argument("--amem-embed-model", default=os.getenv("MEMEVAL_AMEM_EMBED_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--amem-llm-backend", default=os.getenv("MEMEVAL_AMEM_LLM_BACKEND", "openai"))
    parser.add_argument("--amem-llm-model", default=os.getenv("MEMEVAL_AMEM_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--memoryos-storage-root",
        type=Path,
        default=Path(os.getenv("MEMEVAL_MEMORYOS_STORAGE_ROOT", "data/input/memoryos_mem/storage")),
    )
    parser.add_argument("--memoryos-llm-model", default=os.getenv("MEMEVAL_MEMORYOS_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--memoryos-embed-model",
        default=os.getenv("MEMEVAL_MEMORYOS_EMBED_MODEL", "all-MiniLM-L6-v2"),
    )
    parser.add_argument("--memoryos-assistant-id", default=os.getenv("MEMEVAL_MEMORYOS_ASSISTANT_ID", "memeval_assistant"))
    parser.add_argument("--mem0-mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--mem0-repo", default=os.getenv("MEMEVAL_MEM0_REPO", ""))
    parser.add_argument(
        "--mem0-store-dir",
        type=Path,
        default=Path(os.getenv("MEMEVAL_MEM0_STORE_DIR", "data/input/mem0_mem/store")),
    )
    parser.add_argument("--mem0-llm-model", default=os.getenv("MEMEVAL_MEM0_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--mem0-embedding-model",
        default=os.getenv("MEMEVAL_MEM0_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument("--mem0-llm-provider", default="openai")
    parser.add_argument("--mem0-embedder-provider", default="openai")
    parser.add_argument("--mem0-vector-store", default="qdrant")
    parser.add_argument("--mem0-collection", default="memeval_memories")
    parser.add_argument("--mem0-llm-api-key-env", default="")
    parser.add_argument("--mem0-llm-base-url", default="")
    parser.add_argument("--mem0-embedder-api-key-env", default="")
    parser.add_argument("--mem0-embedder-base-url", default="")
    return parser.parse_args()


def load_samples(args):
    adapter = LoCoMoAdapter() if args.dataset_type == "locomo" else LongMemEvalAdapter()
    samples = adapter.load(args.dataset)
    end = len(samples) if args.end is None else min(args.end, len(samples))
    if args.start < 0 or end < args.start:
        raise ValueError(f"Invalid range {args.start}:{end}")
    return samples, samples[args.start:end]


def backend_settings(name, args):
    """Build the shared BackendSettings from argparse output."""
    return BackendSettings(
        name=name,
        top_k=getattr(args, "top_k", 10),
        mem0={
            "mode": args.mem0_mode, "repo": args.mem0_repo, "store_dir": args.mem0_store_dir,
            "llm_model": args.mem0_llm_model, "embedding_model": args.mem0_embedding_model,
            "llm_provider": args.mem0_llm_provider, "embedder_provider": args.mem0_embedder_provider,
            "vector_store": args.mem0_vector_store, "collection": args.mem0_collection,
            "llm_api_key_env": args.mem0_llm_api_key_env, "llm_base_url": args.mem0_llm_base_url,
            "embedder_api_key_env": args.mem0_embedder_api_key_env,
            "embedder_base_url": args.mem0_embedder_base_url,
        },
        openclaw={
            "bin": args.openclaw_bin, "agent": args.openclaw_agent,
            "agent_model": args.openclaw_agent_model, "profile": args.openclaw_profile,
            "session_prefix": args.openclaw_session_prefix,
            "workspace_root": args.openclaw_workspace_root, "timeout": args.openclaw_timeout,
        },
        amem={
            "persist_root": args.amem_persist_root, "embed_model": args.amem_embed_model,
            "llm_backend": args.amem_llm_backend, "llm_model": args.amem_llm_model,
        },
        memoryos={
            "storage_root": args.memoryos_storage_root, "llm_model": args.memoryos_llm_model,
            "embed_model": args.memoryos_embed_model, "assistant_id": args.memoryos_assistant_id,
            "api_key": args.api_key or os.getenv("OPENAI_API_KEY", ""), "base_url": args.base_url,
        },
    )


def build_backend(name, args=None):
    """Delegates to memeval.runners.build_backend so the CLI and this script agree."""
    return shared_build_backend(backend_settings(name, args))


def build_generator(name: str, *, api_key: str | None, base_url: str, temperature: float | None):
    if name == "fake":
        return TracedGenerator(FakeGenerationBackend()), {}
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai generation requires the openai package") from exc
    if not api_key:
        raise RuntimeError("OpenAI generation requires --api-key or OPENAI_API_KEY")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    from memeval.generation.openai import OpenAIChatGenerationBackend
    parameters = {} if temperature is None else {"temperature": temperature}
    return TracedGenerator(OpenAIChatGenerationBackend(OpenAI(**client_kwargs))), parameters


def main() -> int:
    args = parse_args()
    samples, selected = load_samples(args)
    print(f"Loaded {len(samples)} samples")
    print(f"Selected samples: {args.start}..{args.start + len(selected) - 1}")
    if args.dry_run:
        print("Dry run complete")
        return 0
    backend = build_backend(args.backend, args)
    generator, generation_parameters = build_generator(
        args.generation_backend,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    runner = MemoryTraceRunner(
        backend,
        generator,
        args.top_k,
        args.context_limit,
        args.model,
        generation_parameters,
    )
    store = TraceStore(args.output_dir)
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "dataset_type": args.dataset_type,
        "memory_backend": args.backend,
        "generation_backend": args.generation_backend,
        "model": args.model,
        "top_k": args.top_k,
        "context_limit": args.context_limit,
        "temperature": args.temperature,
        "start": args.start,
        "end": args.end,
        "trace_schema_version": TRACE_V2_SCHEMA_VERSION,
    }
    manifest.update(backend_manifest(backend_settings(args.backend, args)))
    store.write_manifest(manifest)
    completed = store.completed_ids() if args.resume else set()
    count = 0
    skipped = 0
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
                envelope = runner.run_sample(sample, question)
                store.append(envelope)
                count += 1
            except Exception as exc:  # keep one failed sample from stopping a run
                store.append_error(record_id, exc)
                print(f"ERROR {record_id}: {exc}")
    store.export_legacy_json()
    summary = store.write_summary(
        selected_samples=len(selected),
        selected_questions=selected_questions,
        added=count,
        skipped=skipped,
    )
    print(f"Completed traces: {count}")
    print(f"Total completed traces: {summary['completed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

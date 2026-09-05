"""Backend construction and manifest metadata, shared by the CLI and scripts.

Keeping this out of the CLI layer means backend selection is testable without
argparse/typer, and the legacy scripts and the new CLI cannot drift apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BACKEND_CHOICES = ("fake", "mem0", "openclaw", "amem", "memoryos")


@dataclass
class BackendSettings:
    """Backend-agnostic knobs plus per-backend option bags."""

    name: str = "fake"
    top_k: int = 10
    mem0: dict[str, Any] = field(default_factory=dict)
    openclaw: dict[str, Any] = field(default_factory=dict)
    amem: dict[str, Any] = field(default_factory=dict)
    memoryos: dict[str, Any] = field(default_factory=dict)

    def options(self) -> dict[str, Any]:
        return getattr(self, self.name, {}) if self.name in {"mem0", "openclaw", "amem", "memoryos"} else {}


def build_backend(settings: BackendSettings) -> Any:
    name = settings.name
    if name == "fake":
        from memeval.memory.fake import FakeMemoryBackend

        return FakeMemoryBackend()
    if name == "mem0":
        return _build_mem0(settings.mem0)
    if name == "openclaw":
        from memeval.memory.openclaw import OpenClawBackend

        options = settings.openclaw
        return OpenClawBackend(
            workspace_root=Path(options["workspace_root"]),
            openclaw_bin=options.get("bin", "openclaw"),
            agent=options.get("agent", "main"),
            agent_model=options.get("agent_model", ""),
            openclaw_profile=options.get("profile", ""),
            session_prefix=options.get("session_prefix", ""),
            timeout=float(options.get("timeout", 300.0)),
        )
    if name == "amem":
        from memeval.memory.amem import AMemBackend

        options = settings.amem
        return AMemBackend(
            persist_root=Path(options["persist_root"]),
            model_name=options.get("embed_model", "all-MiniLM-L6-v2"),
            llm_backend=options.get("llm_backend", "openai"),
            llm_model=options.get("llm_model", "gpt-4o-mini"),
            api_key=options.get("api_key") or os.getenv("OPENAI_API_KEY", ""),
            base_url=options.get("base_url") or os.getenv("OPENAI_BASE_URL", ""),
        )
    if name == "memoryos":
        from memeval.memory.memoryos import MemoryOSBackend

        options = settings.memoryos
        return MemoryOSBackend(
            storage_root=Path(options["storage_root"]),
            openai_api_key=options.get("api_key") or os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=options.get("base_url") or None,
            llm_model=options.get("llm_model", "gpt-4o-mini"),
            assistant_id=options.get("assistant_id", "memeval_assistant"),
            embedding_model_name=options.get("embed_model", "all-MiniLM-L6-v2"),
            retrieval_queue_capacity=settings.top_k,
        )
    raise ValueError(f"Unknown memory backend: {name}")


def _build_mem0(options: dict[str, Any]) -> Any:
    from memeval.memory.mem0 import Mem0Backend
    from memeval.memory.mem0_config import build_local_config, create_cloud_client, create_local_client

    if options.get("mode", "local") == "cloud":
        return Mem0Backend(create_cloud_client())
    config = build_local_config(
        store_dir=Path(options["store_dir"]),
        llm_model=options.get("llm_model", "gpt-4o-mini"),
        embedding_model=options.get("embedding_model", "text-embedding-3-small"),
        llm_provider=options.get("llm_provider", "openai"),
        embedder_provider=options.get("embedder_provider", "openai"),
        vector_store_provider=options.get("vector_store", "qdrant"),
        collection_name=options.get("collection", "memeval_memories"),
        llm_api_key_env=options.get("llm_api_key_env", ""),
        llm_base_url=options.get("llm_base_url", ""),
        embedder_api_key_env=options.get("embedder_api_key_env", ""),
        embedder_base_url=options.get("embedder_base_url", ""),
    )
    return Mem0Backend(create_local_client(config, mem0_repo=options.get("repo") or None))


# Keys whose values must never reach a manifest (see docs/provenance.md).
_SECRET_KEYS = {"api_key", "openai_api_key"}


def backend_manifest(settings: BackendSettings) -> dict[str, Any]:
    """Backend config for the run manifest, with secrets stripped."""
    options = {key: value for key, value in settings.options().items() if key not in _SECRET_KEYS}
    options = {key: (str(value) if isinstance(value, Path) else value) for key, value in options.items()}
    if settings.name == "memoryos":
        # Documented constraints of the MemoryOS adapter; see memory/memoryos.py.
        options["retrieval_is_internal"] = True
        options["score_available"] = False
    return {settings.name: options} if options else {}

"""Mem0 client construction for the self-hosted (``Memory.from_config``) mode.

The cloud client (``MemoryClient``) has its backbone fixed server-side. The
self-hosted ``Memory`` accepts an explicit llm / embedder / vector_store config,
which is what makes swapping the backbone possible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


# Known non-default embedding dimensions; anything else falls back to 1536.
EMBEDDING_MODEL_DIMS = {
    "text-embedding-v4": 1024,
    "text-embedding-3-large": 3072,
    "bge-m3": 1024,
    "all-MiniLM-L6-v2": 384,
}


def embedding_model_dims(model: str) -> int:
    return EMBEDDING_MODEL_DIMS.get(model, 1536)


def add_mem0_repo_to_path(mem0_repo: Path | str | None) -> None:
    """Prepend a local mem0 checkout to sys.path when one is configured."""
    if not mem0_repo:
        return
    repo = str(mem0_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _provider_config(
    model: str,
    *,
    api_key_env: str = "",
    base_url: str = "",
    environ: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    environ = environ if environ is not None else os.environ
    config: dict[str, Any] = {"model": model, **(extra or {})}
    if api_key_env:
        api_key = environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")
        config["api_key"] = api_key
    if base_url:
        config["openai_base_url"] = base_url
    return config


def build_llm_config(
    model: str,
    *,
    api_key_env: str = "",
    base_url: str = "",
    temperature: float = 0.0,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    return _provider_config(
        model, api_key_env=api_key_env, base_url=base_url,
        environ=environ, extra={"temperature": temperature},
    )


def build_embedder_config(
    model: str,
    *,
    api_key_env: str = "",
    base_url: str = "",
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    return _provider_config(model, api_key_env=api_key_env, base_url=base_url, environ=environ)


def build_vector_store_config(
    store_dir: Path,
    embedding_model: str,
    *,
    provider: str = "qdrant",
    collection_name: str = "memeval_memories",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "config": {
            "path": str(Path(store_dir) / provider),
            "collection_name": collection_name,
            "embedding_model_dims": embedding_model_dims(embedding_model),
        },
    }


def build_local_config(
    *,
    store_dir: Path,
    llm_model: str,
    embedding_model: str,
    llm_provider: str = "openai",
    embedder_provider: str = "openai",
    vector_store_provider: str = "qdrant",
    collection_name: str = "memeval_memories",
    llm_api_key_env: str = "",
    llm_base_url: str = "",
    embedder_api_key_env: str = "",
    embedder_base_url: str = "",
    temperature: float = 0.0,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the full Memory.from_config payload."""
    store_dir = Path(store_dir)
    history_db = store_dir / "history.db"
    history_db.parent.mkdir(parents=True, exist_ok=True)
    (store_dir / vector_store_provider).mkdir(parents=True, exist_ok=True)
    return {
        "vector_store": build_vector_store_config(
            store_dir, embedding_model,
            provider=vector_store_provider, collection_name=collection_name,
        ),
        "history_db_path": str(history_db),
        "llm": {
            "provider": llm_provider,
            "config": build_llm_config(
                llm_model, api_key_env=llm_api_key_env, base_url=llm_base_url,
                temperature=temperature, environ=environ,
            ),
        },
        "embedder": {
            "provider": embedder_provider,
            "config": build_embedder_config(
                embedding_model, api_key_env=embedder_api_key_env,
                base_url=embedder_base_url, environ=environ,
            ),
        },
    }


def create_local_client(config: dict[str, Any], *, mem0_repo: Path | str | None = None) -> Any:
    add_mem0_repo_to_path(mem0_repo)
    try:
        from mem0 import Memory
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError("mem0 local mode requires the mem0ai package") from exc
    return Memory.from_config(config)


def create_cloud_client() -> Any:
    try:
        from mem0 import MemoryClient
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError("mem0 cloud mode requires the mem0ai package") from exc
    return MemoryClient(
        api_key=os.getenv("MEM0_API_KEY"),
        org_id=os.getenv("MEM0_ORGANIZATION_ID"),
        project_id=os.getenv("MEM0_PROJECT_ID"),
    )

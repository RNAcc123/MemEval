import pytest

from memeval.memory.mem0_config import (
    build_embedder_config,
    build_llm_config,
    build_local_config,
    build_vector_store_config,
    embedding_model_dims,
)


def test_llm_config_reads_api_key_from_named_env_var():
    config = build_llm_config(
        "qwen-max", api_key_env="DASHSCOPE_API_KEY", base_url="https://example/v1",
        environ={"DASHSCOPE_API_KEY": "secret"},
    )
    assert config["model"] == "qwen-max"
    assert config["api_key"] == "secret"
    assert config["openai_base_url"] == "https://example/v1"
    assert config["temperature"] == 0.0


def test_llm_config_raises_when_env_var_missing():
    with pytest.raises(ValueError, match="MISSING_KEY"):
        build_llm_config("m", api_key_env="MISSING_KEY", environ={})


def test_configs_omit_optional_fields_when_not_supplied():
    config = build_embedder_config("text-embedding-3-small", environ={})
    assert config == {"model": "text-embedding-3-small"}


def test_embedding_dims_fall_back_to_default():
    assert embedding_model_dims("text-embedding-v4") == 1024
    assert embedding_model_dims("bge-m3") == 1024
    assert embedding_model_dims("some-unknown-model") == 1536


def test_vector_store_dims_track_the_embedding_model(tmp_path):
    small = build_vector_store_config(tmp_path, "text-embedding-3-small")
    large = build_vector_store_config(tmp_path, "text-embedding-3-large")
    assert small["config"]["embedding_model_dims"] == 1536
    assert large["config"]["embedding_model_dims"] == 3072


def test_local_config_has_the_three_swappable_backbone_sections(tmp_path):
    config = build_local_config(
        store_dir=tmp_path / "store",
        llm_model="deepseek-chat",
        embedding_model="bge-m3",
        llm_base_url="https://api.deepseek.com",
        environ={},
    )
    assert config["llm"]["config"]["model"] == "deepseek-chat"
    assert config["llm"]["config"]["openai_base_url"] == "https://api.deepseek.com"
    assert config["embedder"]["config"]["model"] == "bge-m3"
    assert config["vector_store"]["config"]["embedding_model_dims"] == 1024
    assert config["history_db_path"].endswith("history.db")


def test_local_config_creates_store_directories(tmp_path):
    store = tmp_path / "store"
    build_local_config(
        store_dir=store, llm_model="m", embedding_model="e", environ={},
    )
    assert (store / "qdrant").is_dir()

"""tests/unit/test_config.py — src/common/config.py の単体テスト"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.common.config import (
    AppConfig,
    BudgetConfig,
    EmbeddingConfig,
    FAISSConfig,
    FAISSIndexConfig,
    FAISSScaleRule,
    GRPOConfig,
    LLMConfig,
    MetadataConfig,
    RAGConfig,
    RewardWeights,
    RouterConfig,
    SandboxConfig,
    SandboxResourceLimits,
    Settings,
    StudentModelConfig,
    TinyLoRAConfig,
    TrainingConfig,
    TrainingRunConfig,
    _build_init_kwargs,
    _deep_merge,
    _load_yaml,
    get_settings,
    load_settings,
)


# ============================================================================
# フィクスチャ
# ============================================================================


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """各テスト後に get_settings() のキャッシュをクリアする。"""
    yield
    get_settings.cache_clear()


@pytest.fixture()
def tmp_configs(tmp_path: Path) -> Path:
    """一時 configs ディレクトリに最小限の YAML ファイルを配置する。"""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    (configs_dir / "default.yaml").write_text(
        yaml.dump(
            {
                "app": {"name": "test-app", "version": "9.9.9", "port": 9000},
                "embedding": {"model": "test-model", "dim": 128},
            }
        )
    )
    (configs_dir / "faiss_config.yaml").write_text(
        yaml.dump(
            {
                "base_dir": "data/test_indices",
                "indices": {
                    "code": {
                        "dim": 128,
                        "initial_type": "Flat",
                        "metric": "inner_product",
                        "nprobe": 8,
                        "scale_rules": [
                            {"threshold": 1000, "migrate_to": "IVF64,Flat"}
                        ],
                    }
                },
            }
        )
    )
    (configs_dir / "llm_config.yaml").write_text(
        yaml.dump(
            {
                "providers": {
                    "anthropic": {
                        "default_model": "claude-test",
                        "haiku_model": "claude-haiku-test",
                    },
                    "openai": {"default_model": "gpt-test", "mini_model": "gpt-mini-test"},
                    "ollama": {"base_url": "http://test:11434", "default_model": "llama-test"},
                },
                "task_routing": {"query_parsing": "haiku", "response_generation": "sonnet"},
                "budget": {"daily_limit_usd": 5.0, "alert_threshold": 0.5},
            }
        )
    )
    (configs_dir / "sandbox_policy.yaml").write_text(
        yaml.dump(
            {
                "max_containers": 2,
                "timeout_seconds": 10,
                "resource_limits": {"memory": "128m", "cpu": "0.25", "pids": 50},
                "network": {"enabled": False, "allowed_domains": []},
                "filesystem": {
                    "read_only_root": True,
                    "writable_tmp": True,
                    "max_file_size_mb": 5,
                },
                "blocked": {"syscalls": ["mount"]},
            }
        )
    )
    (configs_dir / "training.yaml").write_text(
        yaml.dump(
            {
                "default": {
                    "algorithm": "grpo",
                    "adapter": "tinylora",
                    "reward": "composite",
                    "algorithm_kwargs": {
                        "group_size": 4,
                        "kl_coeff": 0.01,
                        "clip_ratio": 0.1,
                        "temperature": 0.8,
                    },
                    "adapter_kwargs": {
                        "frozen_rank": 1,
                        "projection_dim": 2,
                        "tie_factor": 3,
                        "target_modules": ["q_proj"],
                    },
                    "reward_kwargs": {
                        "weights": {
                            "correctness": 0.35,
                            "retrieval_quality": 0.20,
                            "exec_success": 0.20,
                            "efficiency": 0.10,
                            "memory_utilization": 0.15,
                        }
                    },
                    "training": {
                        "epochs": 1,
                        "batch_size": 8,
                        "learning_rate": 1e-3,
                        "max_generation_length": 512,
                        "seeds": [42],
                    },
                    "student_model": {
                        "name": "test-model",
                        "inference_engine": "transformers",
                    },
                }
            }
        )
    )
    (configs_dir / "model_router.yaml").write_text(
        yaml.dump(
            {
                "thresholds": {
                    "simple": {"min_faiss_score": 0.90, "min_results": 5},
                    "moderate": {"min_faiss_score": 0.70},
                },
                "cost_weights": {"prefer_student": 0.8, "prefer_teacher": 0.2},
            }
        )
    )
    (configs_dir / "retrievers.yaml").write_text(
        yaml.dump(
            {
                "max_results_per_source": 3,
                "timeout_seconds": 15,
                "verify_results": False,
                "chunk_size": 256,
                "chunk_overlap": 25,
            }
        )
    )

    return configs_dir


# ============================================================================
# デフォルト値テスト
# ============================================================================


class TestDefaultValues:
    """設定モデルのデフォルト値が正しいことを確認する。"""

    def test_app_config_defaults(self):
        cfg = AppConfig()
        assert cfg.name == "rag-faiss-llm"
        assert cfg.version == "0.4.0"
        assert cfg.debug is False
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.log_level == "INFO"

    def test_embedding_config_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.model == "all-MiniLM-L6-v2"
        assert cfg.dim == 768
        assert cfg.batch_size == 64
        assert cfg.device == "cpu"
        assert cfg.cache_dir is None

    def test_faiss_config_defaults(self):
        cfg = FAISSConfig()
        assert cfg.base_dir == Path("data/faiss_indices")
        assert "code" in cfg.indices
        assert "academic" in cfg.indices
        assert "general" in cfg.indices
        for idx in cfg.indices.values():
            assert idx.dim == 768
            assert idx.initial_type == "Flat"
            assert idx.metric == "inner_product"

    def test_metadata_config_defaults(self):
        cfg = MetadataConfig()
        assert cfg.db_path == Path("data/metadata.db")

    def test_llm_config_defaults(self):
        cfg = LLMConfig()
        assert cfg.anthropic.default_model == "claude-sonnet-4-20250514"
        assert cfg.anthropic.haiku_model == "claude-haiku-4-5-20251001"
        assert cfg.openai.default_model == "gpt-4o"
        assert cfg.ollama.base_url == "http://localhost:11434"
        assert cfg.budget.daily_limit_usd == 10.0
        assert cfg.task_routing["query_parsing"] == "haiku"
        assert cfg.task_routing["response_generation"] == "sonnet"

    def test_sandbox_config_defaults(self):
        cfg = SandboxConfig()
        assert cfg.max_containers == 5
        assert cfg.timeout_seconds == 30
        assert cfg.resource_limits.memory == "256m"
        assert cfg.network_enabled is False
        assert cfg.read_only_root is True
        assert "mount" in cfg.blocked_syscalls
        assert "ptrace" in cfg.blocked_syscalls

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.algorithm == "grpo"
        assert cfg.adapter == "tinylora"
        assert cfg.reward == "composite"
        assert cfg.algorithm_kwargs.group_size == 8
        assert cfg.adapter_kwargs.frozen_rank == 2
        assert cfg.adapter_kwargs.projection_dim == 4
        assert cfg.adapter_kwargs.tie_factor == 7
        assert cfg.reward_weights.correctness == pytest.approx(0.35)
        assert cfg.student_model.name == "Qwen/Qwen2.5-7B-Instruct"

    def test_reward_weights_sum_to_one(self):
        """デフォルトの Reward 重みの合計が 1.0 になることを確認。"""
        w = RewardWeights()
        total = w.correctness + w.retrieval_quality + w.exec_success + w.efficiency + w.memory_utilization
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_router_config_defaults(self):
        cfg = RouterConfig()
        assert cfg.thresholds.simple.min_faiss_score == pytest.approx(0.85)
        assert cfg.thresholds.simple.min_results == 3
        assert cfg.thresholds.moderate.min_faiss_score == pytest.approx(0.60)
        assert cfg.prefer_student_weight == pytest.approx(0.7)
        assert cfg.prefer_teacher_weight == pytest.approx(0.3)


# ============================================================================
# RewardWeights バリデーションテスト
# ============================================================================


class TestRewardWeightsValidation:
    def test_valid_weights(self):
        w = RewardWeights(
            correctness=0.4,
            retrieval_quality=0.2,
            exec_success=0.2,
            efficiency=0.1,
            memory_utilization=0.1,
        )
        assert w.correctness == pytest.approx(0.4)

    def test_invalid_weights_raise_error(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            RewardWeights(
                correctness=0.5,
                retrieval_quality=0.5,
                exec_success=0.5,
                efficiency=0.5,
                memory_utilization=0.5,
            )


# ============================================================================
# _load_yaml テスト
# ============================================================================


class TestLoadYaml:
    def test_load_existing_file(self, tmp_path: Path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnested:\n  a: 1\n")
        result = _load_yaml(yaml_file)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        result = _load_yaml(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_load_empty_file_returns_empty(self, tmp_path: Path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        result = _load_yaml(yaml_file)
        assert result == {}


# ============================================================================
# _deep_merge テスト
# ============================================================================


class TestDeepMerge:
    def test_basic_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}, "y": 10}
        override = {"x": {"b": 99, "c": 3}, "z": 20}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 99, "c": 3}, "y": 10, "z": 20}

    def test_override_wins_for_non_dict(self):
        base = {"key": [1, 2, 3]}
        override = {"key": [4, 5]}
        result = _deep_merge(base, override)
        assert result["key"] == [4, 5]

    def test_base_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}  # base は変更されない


# ============================================================================
# _build_init_kwargs テスト
# ============================================================================


class TestBuildInitKwargs:
    def test_loads_app_config(self, tmp_configs: Path):
        kwargs = _build_init_kwargs(tmp_configs)
        assert "app" in kwargs
        assert kwargs["app"]["name"] == "test-app"
        assert kwargs["app"]["port"] == 9000

    def test_loads_embedding_config(self, tmp_configs: Path):
        kwargs = _build_init_kwargs(tmp_configs)
        assert "embedding" in kwargs
        assert kwargs["embedding"]["model"] == "test-model"
        assert kwargs["embedding"]["dim"] == 128

    def test_loads_faiss_config(self, tmp_configs: Path):
        kwargs = _build_init_kwargs(tmp_configs)
        assert "faiss" in kwargs
        assert kwargs["faiss"]["base_dir"] == "data/test_indices"
        assert "code" in kwargs["faiss"]["indices"]
        code_idx = kwargs["faiss"]["indices"]["code"]
        assert code_idx["dim"] == 128
        assert len(code_idx["scale_rules"]) == 1

    def test_loads_llm_config(self, tmp_configs: Path):
        kwargs = _build_init_kwargs(tmp_configs)
        assert "llm" in kwargs
        assert kwargs["llm"]["anthropic"]["default_model"] == "claude-test"
        assert kwargs["llm"]["task_routing"]["query_parsing"] == "haiku"
        assert kwargs["llm"]["budget"]["daily_limit_usd"] == 5.0

    def test_loads_sandbox_config(self, tmp_configs: Path):
        kwargs = _build_init_kwargs(tmp_configs)
        assert "sandbox" in kwargs
        sandbox = kwargs["sandbox"]
        assert sandbox["max_containers"] == 2
        assert sandbox["timeout_seconds"] == 10
        assert sandbox["network_enabled"] is False
        assert sandbox["read_only_root"] is True
        assert "mount" in sandbox["blocked_syscalls"]

    def test_loads_training_config(self, tmp_configs: Path):
        kwargs = _build_init_kwargs(tmp_configs)
        assert "training" in kwargs
        training = kwargs["training"]
        assert training["algorithm"] == "grpo"
        assert training["algorithm_kwargs"]["group_size"] == 4
        assert training["adapter_kwargs"]["frozen_rank"] == 1
        assert training["student_model"]["name"] == "test-model"

    def test_loads_router_config(self, tmp_configs: Path):
        kwargs = _build_init_kwargs(tmp_configs)
        assert "router" in kwargs
        router = kwargs["router"]
        assert router["thresholds"]["simple"]["min_faiss_score"] == pytest.approx(0.90)
        assert router["prefer_student_weight"] == pytest.approx(0.8)

    def test_loads_rag_config(self, tmp_configs: Path):
        kwargs = _build_init_kwargs(tmp_configs)
        assert "rag" in kwargs
        assert kwargs["rag"]["max_results_per_source"] == 3
        assert kwargs["rag"]["timeout_seconds"] == 15

    def test_missing_configs_dir_returns_empty(self, tmp_path: Path):
        """存在しないディレクトリを指定しても例外を投げない。"""
        kwargs = _build_init_kwargs(tmp_path / "no_such_dir")
        assert kwargs == {}


# ============================================================================
# load_settings テスト
# ============================================================================


class TestLoadSettings:
    def test_loads_from_yaml(self, tmp_configs: Path):
        settings = load_settings(tmp_configs)
        assert settings.app.name == "test-app"
        assert settings.app.port == 9000
        assert settings.embedding.model == "test-model"
        assert settings.llm.anthropic.default_model == "claude-test"

    def test_faiss_indices_parsed(self, tmp_configs: Path):
        settings = load_settings(tmp_configs)
        assert "code" in settings.faiss.indices
        code_idx = settings.faiss.indices["code"]
        assert isinstance(code_idx, FAISSIndexConfig)
        assert code_idx.dim == 128
        assert len(code_idx.scale_rules) == 1
        assert isinstance(code_idx.scale_rules[0], FAISSScaleRule)

    def test_sandbox_parsed(self, tmp_configs: Path):
        settings = load_settings(tmp_configs)
        assert settings.sandbox.max_containers == 2
        assert isinstance(settings.sandbox.resource_limits, SandboxResourceLimits)
        assert settings.sandbox.resource_limits.memory == "128m"

    def test_training_parsed(self, tmp_configs: Path):
        settings = load_settings(tmp_configs)
        assert isinstance(settings.training.algorithm_kwargs, GRPOConfig)
        assert settings.training.algorithm_kwargs.group_size == 4
        assert isinstance(settings.training.adapter_kwargs, TinyLoRAConfig)
        assert settings.training.adapter_kwargs.frozen_rank == 1

    def test_env_var_overrides_yaml(self, tmp_configs: Path, monkeypatch: pytest.MonkeyPatch):
        """環境変数が YAML より高優先であることを確認。"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-from-env")
        get_settings.cache_clear()
        settings = load_settings(tmp_configs)
        assert settings.anthropic_api_key == "sk-test-key-from-env"

    def test_api_key_defaults_to_empty(self, tmp_configs: Path, monkeypatch: pytest.MonkeyPatch):
        """API キーが未設定のときに空文字列になることを確認。"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = load_settings(tmp_configs)
        assert settings.anthropic_api_key == ""
        assert settings.openai_api_key == ""

    def test_default_settings_without_yaml(self):
        """YAML なしでもデフォルト値で初期化できることを確認。"""
        settings = Settings()
        assert settings.app.name == "rag-faiss-llm"
        assert settings.embedding.dim == 768
        assert settings.llm.anthropic.default_model == "claude-sonnet-4-20250514"


# ============================================================================
# get_settings (シングルトン) テスト
# ============================================================================


class TestGetSettings:
    def test_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_singleton_returns_same_instance(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_allows_reload(self):
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # 異なるオブジェクトだが同じ値
        assert s1 is not s2
        assert s1.app.name == s2.app.name


# ============================================================================
# Settings ヘルパーメソッドテスト
# ============================================================================


class TestSettingsHelpers:
    def test_has_anthropic_key_false_when_empty(self):
        s = Settings()
        assert s.has_anthropic_key() is False

    def test_has_anthropic_key_true_when_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        s = Settings()
        assert s.has_anthropic_key() is True

    def test_has_openai_key_false_when_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings()
        assert s.has_openai_key() is False

    def test_has_github_token_false_when_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        s = Settings()
        assert s.has_github_token() is False

    def test_get_llm_model_anthropic_sonnet(self):
        s = Settings()
        model = s.get_llm_model("response_generation", "anthropic")
        assert model == s.llm.anthropic.default_model

    def test_get_llm_model_anthropic_haiku(self):
        s = Settings()
        model = s.get_llm_model("query_parsing", "anthropic")
        assert model == s.llm.anthropic.haiku_model

    def test_get_llm_model_openai_default(self):
        s = Settings()
        model = s.get_llm_model("verification", "openai")
        assert model == s.llm.openai.default_model

    def test_get_llm_model_openai_mini(self):
        s = Settings()
        model = s.get_llm_model("query_parsing", "openai")
        assert model == s.llm.openai.mini_model

    def test_get_llm_model_unknown_provider_raises(self):
        s = Settings()
        with pytest.raises(ValueError, match="Unknown provider"):
            s.get_llm_model("query_parsing", "unknown_provider")

    def test_get_llm_model_unknown_task_defaults_to_sonnet(self):
        """未知のタスク種別は sonnet にフォールバックする。"""
        s = Settings()
        model = s.get_llm_model("nonexistent_task", "anthropic")
        assert model == s.llm.anthropic.default_model


# ============================================================================
# FAISSConfig テスト
# ============================================================================


class TestFAISSConfig:
    def test_scale_rules_are_ordered(self, tmp_configs: Path):
        settings = load_settings(tmp_configs)
        if "code" in settings.faiss.indices:
            rules = settings.faiss.indices["code"].scale_rules
            thresholds = [r.threshold for r in rules]
            assert thresholds == sorted(thresholds), "scale_rules は threshold 昇順であるべき"

    def test_faiss_index_config_metric(self):
        idx = FAISSIndexConfig(metric="l2")
        assert idx.metric == "l2"

    def test_faiss_scale_rule(self):
        rule = FAISSScaleRule(threshold=100_000, migrate_to="IVF1024,Flat")
        assert rule.threshold == 100_000
        assert rule.migrate_to == "IVF1024,Flat"


# ============================================================================
# TinyLoRA 設定テスト (CLAUDE.md の重要設計判断に準拠)
# ============================================================================


class TestTinyLoRAConfig:
    def test_default_tinylora_matches_paper(self):
        """TinyLoRA論文 (Morris et al., 2026) の推奨値であることを確認。"""
        cfg = TinyLoRAConfig()
        assert cfg.frozen_rank == 2
        assert cfg.projection_dim == 4
        assert cfg.tie_factor == 7

    def test_default_target_modules_contain_attention(self):
        cfg = TinyLoRAConfig()
        attention_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
        assert attention_modules.issubset(set(cfg.target_modules))

    def test_default_target_modules_contain_ffn(self):
        cfg = TinyLoRAConfig()
        ffn_modules = {"up_proj", "down_proj", "gate_proj"}
        assert ffn_modules.issubset(set(cfg.target_modules))

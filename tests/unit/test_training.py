"""tests/unit/test_training.py — Phase 3 学習フレームワークの単体テスト

torch のない環境でも動く部分をすべてテストする。
torch-dependent テストは HAS_TORCH フラグで制御する。

テスト対象:
- TrainingBatch / TrainingConfig / base classes
- TrainingRegistry
- Algorithms: GRPO, PPO, DPO, SFT, REINFORCE
- Adapters: TinyLoRA, LoRA, LoRA-XS, FullFT
- Rewards: Composite, CodeExec, TeacherEval, Hybrid
- TrainingLogger
- TrainingPipeline
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

import src.training.adapters.full_ft  # noqa: F401
import src.training.adapters.lora  # noqa: F401
import src.training.adapters.lora_xs  # noqa: F401
import src.training.adapters.tinylora  # noqa: F401
import src.training.algorithms.dpo  # noqa: F401

# Import to trigger registry registration
import src.training.algorithms.grpo  # noqa: F401
import src.training.algorithms.ppo  # noqa: F401
import src.training.algorithms.reinforce  # noqa: F401
import src.training.algorithms.sft  # noqa: F401
import src.training.rewards.code_exec  # noqa: F401
import src.training.rewards.composite  # noqa: F401
import src.training.rewards.hybrid  # noqa: F401
import src.training.rewards.teacher_eval  # noqa: F401
from src.training.adapters.full_ft import FullFTAdapter
from src.training.adapters.lora import LoRAAdapter
from src.training.adapters.lora_xs import LoRAXSAdapter
from src.training.adapters.tinylora import TinyLoRAAdapter
from src.training.algorithms.dpo import DPOAlgorithm
from src.training.algorithms.grpo import GRPOAlgorithm
from src.training.algorithms.ppo import PPOAlgorithm
from src.training.algorithms.reinforce import REINFORCEAlgorithm
from src.training.algorithms.sft import SFTAlgorithm
from src.training.base import (
    RewardFunction,
    TrainingAlgorithm,
    TrainingBatch,
    TrainingResult,
    TrainingStep,
)
from src.training.logger import TrainingLogger
from src.training.pipeline import PipelineConfig, TrainingPipeline
from src.training.registry import TrainingRegistry
from src.training.rewards.code_exec import CodeExecReward
from src.training.rewards.composite import CompositeReward, RewardBreakdown
from src.training.rewards.hybrid import HybridReward
from src.training.rewards.teacher_eval import TeacherEvalReward

# ──────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────


def _make_batch(
    n: int = 4,
    with_rewards: bool = True,
) -> TrainingBatch:
    prompts = [f"Question {i}" for i in range(n)]
    responses = [f"Answer {i}" for i in range(n)]
    rewards = [float(i % 2) for i in range(n)] if with_rewards else []
    return TrainingBatch(prompts=prompts, responses=responses, rewards=rewards)


def _make_model(hidden_dim: int = 16) -> nn.Sequential:
    if not HAS_TORCH:
        return None  # type: ignore[return-value]
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


# ──────────────────────────────────────────────
# TrainingBatch テスト (no torch required)
# ──────────────────────────────────────────────


class TestTrainingBatch:
    def test_len(self) -> None:
        batch = _make_batch(6)
        assert len(batch) == 6

    def test_validate_ok(self) -> None:
        batch = _make_batch(4)
        batch.validate()

    def test_validate_mismatched_responses(self) -> None:
        batch = TrainingBatch(
            prompts=["a", "b"],
            responses=["x"],
            rewards=[0.5, 0.8],
        )
        with pytest.raises(ValueError, match="prompts"):
            batch.validate()

    def test_validate_mismatched_rewards(self) -> None:
        batch = TrainingBatch(
            prompts=["a", "b"],
            responses=["x", "y"],
            rewards=[0.5],
        )
        with pytest.raises(ValueError, match="rewards"):
            batch.validate()

    def test_metadata_default_empty(self) -> None:
        batch = _make_batch(2)
        assert batch.metadata == []


# ──────────────────────────────────────────────
# TrainingRegistry テスト (no torch required)
# ──────────────────────────────────────────────


class TestTrainingRegistry:
    def test_list_algorithms(self) -> None:
        algos = TrainingRegistry.list_algorithms()
        assert "grpo" in algos
        assert "ppo" in algos
        assert "dpo" in algos
        assert "sft" in algos
        assert "reinforce" in algos

    def test_list_adapters(self) -> None:
        adapters = TrainingRegistry.list_adapters()
        assert "tinylora" in adapters
        assert "lora" in adapters
        assert "lora_xs" in adapters
        assert "full_ft" in adapters

    def test_list_rewards(self) -> None:
        rewards = TrainingRegistry.list_rewards()
        assert "composite" in rewards
        assert "code_exec" in rewards
        assert "teacher_eval" in rewards
        assert "hybrid" in rewards

    def test_get_algorithm(self) -> None:
        cls = TrainingRegistry.get_algorithm("grpo")
        assert cls is GRPOAlgorithm

    def test_get_adapter(self) -> None:
        cls = TrainingRegistry.get_adapter("tinylora")
        assert cls is TinyLoRAAdapter

    def test_get_reward(self) -> None:
        cls = TrainingRegistry.get_reward("composite")
        assert cls is CompositeReward

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            TrainingRegistry.get_algorithm("nonexistent")

    def test_summary(self) -> None:
        s = TrainingRegistry.summary()
        assert "algorithms" in s
        assert "adapters" in s
        assert "rewards" in s

    def test_register_custom(self) -> None:
        @TrainingRegistry.algorithm("test_algo_xyz")
        class _TestAlgo(TrainingAlgorithm):
            @property
            def name(self): return "test_algo_xyz"
            def compute_loss(self, batch, model, adapter): return None

        assert "test_algo_xyz" in TrainingRegistry.list_algorithms()
        # cleanup
        del TrainingRegistry._algorithms["test_algo_xyz"]


# ──────────────────────────────────────────────
# Algorithm: name + instantiation (no torch)
# ──────────────────────────────────────────────


class TestAlgorithmNames:
    def test_grpo_name(self) -> None:
        assert GRPOAlgorithm().name == "grpo"

    def test_ppo_name(self) -> None:
        assert PPOAlgorithm().name == "ppo"

    def test_dpo_name(self) -> None:
        assert DPOAlgorithm().name == "dpo"

    def test_sft_name(self) -> None:
        assert SFTAlgorithm().name == "sft"

    def test_reinforce_name(self) -> None:
        assert REINFORCEAlgorithm().name == "reinforce"

    def test_reinforce_invalid_baseline(self) -> None:
        with pytest.raises(ValueError, match="baseline"):
            REINFORCEAlgorithm(baseline="unknown")

    def test_dpo_requires_even_batch_size(self) -> None:
        """DPO requires even batch. Validation happens in compute_loss but error msg should match."""
        algo = DPOAlgorithm()
        batch = _make_batch(3)
        # If torch is available, this raises ValueError; without torch it might too
        if HAS_TORCH:
            with pytest.raises(ValueError, match="even"):
                algo.compute_loss(batch, None, None)


# ──────────────────────────────────────────────
# Algorithm compute_loss with torch
# ──────────────────────────────────────────────


class TestGRPOAlgorithmTorch:
    @requires_torch
    def test_compute_loss_returns_tensor(self) -> None:
        algo = GRPOAlgorithm()
        batch = _make_batch(4)
        loss = algo.compute_loss(batch, None, None)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

    @requires_torch
    def test_compute_loss_requires_rewards(self) -> None:
        algo = GRPOAlgorithm()
        batch = _make_batch(4, with_rewards=False)
        with pytest.raises(ValueError, match="rewards"):
            algo.compute_loss(batch, None, None)

    @requires_torch
    def test_loss_differentiable(self) -> None:
        algo = GRPOAlgorithm()
        batch = _make_batch(4)
        loss = algo.compute_loss(batch, None, None)
        loss.backward()

    @requires_torch
    def test_constant_rewards_near_zero_loss(self) -> None:
        algo = GRPOAlgorithm()
        batch = TrainingBatch(
            prompts=["q"] * 4,
            responses=["a"] * 4,
            rewards=[1.0] * 4,
        )
        loss = algo.compute_loss(batch, None, None)
        assert abs(loss.item()) < 1e-3


class TestPPOAlgorithmTorch:
    @requires_torch
    def test_compute_loss(self) -> None:
        algo = PPOAlgorithm()
        batch = _make_batch(4)
        loss = algo.compute_loss(batch, None, None)
        assert isinstance(loss, torch.Tensor)


class TestDPOAlgorithmTorch:
    @requires_torch
    def test_compute_loss_even(self) -> None:
        algo = DPOAlgorithm()
        batch = _make_batch(4)
        loss = algo.compute_loss(batch, None, None)
        assert isinstance(loss, torch.Tensor)

    @requires_torch
    def test_reference_free(self) -> None:
        algo = DPOAlgorithm(reference_free=True)
        batch = _make_batch(4)
        loss = algo.compute_loss(batch, None, None)
        assert isinstance(loss, torch.Tensor)


class TestSFTAlgorithmTorch:
    @requires_torch
    def test_compute_loss(self) -> None:
        algo = SFTAlgorithm()
        batch = _make_batch(4)
        loss = algo.compute_loss(batch, None, None)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0


class TestREINFORCEAlgorithmTorch:
    @requires_torch
    def test_compute_loss(self) -> None:
        algo = REINFORCEAlgorithm()
        batch = _make_batch(4)
        loss = algo.compute_loss(batch, None, None)
        assert isinstance(loss, torch.Tensor)

    @requires_torch
    def test_no_baseline(self) -> None:
        algo = REINFORCEAlgorithm(baseline="none")
        batch = _make_batch(4)
        loss = algo.compute_loss(batch, None, None)
        assert isinstance(loss, torch.Tensor)


# ──────────────────────────────────────────────
# Adapter: name + instantiation (no torch required)
# ──────────────────────────────────────────────


class TestAdapterNames:
    def test_tinylora_name(self) -> None:
        assert TinyLoRAAdapter.__name__ == "TinyLoRAAdapter"

    def test_lora_name(self) -> None:
        assert LoRAAdapter.__name__ == "LoRAAdapter"

    def test_lora_xs_name(self) -> None:
        assert LoRAXSAdapter.__name__ == "LoRAXSAdapter"

    def test_full_ft_name(self) -> None:
        assert FullFTAdapter.__name__ == "FullFTAdapter"


class TestTinyLoRAAdapterTorch:
    @requires_torch
    def test_name(self) -> None:
        assert TinyLoRAAdapter(hidden_dim=16).name == "tinylora"

    @requires_torch
    def test_trainable_params_count(self) -> None:
        adapter = TinyLoRAAdapter(hidden_dim=16, frozen_rank=2, projection_dim=4)
        assert adapter.num_trainable_params == 8  # B: (2,4)

    @requires_torch
    def test_get_trainable_params(self) -> None:
        adapter = TinyLoRAAdapter(hidden_dim=16)
        params = adapter.get_trainable_params()
        assert len(params) == 1
        assert params[0].requires_grad

    @requires_torch
    def test_apply_to_model(self) -> None:
        model = _make_model(16)
        adapter = TinyLoRAAdapter(hidden_dim=16, frozen_rank=2, projection_dim=4)
        adapter.apply_to(model)

    @requires_torch
    def test_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = TinyLoRAAdapter(hidden_dim=16, frozen_rank=2, projection_dim=4)
            path = str(Path(d) / "tinylora.pkl")
            adapter.save(path)
            assert Path(path).exists()

            adapter2 = TinyLoRAAdapter(hidden_dim=16, frozen_rank=2, projection_dim=4)
            adapter2.load(path)
            assert torch.allclose(adapter._B, adapter2._B)


class TestLoRAAdapterTorch:
    @requires_torch
    def test_name(self) -> None:
        assert LoRAAdapter(hidden_dim=16).name == "lora"

    @requires_torch
    def test_trainable_params_count(self) -> None:
        adapter = LoRAAdapter(hidden_dim=16, rank=4)
        assert adapter.num_trainable_params == 128  # A(16,4)+B(4,16)

    @requires_torch
    def test_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = LoRAAdapter(hidden_dim=16, rank=4)
            adapter._lora_B.data.fill_(0.5)
            path = str(Path(d) / "lora.pkl")
            adapter.save(path)
            adapter2 = LoRAAdapter(hidden_dim=16, rank=4)
            adapter2.load(path)
            assert torch.allclose(adapter._lora_B, adapter2._lora_B)


class TestLoRAXSAdapterTorch:
    @requires_torch
    def test_name(self) -> None:
        assert LoRAXSAdapter(hidden_dim=16).name == "lora_xs"

    @requires_torch
    def test_trainable_params_count(self) -> None:
        adapter = LoRAXSAdapter(hidden_dim=16, n_scalars=4)
        assert adapter.num_trainable_params == 4


class TestFullFTAdapterTorch:
    @requires_torch
    def test_name(self) -> None:
        assert FullFTAdapter().name == "full_ft"

    @requires_torch
    def test_apply_to_model(self) -> None:
        model = _make_model(16)
        adapter = FullFTAdapter()
        adapter.apply_to(model)
        assert adapter.num_trainable_params > 0


# ──────────────────────────────────────────────
# CompositeReward テスト (no torch required)
# ──────────────────────────────────────────────


class TestCompositeReward:
    def test_name(self) -> None:
        assert CompositeReward().name == "composite"

    def test_invalid_weights(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            CompositeReward(weights={"correctness": 0.5, "retrieval_quality": 0.2})

    @pytest.mark.asyncio
    async def test_compute_returns_float(self) -> None:
        reward = CompositeReward()
        score = await reward.compute("What is X?", "X is a thing.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_compute_breakdown(self) -> None:
        reward = CompositeReward()
        bd = await reward.compute_breakdown("Q", "A")
        assert isinstance(bd, RewardBreakdown)
        assert 0.0 <= bd.composite <= 1.0

    @pytest.mark.asyncio
    async def test_metadata_overrides(self) -> None:
        reward = CompositeReward()
        meta = {
            "correctness_score": 1.0,
            "retrieval_score": 0.9,
            "exec_success": True,
            "context_doc_ids": ["doc1", "doc2"],
        }
        score = await reward.compute("Q", "A" * 100, meta)
        assert score > 0.5

    @pytest.mark.asyncio
    async def test_empty_response_low_efficiency(self) -> None:
        reward = CompositeReward()
        bd = await reward.compute_breakdown("Q", "")
        assert bd.efficiency == 0.0

    @pytest.mark.asyncio
    async def test_very_long_response_efficiency_penalty(self) -> None:
        reward = CompositeReward(max_response_chars=100)
        bd = await reward.compute_breakdown("Q", "A" * 2000)
        assert bd.efficiency < 1.0

    @pytest.mark.asyncio
    async def test_compute_batch(self) -> None:
        reward = CompositeReward()
        batch = _make_batch(3, with_rewards=False)
        result = await reward.compute_batch(batch)
        assert len(result.rewards) == 3
        assert all(0.0 <= r <= 1.0 for r in result.rewards)

    @pytest.mark.asyncio
    async def test_no_context_low_memory_utilization(self) -> None:
        reward = CompositeReward()
        bd = await reward.compute_breakdown("Q", "Answer")
        assert bd.memory_utilization < 0.5

    @pytest.mark.asyncio
    async def test_with_context_doc_ids(self) -> None:
        reward = CompositeReward()
        bd = await reward.compute_breakdown("Q", "Answer", {"context_doc_ids": ["d1", "d2"]})
        assert bd.memory_utilization == 1.0  # 2 docs = ideal


# ──────────────────────────────────────────────
# CodeExecReward テスト
# ──────────────────────────────────────────────


class TestCodeExecReward:
    def test_name(self) -> None:
        assert CodeExecReward().name == "code_exec"

    @pytest.mark.asyncio
    async def test_no_code_returns_no_code_score(self) -> None:
        reward = CodeExecReward(no_code_score=0.4)
        score = await reward.compute("Q", "No code here.")
        assert score == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_meta_exec_success(self) -> None:
        reward = CodeExecReward(success_score=1.0, failure_score=0.0)
        score = await reward.compute("Q", "A", {"exec_success": True})
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_meta_exec_failure(self) -> None:
        reward = CodeExecReward(success_score=1.0, failure_score=0.0)
        score = await reward.compute("Q", "A", {"exec_success": False})
        assert score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_no_sandbox_half_score(self) -> None:
        reward = CodeExecReward(success_score=1.0, failure_score=0.0, no_code_score=0.3)
        response = "```python\nprint('hello')\n```"
        score = await reward.compute("Q", response)
        assert score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_extract_code_block(self) -> None:
        reward = CodeExecReward()
        code = reward._extract_code("Some text\n```python\nprint(1)\n```\nmore text")
        assert "print(1)" in code

    @pytest.mark.asyncio
    async def test_no_code_block_extracts_empty(self) -> None:
        reward = CodeExecReward()
        code = reward._extract_code("Just plain text, no code.")
        assert code == ""


# ──────────────────────────────────────────────
# TeacherEvalReward テスト
# ──────────────────────────────────────────────


class TestTeacherEvalReward:
    def test_name(self) -> None:
        assert TeacherEvalReward().name == "teacher_eval"

    @pytest.mark.asyncio
    async def test_no_gateway_returns_fallback(self) -> None:
        reward = TeacherEvalReward(gateway=None, fallback_score=0.42)
        score = await reward.compute("Q", "A")
        assert score == pytest.approx(0.42)


# ──────────────────────────────────────────────
# HybridReward テスト
# ──────────────────────────────────────────────


class TestHybridReward:
    def test_name(self) -> None:
        assert HybridReward().name == "hybrid"

    def test_invalid_weights(self) -> None:
        r1 = CodeExecReward()
        r2 = TeacherEvalReward()
        with pytest.raises(ValueError, match="sum"):
            HybridReward([(r1, 0.3), (r2, 0.3)])

    @pytest.mark.asyncio
    async def test_empty_returns_half(self) -> None:
        reward = HybridReward()
        score = await reward.compute("Q", "A")
        assert score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_weighted_combination(self) -> None:
        class ConstantReward(RewardFunction):
            def __init__(self, value: float):
                self._value = value

            @property
            def name(self): return "const"

            async def compute(self, prompt, response, metadata=None):
                return self._value

        r_high = ConstantReward(1.0)
        r_low = ConstantReward(0.0)
        hybrid = HybridReward([(r_high, 0.7), (r_low, 0.3)])
        score = await hybrid.compute("Q", "A")
        assert score == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_add_component(self) -> None:
        hybrid = HybridReward()
        r = CodeExecReward()
        hybrid.add_component(r, 1.0)
        assert len(hybrid._components) == 1


# ──────────────────────────────────────────────
# TrainingLogger テスト (no torch required)
# ──────────────────────────────────────────────


class TestTrainingLogger:
    def _make_step(self, step_n: int = 1) -> TrainingStep:
        return TrainingStep(
            step=step_n,
            loss=0.5 / step_n,
            reward_mean=0.3 * step_n,
            reward_std=0.1,
            algorithm="grpo",
            adapter="tinylora",
        )

    def test_log_and_retrieve(self) -> None:
        logger = TrainingLogger(run_name="test", use_wandb=False)
        step = self._make_step(1)
        logger.log_step(step)
        assert len(logger.steps) == 1
        assert logger.latest_step == step

    def test_summary(self) -> None:
        logger = TrainingLogger(run_name="test", use_wandb=False)
        for i in range(1, 4):
            logger.log_step(self._make_step(i))
        s = logger.summary()
        assert s["total_steps"] == 3
        assert "best_reward" in s
        assert "final_loss" in s

    def test_summary_empty(self) -> None:
        logger = TrainingLogger(use_wandb=False)
        assert logger.summary() == {}

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "log.jsonl")
            logger = TrainingLogger(run_name="run1", use_wandb=False)
            for i in range(1, 4):
                logger.log_step(self._make_step(i))
            logger.save(path)

            assert Path(path).exists()
            loaded = TrainingLogger.load(path, run_name="run1")
            assert len(loaded.steps) == 3
            assert loaded.steps[0].step == 1

    def test_finish_no_wandb(self) -> None:
        logger = TrainingLogger(use_wandb=False)
        logger.finish()

    def test_log_metrics(self) -> None:
        logger = TrainingLogger(use_wandb=False)
        logger.log_metrics({"eval_reward": 0.75}, step=100)

    def test_save_creates_parent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "subdir" / "run.jsonl")
            logger = TrainingLogger(use_wandb=False)
            logger.log_step(self._make_step(1))
            logger.save(path)
            assert Path(path).exists()

    def test_steps_copy(self) -> None:
        """steps プロパティは内部リストのコピーを返す。"""
        logger = TrainingLogger(use_wandb=False)
        s = self._make_step(1)
        logger.log_step(s)
        steps = logger.steps
        steps.clear()
        assert len(logger.steps) == 1  # 内部は変わらない


# ──────────────────────────────────────────────
# TrainingPipeline テスト
# ──────────────────────────────────────────────


class TestTrainingPipeline:
    def _make_config(
        self,
        sft_steps: int = 2,
        grpo_steps: int = 3,
        ckpt_dir: str = "",
    ) -> PipelineConfig:
        return PipelineConfig(
            sft_steps=sft_steps,
            grpo_steps=grpo_steps,
            algorithm="grpo",
            adapter="tinylora",
            reward="composite",
            adapter_params={"hidden_dim": 16},
            checkpoint_dir=ckpt_dir or tempfile.mkdtemp(),
            run_name="test_run",
            use_wandb=False,
            save_interval=9999,
        )

    @pytest.mark.asyncio
    async def test_run_returns_result(self) -> None:
        config = self._make_config()
        pipeline = TrainingPipeline.from_config(config, model=_make_model(16))
        batches = [_make_batch(4, with_rewards=False) for _ in range(3)]
        result = await pipeline.run(batches)
        assert isinstance(result, TrainingResult)
        assert result.algorithm == "grpo"
        assert result.adapter == "tinylora"

    @pytest.mark.asyncio
    async def test_run_all_steps_executed(self) -> None:
        config = self._make_config(sft_steps=2, grpo_steps=3)
        pipeline = TrainingPipeline.from_config(config, model=_make_model(16))
        batches = [_make_batch(4, with_rewards=False)]
        result = await pipeline.run(batches)
        assert result.total_steps == 5

    @pytest.mark.asyncio
    async def test_run_no_sft(self) -> None:
        config = self._make_config(sft_steps=0, grpo_steps=3)
        pipeline = TrainingPipeline.from_config(config, model=_make_model(16))
        batches = [_make_batch(4, with_rewards=False)]
        result = await pipeline.run(batches)
        assert result.total_steps == 3

    @pytest.mark.asyncio
    async def test_run_empty_batches(self) -> None:
        config = self._make_config()
        pipeline = TrainingPipeline.from_config(config, model=_make_model(16))
        result = await pipeline.run([])
        assert isinstance(result, TrainingResult)

    @pytest.mark.asyncio
    async def test_from_config_instantiates_components(self) -> None:
        config = PipelineConfig(
            sft_steps=1,
            grpo_steps=1,
            algorithm="grpo",
            adapter="lora",
            reward="code_exec",
            adapter_params={"hidden_dim": 16},
            checkpoint_dir=tempfile.mkdtemp(),
            use_wandb=False,
        )
        pipeline = TrainingPipeline.from_config(config, model=_make_model(16))
        assert pipeline._algorithm.name == "grpo"
        assert pipeline._adapter.name == "lora"
        assert pipeline._reward_fn.name == "code_exec"

    @pytest.mark.asyncio
    async def test_logger_records_steps(self) -> None:
        config = self._make_config(sft_steps=2, grpo_steps=3)
        pipeline = TrainingPipeline.from_config(config, model=_make_model(16))
        batches = [_make_batch(4, with_rewards=False)]
        await pipeline.run(batches)
        assert len(pipeline._logger.steps) == 5

    @pytest.mark.asyncio
    async def test_total_steps_property(self) -> None:
        config = self._make_config(sft_steps=10, grpo_steps=20)
        assert config.total_steps == 30

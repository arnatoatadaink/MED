"""src/training/rewards/composite.py — Composite 報酬関数

CLAUDE.md 設計仕様:
    correctness(0.35) + retrieval_quality(0.20) + exec_success(0.20)
    + efficiency(0.10) + memory_utilization(0.15)

各サブ報酬の役割:
- correctness: LLM の出力が正しいかを Teacher が評価 (0〜1)
- retrieval_quality: 取得した文書の関連度 (0〜1)
- exec_success: コード実行が成功したか (0 or 1)
- efficiency: 応答の簡潔さ・速度 (0〜1)
- memory_utilization: FAISSメモリを有効活用できたか (0〜1)

使い方:
    from src.training.rewards.composite import CompositeReward

    reward_fn = CompositeReward(gateway)
    score = await reward_fn.compute(prompt, response, metadata)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.training.base import RewardFunction
from src.training.registry import TrainingRegistry

logger = logging.getLogger(__name__)

# デフォルト重み (合計 = 1.0)
_DEFAULT_WEIGHTS = {
    "correctness": 0.35,
    "retrieval_quality": 0.20,
    "exec_success": 0.20,
    "efficiency": 0.10,
    "memory_utilization": 0.15,
}


@dataclass
class RewardBreakdown:
    """報酬の内訳。"""

    correctness: float = 0.0
    retrieval_quality: float = 0.0
    exec_success: float = 0.0
    efficiency: float = 0.0
    memory_utilization: float = 0.0
    composite: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "retrieval_quality": self.retrieval_quality,
            "exec_success": self.exec_success,
            "efficiency": self.efficiency,
            "memory_utilization": self.memory_utilization,
            "composite": self.composite,
        }


@TrainingRegistry.reward("composite")
class CompositeReward(RewardFunction):
    """多信号加重合計 Composite 報酬関数。

    Args:
        gateway: Teacher 評価に使用する LLMGateway (省略可)。
        weights: サブ報酬の重み辞書。合計が 1.0 になること。
        provider: 優先 LLM プロバイダ。
        max_response_chars: 効率スコア計算の基準最大文字数。
        curio_coef: CURIO 情報利得報酬の係数 (0 で無効)。
            正の値を設定すると、metadata['new_doc_ids'] から
            FAISSで新たに得た情報量を内発的報酬として加算する。
            memory_utilization の代替/補完として使用。
            推奨値: 0.05〜0.10（memory_utilization の重みを下げて調整）。
    """

    def __init__(
        self,
        gateway: Any = None,
        weights: dict[str, float] | None = None,
        provider: str | None = None,
        max_response_chars: int = 2000,
        curio_coef: float = 0.0,
    ) -> None:
        self._gateway = gateway
        self._weights = weights or dict(_DEFAULT_WEIGHTS)
        self._provider = provider
        self._max_response_chars = max_response_chars
        self._curio_coef = curio_coef

        total = sum(self._weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Reward weights must sum to 1.0, got {total:.4f}")

    @property
    def name(self) -> str:
        return "composite"

    async def compute(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Composite 報酬スコアを計算する。

        Returns:
            0.0〜1.0 の複合報酬スコア。
        """
        bd = await self.compute_breakdown(prompt, response, metadata)
        return bd.composite

    async def compute_breakdown(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> RewardBreakdown:
        """各サブ報酬の内訳付きで計算する。"""
        meta = metadata or {}

        correctness = await self._correctness(prompt, response, meta)
        retrieval_quality = self._retrieval_quality(meta)
        exec_success = self._exec_success(meta)
        efficiency = self._efficiency(response)
        memory_utilization = self._memory_utilization(meta)

        # CURIO: 情報利得報酬（内発的報酬）
        # 新規取得ドキュメント数を「ユーザーについて新たに学んだ情報量」の代理指標とする
        # metadata['new_doc_ids'] = 前ターンと比較して新たに参照したドキュメントIDリスト
        curio_bonus = self._curio_information_gain(meta) if self._curio_coef > 0 else 0.0

        w = self._weights
        composite = (
            w["correctness"] * correctness
            + w["retrieval_quality"] * retrieval_quality
            + w["exec_success"] * exec_success
            + w["efficiency"] * efficiency
            + w["memory_utilization"] * memory_utilization
            + self._curio_coef * curio_bonus
        )
        # curio_coef > 0 の場合でも 0〜1 の範囲に正規化
        if self._curio_coef > 0:
            composite = composite / (1.0 + self._curio_coef)

        bd = RewardBreakdown(
            correctness=correctness,
            retrieval_quality=retrieval_quality,
            exec_success=exec_success,
            efficiency=efficiency,
            memory_utilization=memory_utilization,
            composite=composite,
        )

        logger.debug(
            "Reward breakdown: correctness=%.2f retrieval=%.2f exec=%.2f "
            "eff=%.2f mem=%.2f curio=%.2f → composite=%.3f",
            correctness, retrieval_quality, exec_success,
            efficiency, memory_utilization, curio_bonus, composite,
        )
        return bd

    # ── サブ報酬 ────────────────────────────────

    async def _correctness(
        self,
        prompt: str,
        response: str,
        meta: dict[str, Any],
    ) -> float:
        """Teacher LLM による正確性評価。LLM なし or 失敗時はメタデータから推定。"""
        # メタデータに既存スコアがある場合はそれを使う
        if "correctness_score" in meta:
            return float(meta["correctness_score"])

        if self._gateway is None:
            # ヒューリスティック: 回答が空でなければ 0.5
            return 0.5 if response.strip() else 0.0

        try:
            from src.llm.gateway import LLMGateway
            assert isinstance(self._gateway, LLMGateway)
            resp = await self._gateway.complete(
                f"Question: {prompt[:300]}\n\nAnswer: {response[:500]}\n\n"
                "Rate answer correctness 0.0-1.0. Reply with only a number.",
                system="You are a quality evaluator. Output only a float 0.0-1.0.",
                provider=self._provider,
                max_tokens=5,
                temperature=0.0,
            )
            import re
            m = re.search(r"([0-9]*\.?[0-9]+)", resp.content.strip())
            if m:
                return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            logger.debug("Correctness LLM eval failed; using fallback")
        return 0.5

    def _retrieval_quality(self, meta: dict[str, Any]) -> float:
        """取得文書の関連度スコア。"""
        # メタデータから FAISS スコア or 直接スコアを取得
        if "retrieval_score" in meta:
            return float(meta["retrieval_score"])
        if "context_scores" in meta:
            scores = meta["context_scores"]
            if scores:
                return float(sum(scores) / len(scores))
        # FAISSコンテキストが使われていれば中程度のスコア
        if meta.get("used_context"):
            return 0.6
        return 0.3  # コンテキストなし

    def _exec_success(self, meta: dict[str, Any]) -> float:
        """コード実行成功スコア (0 or 1)。"""
        if "exec_success" in meta:
            return 1.0 if meta["exec_success"] else 0.0
        if "execution_result" in meta:
            result = meta["execution_result"]
            if hasattr(result, "success"):
                return 1.0 if result.success else 0.0
        # コード実行なし → 中立スコア
        return 0.5

    def _efficiency(self, response: str) -> float:
        """回答効率スコア（簡潔さ・適切な長さ）。"""
        n = len(response.strip())
        if n == 0:
            return 0.0
        # 短すぎる (< 20 chars) や長すぎる (> max_chars) にペナルティ
        if n < 20:
            return 0.3
        if n > self._max_response_chars:
            # 超過分に比例してペナルティ
            excess_ratio = n / self._max_response_chars
            return max(0.1, 1.0 - (excess_ratio - 1.0) * 0.3)
        return 1.0

    def _curio_information_gain(self, meta: dict[str, Any]) -> float:
        """CURIO 情報利得報酬 (arXiv:2504.03206)。

        新規取得ドキュメント数を情報利得の代理指標として使用する。
        metadata['new_doc_ids'] = このターンで初めて参照したドキュメントIDリスト
        metadata['seen_doc_ids'] = 過去ターンで参照済みのドキュメントIDセット

        スコアの意味:
            0件の新規参照 → 0.0（新しい情報を得ていない）
            1〜3件の新規参照 → 0.5〜1.0（適度な情報利得）
            4件以上 → 0.8（多すぎると散漫になる可能性があるためキャップ）
        """
        if "curio_score" in meta:
            return float(meta["curio_score"])

        new_docs = meta.get("new_doc_ids", [])
        seen_docs = set(meta.get("seen_doc_ids", []))
        all_docs = meta.get("context_doc_ids", [])

        if new_docs:
            # 明示的な新規ドキュメントリストがある場合
            n_new = len(new_docs)
        elif all_docs and seen_docs:
            # 差分で新規を計算
            n_new = sum(1 for d in all_docs if d not in seen_docs)
        else:
            # 情報なし → 中立スコア
            return 0.3

        if n_new == 0:
            return 0.0
        elif n_new == 1:
            return 0.5
        elif n_new <= 3:
            return 1.0
        else:
            return 0.8  # 過剰取得にはキャップ

    def _memory_utilization(self, meta: dict[str, Any]) -> float:
        """FAISSメモリ活用スコア。"""
        if "memory_utilization" in meta:
            return float(meta["memory_utilization"])
        if "context_doc_ids" in meta:
            used = len(meta["context_doc_ids"])
            # 1〜3 件使用が理想
            if 1 <= used <= 3:
                return 1.0
            elif used == 0:
                return 0.2
            else:
                return max(0.5, 1.0 - (used - 3) * 0.1)
        return 0.3  # メモリ未使用

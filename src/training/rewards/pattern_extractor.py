"""src/training/rewards/pattern_extractor.py — 成功パターン抽出 → KG 自動登録

思考ログ (thought_logs) を集計し、success_rate > 0.9 のパターンを
Knowledge Graph (NetworkX / Neo4j) に Entity として自動登録する。

使い方:
    from src.training.rewards.pattern_extractor import PatternExtractor
    from src.memory.metadata_store import MetadataStore
    from src.knowledge_graph.store import KnowledgeGraphStore

    extractor = PatternExtractor()
    registered = await extractor.extract_and_register(store, kg)
    print(f"{registered} patterns registered to KG")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 成功判定の reward 閾値
_SUCCESS_REWARD_THRESHOLD = 0.8
# KG 登録を行う成功率の閾値
_REGISTRATION_SUCCESS_RATE = 0.9
# 統計的に意味あるサンプル数の最低値
_MIN_SAMPLE_COUNT = 5


class PatternExtractor:
    """thought_logs から成功パターンを抽出し KG に Entity として登録する。

    N-1 の実装範囲:
    - MetadataStore.list_patterns_above_threshold() で高成功率パターンを取得
    - KnowledgeGraphStore.add_entity() で Pattern ノードを追加
    - 既存ノードへの重複登録は entity_exists() でスキップ

    Args:
        success_threshold:      reward >= この値を「成功」とカウント。
        registration_threshold: この成功率以上のパターンを KG 登録。
        min_sample_count:       統計的意義のある最小サンプル数。
    """

    def __init__(
        self,
        success_threshold: float = _SUCCESS_REWARD_THRESHOLD,
        registration_threshold: float = _REGISTRATION_SUCCESS_RATE,
        min_sample_count: int = _MIN_SAMPLE_COUNT,
    ) -> None:
        self._success_thr = success_threshold
        self._reg_thr = registration_threshold
        self._min_count = min_sample_count

    async def extract_and_register(
        self,
        store: Any,
        kg: Any,
    ) -> int:
        """高成功率パターンを抽出して KG に登録する。

        Args:
            store: MetadataStore インスタンス。
            kg:    KnowledgeGraphStore インスタンス。

        Returns:
            新規登録したパターン数（既存スキップは含まない）。
        """
        patterns = await store.list_patterns_above_threshold(
            success_threshold=self._success_thr,
            registration_threshold=self._reg_thr,
            min_count=self._min_count,
        )

        registered = 0
        for p in patterns:
            pattern_id: str = p["pattern_id"]
            entity_name = f"pattern:{pattern_id}"

            if kg.entity_exists(entity_name):
                # 成功率・平均報酬を properties として更新
                entity = kg.get_entity(entity_name)
                if entity is not None:
                    entity.properties["success_rate"] = p["success_rate"]
                    entity.properties["mean_reward"] = p["mean_reward"]
                    entity.properties["total_count"] = p["total_count"]
                    logger.debug("Updated KG entity: %s (success_rate=%.3f)", entity_name, p["success_rate"])
                continue

            kg.add_entity(
                name=entity_name,
                entity_type="reasoning_pattern",
                properties={
                    "pattern_id": pattern_id,
                    "success_rate": p["success_rate"],
                    "mean_reward": p["mean_reward"],
                    "total_count": p["total_count"],
                },
            )
            registered += 1
            logger.info(
                "Registered KG pattern: %s (success_rate=%.3f, n=%d)",
                pattern_id,
                p["success_rate"],
                p["total_count"],
            )

        if patterns:
            logger.info(
                "PatternExtractor: %d patterns checked, %d newly registered",
                len(patterns),
                registered,
            )
        return registered

    async def run(
        self,
        store: Any,
        kg: Any,
    ) -> dict[str, Any]:
        """extract_and_register のラッパー。結果サマリを dict で返す。"""
        registered = await self.extract_and_register(store, kg)
        return {
            "registered": registered,
            "success_threshold": self._success_thr,
            "registration_threshold": self._reg_thr,
            "min_sample_count": self._min_count,
        }

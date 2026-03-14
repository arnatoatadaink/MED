"""src/orchestrator/model_router.py — Graph-aware Model Router

クエリの複雑度・KG エンティティ・ドメインに基づいて
Teacher Model または Student Model にルーティングする。

ルーティング戦略 (CLAUDE.md):
- simple  → Student (Qwen-7B + TinyLoRA)
- moderate → Student + FAISS + 外部RAG
- complex → Teacher (Claude/GPT)

KG 参照ロジック:
- クエリにエンティティが含まれる場合、KGRouterBridge で関連 doc_id を拡張
- RELATIONAL クエリは KG パスで補強

使い方:
    from src.orchestrator.model_router import ModelRouter, RoutingDecision

    router = ModelRouter(gateway, kg_bridge=kg_bridge)
    decision = await router.route(parsed_query)
    # → RoutingDecision(target="teacher", use_kg=True, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.llm.gateway import LLMGateway
from src.orchestrator.query_parser import ParsedQuery

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """ルーティング判断結果。"""

    target: str                          # "student" | "teacher"
    use_faiss: bool = True               # FAISS 検索を使用するか
    use_kg: bool = False                 # Knowledge Graph を使用するか
    use_external_rag: bool = False       # 外部 RAG (GitHub/SO/Tavily) を使用するか
    use_sandbox: bool = False            # コード実行サンドボックスを使用するか
    kg_entities: list[str] = field(default_factory=list)  # KG 検索エンティティ
    expanded_doc_ids: list[str] = field(default_factory=list)  # KG で拡張した doc_ids
    reason: str = ""                     # ルーティング理由（デバッグ用）

    @property
    def use_student(self) -> bool:
        return self.target == "student"

    @property
    def use_teacher(self) -> bool:
        return self.target == "teacher"


class ModelRouter:
    """Graph-aware Model Router。

    ParsedQuery の complexity / intent / entities に基づいてモデルを選択し、
    KGRouterBridge で doc_id を拡張する。

    Args:
        gateway: LLMGateway（Teacher として使用）。
        kg_bridge: KGRouterBridge インスタンス（省略時は KG なし）。
        student_complexity_threshold: これ以下の complexity は Student に送る。
            "simple" → Student のみ、"moderate" → Student+RAG、"complex" → Teacher
        always_use_teacher_for_code: True なら code intent は必ず Teacher。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        kg_bridge: object | None = None,
        student_complexity_threshold: str = "moderate",
        always_use_teacher_for_code: bool = False,
    ) -> None:
        self._gateway = gateway
        self._kg_bridge = kg_bridge
        self._threshold = student_complexity_threshold
        self._always_teacher_code = always_use_teacher_for_code

        _VALID = {"simple", "moderate", "complex"}
        if student_complexity_threshold not in _VALID:
            raise ValueError(
                f"student_complexity_threshold must be one of {_VALID}, "
                f"got {student_complexity_threshold!r}"
            )

    async def route(self, parsed: ParsedQuery) -> RoutingDecision:
        """ルーティング判断を返す。

        Args:
            parsed: QueryParser.parse() の結果。

        Returns:
            RoutingDecision。
        """
        target, reason = self._select_target(parsed)
        use_external = parsed.complexity in ("moderate", "complex") or parsed.requires_retrieval
        use_sandbox = parsed.requires_execution or parsed.is_code
        use_kg = bool(parsed.entities) and self._kg_bridge is not None

        # KG 拡張
        kg_entities: list[str] = []
        expanded_doc_ids: list[str] = []
        if use_kg:
            kg_entities, expanded_doc_ids = await self._expand_with_kg(parsed.entities)
            if not expanded_doc_ids:
                use_kg = False  # KG に情報がなければ無効化

        decision = RoutingDecision(
            target=target,
            use_faiss=True,
            use_kg=use_kg,
            use_external_rag=use_external,
            use_sandbox=use_sandbox,
            kg_entities=kg_entities,
            expanded_doc_ids=expanded_doc_ids,
            reason=reason,
        )

        logger.info(
            "ModelRouter: %r → target=%s use_kg=%s use_rag=%s reason=%s",
            parsed.raw[:60], target, use_kg, use_external, reason,
        )
        return decision

    def _select_target(self, parsed: ParsedQuery) -> tuple[str, str]:
        """ターゲットモデルを選択する。"""
        # Code intent 強制 Teacher
        if self._always_teacher_code and parsed.is_code:
            return "teacher", "code intent forced to teacher"

        # Complexity ベースルーティング
        if parsed.complexity == "simple":
            return "student", "simple complexity → student"
        if parsed.complexity == "complex":
            return "teacher", "complex query → teacher"

        # moderate: threshold で判定
        if self._threshold == "simple":
            return "teacher", "threshold=simple, moderate → teacher"
        # threshold == "moderate" or "complex"
        return "student", "moderate complexity → student + RAG"

    async def _expand_with_kg(
        self,
        entity_names: list[str],
    ) -> tuple[list[str], list[str]]:
        """KGRouterBridge で doc_id を拡張する。"""
        try:
            from src.knowledge_graph.router_bridge import KGRouterBridge
            assert isinstance(self._kg_bridge, KGRouterBridge)
            doc_ids = await self._kg_bridge.expand_query_with_kg(entity_names)
            related = await self._kg_bridge.get_related_entities(entity_names)
            all_entities = list(set(entity_names + related))
            return all_entities, doc_ids
        except Exception:
            logger.exception("KG expansion failed for entities=%s", entity_names)
            return entity_names, []

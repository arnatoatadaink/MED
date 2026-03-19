# plan_think.md — Teacher思考過程の抽出・保存スキーム実装計画

## 背景と目的

### 解決する問題

現在のRAGシステムは「何を知っているか（事実知識）」は蓄積できるが、
「なぜその知識が必要か（推論構造）」を保存していない。

知識には2種類が混在している：

```
① 外部化・最新化すべき事実知識
   例: APIのバージョン、ライブラリの使い方、最新論文の結果
   → FAISS + RAGで最新化済み ✅

② 外部化すると性能低下するリスクのある論理構造・判断基準
   例: 「なぜそのAPIを選ぶか」「どう問題を分解するか」「トレードオフの評価基準」
   → LLM weightsに圧縮されており、現状FAISSには入っていない ❌
```

**思考過程（ReasoningTrace）を保存することで**：
- 類似クエリへの回答品質向上（推論テンプレートとして検索可能）
- Student ModelのGRPO/SFT学習データとして活用
- 「なぜその知識が必要か」を回答に紐付けて保存

### 技術的注意点（プロンプトハックではない理由）

- Extended Thinking: Anthropicが公式提供するAPI機能（ToS準拠）
- CoTプロンプト: Wei et al. (2022) で確立した公開手法
- ただし、CoTで引き出した思考は **post-hoc rationalization** の可能性がある
  （モデルの実際の計算過程ではなく「もっともらしい説明の後付け生成」）
- Extended Thinkingのほうが実際の推論トークン列に近く信頼性が高い
- 保存データには `trace_method: "extended_thinking" | "cot_prompt"` で区別する

---

## 実装範囲（最小MVP）

### 追加・変更ファイル一覧

```
変更:
  src/llm/gateway.py                      LLMResponseにthinking_text追加、complete()にenable_thinking追加
  src/llm/providers/anthropic.py          Extended Thinking API対応
  src/memory/schema.py                    ReasoningTrace / KnowledgeType 追加
  src/memory/metadata_store.py            reasoning_traces / trace_documents テーブル追加
  src/memory/memory_manager.py            save_reasoning_trace() メソッド追加

新規:
  src/llm/prompt_templates/reasoning_extraction.yaml   CoT抽出プロンプト（非Anthropic用）
  src/llm/thinking_extractor.py           ThinkingExtractor クラス（プロバイダ別抽出ロジック統合）

任意（後続フェーズ）:
  src/orchestrator/pipeline.py            Teacher呼び出し後の自動保存フック
  src/gui/tabs/chat.py                    デバッグパネルへのthinking表示
```

---

## Step 1: `LLMResponse` と `BaseLLMProvider` の拡張

### `src/llm/gateway.py` 変更箇所

```python
@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: Any | None = None
    # ── 追加 ──
    thinking_text: str | None = None   # Extended Thinking / CoT で得た思考テキスト
    thinking_tokens: int = 0           # thinking block のトークン数（Anthropic用）
```

`BaseLLMProvider.complete()` シグネチャに `enable_thinking: bool = False` を追加。
他プロバイダ（OpenAI/Ollama）はデフォルト `False` のまま変更不要。

---

## Step 2: `AnthropicProvider` のExtended Thinking対応

### `src/llm/providers/anthropic.py` 変更箇所

```python
async def complete(
    self,
    messages: list[LLMMessage],
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    enable_thinking: bool = False,        # ← 追加
    thinking_budget_tokens: int = 8000,   # ← 追加
) -> LLMResponse:
    ...
    if enable_thinking:
        # Extended Thinking 有効時: temperature は 1.0 固定（API要件）
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
        kwargs.pop("temperature", None)
    ...
    # レスポンスパース
    thinking_text = None
    thinking_tokens = 0
    content_text = ""
    for block in raw.content:
        if block.type == "thinking":
            thinking_text = block.thinking
            thinking_tokens = getattr(block, "thinking_tokens", 0)  # SDK版依存
        elif block.type == "text":
            content_text = block.text

    return LLMResponse(
        content=content_text,
        thinking_text=thinking_text,
        thinking_tokens=thinking_tokens,
        ...
    )
```

**注意**: Extended Thinking は `temperature=1.0` 固定が API 要件。
モデルは `claude-opus-4-6` 以上が必要（haiku は非対応）。

---

## Step 3: `ReasoningTrace` スキーマ追加

### `src/memory/schema.py` 追加クラス

```python
class KnowledgeType(str, Enum):
    """思考過程の中で識別された知識の種別。"""
    FACTUAL     = "factual"      # 陳腐化する事実知識 → FAISS更新対象
    STRUCTURAL  = "structural"   # 推論パターン・判断基準 → weightに焼くか思考として保存
    PROCEDURAL  = "procedural"   # 手順・アルゴリズム
    CAUSAL      = "causal"       # 因果関係
    ANALOGICAL  = "analogical"   # 類推・アナロジー

class TraceMethod(str, Enum):
    """思考過程の抽出方法。"""
    EXTENDED_THINKING = "extended_thinking"  # Anthropic Extended Thinking API
    COT_PROMPT        = "cot_prompt"         # CoTプロンプトで引き出した（post-hoc rationalizationの可能性あり）
    MANUAL            = "manual"             # 人手アノテーション

class ReasoningTrace(BaseModel):
    """Teacher LLM の思考過程を保存するエンティティ。"""

    id: str = Field(default_factory=_generate_doc_id)

    # 元のクエリと回答
    query: str
    answer: str

    # 思考過程テキスト（生）
    raw_thinking: str | None = None        # Extended Thinking の生テキスト
    trace_method: TraceMethod = TraceMethod.COT_PROMPT

    # 構造化された思考要素（CoTプロンプトで抽出 or 生テキストをパース）
    knowledge_audit: list[dict] = Field(default_factory=list)
    # 例: [{"item": "FAISS IVFIndex仕様", "kind": "factual", "needs_retrieval": True, "confidence": "高"}]

    reasoning_chain: list[str] = Field(default_factory=list)
    # 例: ["問題をXとYに分解", "XはFAISSで検索", "YはKGで解決", "統合して回答"]

    judgment_criteria: list[str] = Field(default_factory=list)
    # 例: ["データ量10万件超ならIVFが適切", "鮮度重視なら外部RAGを優先"]

    retrieval_rationale: list[dict] = Field(default_factory=list)
    # 例: [{"doc_id": "xxx", "why": "APIの最新仕様が含まれるため", "trust": "高"}]

    # 分類
    primary_knowledge_type: KnowledgeType = KnowledgeType.FACTUAL

    # Teacher情報
    teacher_model: str | None = None
    teacher_provider: str | None = None
    thinking_tokens: int = 0    # Extended Thinking消費トークン（コスト管理用）

    # 品質
    confidence: float = 0.5
    doc_ids: list[str] = Field(default_factory=list)  # 参照したDocumentのID群

    created_at: datetime = Field(default_factory=datetime.utcnow)
```

---

## Step 4: SQLiteテーブル追加

### `src/memory/metadata_store.py` 追加DDL

```sql
-- 思考過程メインテーブル
CREATE TABLE IF NOT EXISTS reasoning_traces (
    id                   TEXT PRIMARY KEY,
    query                TEXT NOT NULL,
    answer               TEXT NOT NULL,
    raw_thinking         TEXT,             -- Extended Thinking の生テキスト
    trace_method         TEXT DEFAULT 'cot_prompt',
    knowledge_audit      TEXT,             -- JSON配列
    reasoning_chain      TEXT,             -- JSON配列
    judgment_criteria    TEXT,             -- JSON配列
    retrieval_rationale  TEXT,             -- JSON配列
    primary_knowledge_type TEXT DEFAULT 'factual',
    teacher_model        TEXT,
    teacher_provider     TEXT,
    thinking_tokens      INTEGER DEFAULT 0,
    confidence           REAL DEFAULT 0.5,
    created_at           TEXT DEFAULT (datetime('now'))
);

-- トレースと参照ドキュメントの多対多
CREATE TABLE IF NOT EXISTS trace_documents (
    trace_id  TEXT NOT NULL,
    doc_id    TEXT NOT NULL,
    role      TEXT DEFAULT 'supporting',  -- "primary" | "supporting" | "contradicting"
    PRIMARY KEY (trace_id, doc_id)
);

-- クエリ全文検索用インデックス
CREATE INDEX IF NOT EXISTS idx_reasoning_traces_created
    ON reasoning_traces(created_at DESC);
```

---

## Step 5: `ThinkingExtractor` 新規クラス

### `src/llm/thinking_extractor.py`

思考過程抽出をプロバイダ別に統合するファサード。

```python
class ThinkingExtractor:
    """思考過程の抽出を担当する。

    Anthropic: Extended Thinking API を呼び出し ThinkingBlock を取得
    他プロバイダ: CoTプロンプト（reasoning_extraction.yaml）で構造化思考を引き出す
    """

    async def extract(
        self,
        query: str,
        context_docs: list[SearchResult],
        gateway: LLMGateway,
        provider: str | None = None,
        *,
        enable_extended_thinking: bool = True,
        thinking_budget_tokens: int = 8000,
    ) -> tuple[LLMResponse, ReasoningTrace]:
        """クエリに対してTeacher呼び出しを行い、回答と思考過程を返す。"""
        ...
```

**CoT抽出プロンプト（Anthropic以外のプロバイダ用）**:

```yaml
# src/llm/prompt_templates/reasoning_extraction.yaml

system: |
  あなたは回答前に必ず以下の形式で思考を外部化してください。
  ※ これはモデルの説明能力を活用したものであり、実際の内部計算とは異なる場合があります。

  <knowledge_audit>
  この質問に答えるために必要な知識を列挙する。
  各知識について以下を記載:
    - item: 知識の内容
    - kind: factual（陳腐化する）/ structural（推論パターン）/ procedural（手順）/ causal（因果）
    - needs_retrieval: true/false
    - confidence: 高/中/低
  </knowledge_audit>

  <reasoning_chain>
  問題を解く推論ステップを1ステップ1文で記述
  </reasoning_chain>

  <judgment_criteria>
  この判断で適用した基準・原則（他のクエリでも再利用可能な形で記述）
  </judgment_criteria>

  <retrieval_rationale>
  参照した各ドキュメントについて:
    - doc_id: ドキュメントID
    - why: なぜこのドキュメントが関連するか
    - trust: 高/中/低
  </retrieval_rationale>

user_template: |
  以下のコンテキストを参照して質問に答えてください。

  【参照ドキュメント】
  {context}

  【質問】
  {query}
```

---

## Step 6: `MemoryManager.save_reasoning_trace()` 追加

### `src/memory/memory_manager.py` 追加メソッド

```python
async def save_reasoning_trace(self, trace: ReasoningTrace) -> str:
    """ReasoningTrace を SQLite に保存し、trace.id を返す。

    FAISS への保存は行わない（構造化クエリで検索するため）。
    reasoning_chainやjudgment_criteriaをDocument化してFAISSに入れたい場合は
    別途 add_from_text() を呼ぶこと。
    """
    await self.store.save_reasoning_trace(trace)
    # trace_documents の多対多も保存
    for doc_id in trace.doc_ids:
        await self.store.save_trace_document(trace.id, doc_id)
    return trace.id
```

---

## FAISSへの格納方針（任意・後続フェーズ）

思考過程をFAISSで検索可能にする場合の方針：

```
方法A: reasoning_chain を1ドキュメントとして保存
  source_type = TEACHER
  domain      = "general"
  content     = "クエリ: Xの場合 → ステップ1: ... ステップ2: ..."
  → 類似クエリで「推論テンプレート」として検索される

方法B: judgment_criteria の structural 知識のみ抽出して別ドメインで保存
  domain  = "reasoning"（新設）
  content = "データ量10万件超の場合はIVFIndexが適切。理由: ..."
  → 事実知識と推論パターンを検索時に区別可能
```

MVP では方法A・Bともに実装せず、SQLite保存のみとする。
学習データとして使う際に `reasoning_traces` テーブルから取得する。

---

## テスト計画

```
tests/unit/test_thinking_extractor.py
  - Extended Thinking レスポンスのパース
  - CoTプロンプト応答のパース
  - ReasoningTrace dataclass の検証

tests/unit/test_reasoning_store.py
  - reasoning_traces テーブルへの保存・取得
  - trace_documents の多対多
  - JSON フィールドのシリアライズ・デシリアライズ
```

---

## 実装順序

```
1. schema.py          — ReasoningTrace / KnowledgeType / TraceMethod 追加
2. gateway.py         — LLMResponse.thinking_text / enable_thinking パラメータ追加
3. anthropic.py       — Extended Thinking API 呼び出し実装
4. metadata_store.py  — reasoning_traces / trace_documents テーブル追加
5. memory_manager.py  — save_reasoning_trace() 追加
6. prompt_templates/  — reasoning_extraction.yaml 作成
7. thinking_extractor.py — ThinkingExtractor クラス新規作成
8. tests/             — 単体テスト追加
```

---

## 非対応事項（スコープ外）

- `pipeline.py` への自動保存フック（後続フェーズ）
- GUIでのthinking表示（後続フェーズ）
- FAISSへのreasoningドメイン追加（後続フェーズ）
- OpenAI o1/o3系のreasoning_content対応（API仕様確認後）
- ローカルモデル（Ollama）のCoT品質評価

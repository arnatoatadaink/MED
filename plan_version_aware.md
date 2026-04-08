# plan_version_aware.md — バージョン対応知識管理

## 背景と課題

プログラミング知識はバージョンによって変化が激しく、LLMの学習データに含まれる古い情報がコード生成の誤りに直結する。
MEDは外部RAGでリアルタイムな知識を補完できるが、**同一トピックで複数バージョンの知識が混在すると検索精度が低下する**。

例:
- `pydantic.validator` → v1では正しい、v2では廃止（`field_validator`が正解）
- `asyncio.get_event_loop()` → Python 3.10以降は非推奨
- `torch.jit.script()` → PyTorch 2.0でcompile()に役割が移行

---

## 設計方針

**KGのバージョンノードとFAISSメタデータの組み合わせ**で対応する。
バージョン知識をKGに構造化し、FAISSメタデータでフィルタリング可能にする。

```
クエリ: "pydantic validator Python 3.12"
    ↓
QueryParser: tech="pydantic", target_version="2.x"
    ↓
KG検索: pydantic → version_nodes → 2.x → 関連doc_ids
    ↓
FAISS: metadata filter (version_min <= 2 AND version_status != "removed")
    ↓
結果: v2対応ドキュメントのみ返却
```

---

## Schema 変更

### documents テーブルへの追加カラム

```sql
ALTER TABLE documents ADD COLUMN tech_name TEXT;          -- "python", "pydantic", "react"
ALTER TABLE documents ADD COLUMN version_introduced TEXT; -- "3.10", "2.0", null(unknown)
ALTER TABLE documents ADD COLUMN version_deprecated TEXT; -- "3.12", null
ALTER TABLE documents ADD COLUMN version_removed TEXT;    -- null, "3.13"
ALTER TABLE documents ADD COLUMN is_changelog BOOLEAN DEFAULT 0;
ALTER TABLE documents ADD COLUMN version_status TEXT      -- "current", "deprecated", "removed", "unknown"
    DEFAULT 'unknown';
```

### Document dataclass への追加（schema.py）

```python
@dataclass
class Document:
    # ... 既存フィールド ...
    tech_name: str | None = None
    version_introduced: str | None = None
    version_deprecated: str | None = None
    version_removed: str | None = None
    is_changelog: bool = False
    version_status: str = "unknown"  # current | deprecated | removed | unknown
```

---

## Knowledge Graph バージョンノード設計

```
Entity: ("Python", type="language")
  → has_version → Entity("3.11", type="version", tech="python")
  → has_version → Entity("3.12", type="version", tech="python")
  → has_version → Entity("3.13", type="version", tech="python")

Entity: ("asyncio.TaskGroup", type="feature")
  → introduced_in → Entity("3.11", type="version")
  → related_to   → Entity("Python", type="language")

Entity: ("distutils", type="module")
  → deprecated_in → Entity("3.10", type="version")
  → removed_in    → Entity("3.12", type="version")

Entity: ("pydantic.validator", type="api")
  → deprecated_in → Entity("2.0", type="version", tech="pydantic")
  → replaced_by   → Entity("pydantic.field_validator", type="api")
```

### KG relation types（version専用）

| relation | 意味 |
|---------|------|
| `introduced_in` | このバージョンで追加 |
| `deprecated_in` | このバージョンで非推奨化 |
| `removed_in` | このバージョンで削除 |
| `replaced_by` | 代替APIへのポインタ |
| `changed_in` | 動作が変わったバージョン |
| `has_version` | technology → versionノードの親子関係 |

---

## バージョン対応検索フロー

### 1. バージョン検出（QueryParser拡張）

クエリまたはプロジェクトコンテキストからターゲットバージョンを検出：

```python
# 例: "pydantic v2でvalidatorを使う方法"
ParsedQuery(
    original="pydantic v2でvalidatorを使う方法",
    entities=["pydantic", "validator"],
    tech_versions={"pydantic": "2.x"},  # ← 新規フィールド
    intent="how_to"
)
```

バージョン検出優先順:
1. クエリ中の明示的バージョン指定（"v2", "3.12", "latest"）
2. プロジェクト設定ファイル（`pyproject.toml`, `package.json`）からの推定（将来）
3. デフォルト: `version_status = "current"` のドキュメントのみ

### 2. FAISS メタデータフィルタリング

```python
# metadata_store.py の検索メソッドに version フィルタを追加
async def search_with_version(
    self,
    doc_ids: list[str],
    target_version: str | None,
    tech: str | None,
) -> list[Document]:
    if target_version is None:
        # デフォルト: current のみ（deprecated/removed を除外）
        filter_sql = "AND version_status != 'removed'"
    else:
        # バージョン範囲フィルタ
        filter_sql = """
            AND (version_introduced IS NULL OR version_introduced <= ?)
            AND (version_removed IS NULL OR version_removed > ?)
        """
```

### 3. バージョン競合の検出と表示

同一エンティティについて複数バージョンのドキュメントが返った場合：

```python
# 例: pydantic.validator の v1 ドキュメントと v2 ドキュメントが混在
# → バージョンタグを付けて両方提示
[
  {"content": "Use @validator ...", "version": "v1 (deprecated in v2)"},
  {"content": "Use @field_validator ...", "version": "v2+ (current)"},
]
```

---

## リリースノート専用インデクサ

### 収集戦略

```python
# changelog専用質問パターン
CHANGELOG_QUERIES = [
    "Python {version} what's new release notes breaking changes",
    "Pydantic {version} migration guide breaking changes",
    "PyTorch {version} release notes deprecated removed APIs",
    # ...
]

# is_changelog=True でタグ付けして格納
# version_introduced = "3.12" (このchangelogが対象とするバージョン)
```

### 取得先

| 技術 | URL パターン |
|-----|------------|
| Python | `docs.python.org/3/whatsnew/3.{N}.html` |
| Pydantic | `docs.pydantic.dev/latest/migration/` |
| PyTorch | `pytorch.org/docs/stable/notes/releasenotes.html` |
| FastAPI | `fastapi.tiangolo.com/release-notes/` |
| React | `react.dev/blog` (major releases) |
| numpy | `numpy.org/doc/stable/release/` |

---

## 実装ロードマップ

### Step 1: Schema 追加（最小変更）
- `src/memory/schema.py` に version フィールド追加
- `src/memory/metadata_store.py` にカラム追加マイグレーション
- 既存ドキュメントは `version_status="unknown"` でデフォルト埋め

### Step 2: Seed 収集
- `questions.txt` にリリースノート質問追加（✅ 完了）
- `url_fetcher.py` で changelog URL を直接フェッチ
- `is_changelog=True` タグ付きで mature

### Step 3: KG バージョンノード
- `knowledge_graph/extractor.py` に version entity 抽出ロジック追加
- EntityExtractor がリリースノートから `introduced_in/deprecated_in/removed_in` を抽出

### Step 4: バージョン対応検索
- `orchestrator/query_parser.py` にバージョン検出追加
- `memory/metadata_store.py` に version フィルタ付き検索追加
- `orchestrator/model_router.py` でバージョンコンテキストを伝播

### Step 5: 競合検出（将来）
- 同一トピックで複数バージョンが返った場合の表示ロジック
- GUI でバージョンタグ付き結果の可視化

---

## 優先度・状態

| ステップ | 優先度 | 状態 | 依存 |
|---------|--------|------|------|
| Step 1: Schema 追加 | 高 | ⬜ | なし |
| Step 2: リリースノート Seed | 高 | ⬜ | questions.txt 追加済み ✅ |
| Step 3: KG バージョンノード | 中 | ⬜ | Step 1 |
| Step 4: バージョン対応検索 | 中 | ⬜ | Step 1, 3 |
| Step 5: 競合検出 GUI | 低 | ⬜ | Step 4 |

**Step 1 は既存データを壊さない（デフォルト値付きカラム追加のみ）のですぐに実施可能。**

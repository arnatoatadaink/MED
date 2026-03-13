# テストガイド

## テスト実行

```bash
# 全テスト
pytest tests/

# 特定モジュール
pytest tests/unit/test_memory_manager.py -v

# カバレッジレポート付き
pytest tests/ --cov=src --cov-report=html
```

## テスト構成

```
tests/
├── unit/              単体テスト (モック使用)
│   ├── test_config.py          ✅ 53 テスト
│   ├── test_schema.py
│   ├── test_embedder.py
│   ├── test_faiss_index.py
│   ├── test_metadata_store.py
│   ├── test_memory_manager.py
│   ├── test_scoring.py
│   ├── test_ltr_ranker.py
│   ├── test_iterative_retrieval.py
│   ├── test_llm_gateway.py
│   ├── test_llm_generators.py
│   ├── test_rag.py
│   ├── test_knowledge_graph.py
│   ├── test_maturation.py
│   ├── test_orchestrator.py
│   ├── test_sandbox.py
│   ├── test_training.py
│   ├── test_phase2.py          Phase 2 TeacherRegistry / CompositeScorer
│   ├── test_teacher_provenance_step1〜5.py
│   └── ...
└── integration/       統合テスト (実 Docker / 実 DB)
    └── (実装予定)
```

## 単体テストの書き方

```python
# tests/unit/test_my_module.py
import pytest
from unittest.mock import AsyncMock, patch
from src.my_module import MyClass

@pytest.fixture
def my_instance():
    return MyClass(config={"key": "value"})

@pytest.mark.asyncio
async def test_basic_functionality(my_instance):
    result = await my_instance.do_something("input")
    assert result.status == "ok"

@pytest.mark.asyncio
async def test_with_mock():
    with patch("src.my_module.external_call", new_callable=AsyncMock) as mock:
        mock.return_value = {"data": "mocked"}
        result = await MyClass().call_external()
    assert result["data"] == "mocked"
```

## Phase 2 テスト

Teacher 信頼度評価の主要テスト:

```bash
pytest tests/unit/test_teacher_provenance_step1.py  # teacher_id 標準化
pytest tests/unit/test_teacher_provenance_step2.py  # TeacherRegistry EWMA
pytest tests/unit/test_teacher_provenance_step3.py  # MetadataStore teacher_id
pytest tests/unit/test_teacher_provenance_step4.py  # CompositeScorer 乗算
pytest tests/unit/test_teacher_provenance_step5.py  # TeacherFeedbackPipeline
pytest tests/unit/test_phase2.py                    # 統合確認
```

## CI チェックリスト

PR 前に以下を確認してください:

- [ ] `pytest tests/unit/ -v` が全て PASS
- [ ] `python -m py_compile src/**/*.py` にエラーなし
- [ ] 新機能には対応するテストファイルを追加
- [ ] `mkdocs build --strict` でドキュメントビルドが通る

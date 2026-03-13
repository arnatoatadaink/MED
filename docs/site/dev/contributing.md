# コントリビューションガイド

## 開発環境のセットアップ

```bash
git clone https://github.com/your-org/MED.git
cd MED
pip install -e ".[dev]"
```

## ブランチ戦略

```
main          本番稼働コード
  └── feature/xxx   機能追加
  └── fix/xxx       バグ修正
  └── claude/xxx    Claude Code セッション
```

## コーディング規約

- **型アノテーション**: 全関数に型ヒントを付ける
- **抽象 IF ファースト**: `base.py` にインターフェースを定義してから実装
- **非同期**: I/O 処理は `async/await` を使用
- **テスト**: 各モジュールに対応する `test_*.py` を作成

## 新しいアルゴリズムの追加

`src/training/algorithms/` にファイルを追加し、Registry に登録します:

```python
# src/training/algorithms/my_algo.py
from src.training.base import TrainingAlgorithm
from src.training.registry import register_algorithm

@register_algorithm("my_algo")
class MyAlgorithm(TrainingAlgorithm):
    async def train_step(self, batch, model, optimizer) -> dict:
        ...
```

設定ファイルで選択できるようになります:

```yaml
# configs/training.yaml
algorithm: "my_algo"
```

## 新しいプロバイダーの追加

```python
# src/llm/providers/my_provider.py
from src.llm.gateway import LLMProvider

class MyProvider(LLMProvider):
    async def generate(self, messages, **kwargs) -> str:
        ...
```

`src/llm/gateway.py` の `_PROVIDER_MAP` に登録します。

## ドキュメントの追加・更新

```bash
# ドキュメントを編集
vim docs/site/features/my-feature.md

# ローカルでプレビュー
mkdocs serve

# ビルド確認
mkdocs build --strict
```

## プルリクエスト

1. フィーチャーブランチを作成
2. 変更を実装してテストを追加
3. `pytest tests/` が通ることを確認
4. PR を作成してレビューを依頼

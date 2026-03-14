# sandbox_policy.yaml リファレンス

`configs/sandbox_policy.yaml` の全フィールドの説明です。

## 設定例

```yaml
# 実行環境
runtime:
  timeout_seconds: 30
  memory_limit_mb: 256
  cpu_shares: 1024        # Docker CPU シェア (1024 = 1コア)
  network: "none"         # none / bridge

# ファイルシステム
filesystem:
  readonly: true
  writable_dirs:
    - "/tmp"

# 対応言語
languages:
  python:
    image: "python:3.11-slim"
    enabled: true
  javascript:
    image: "node:20-slim"
    enabled: true
  bash:
    image: "bash:5"
    enabled: true

# セキュリティ
security:
  # 禁止するコードパターン (正規表現)
  blocked_patterns:
    - "rm\\s+-rf\\s+/"
    - ":(\\s*\\(\\s*\\)){2}"  # fork bomb
    - "import\\s+subprocess"
    - "os\\.system"
    - "__import__\\(.*subprocess"

  # 禁止する Python モジュール
  blocked_imports:
    - subprocess
    - socket
    - ctypes
    - multiprocessing

# リトライ設定
retry:
  max_attempts: 3
  auto_fix: true          # LLM による自動修正を有効化
  fix_model: "haiku"      # 修正に使用するモデル
```

## セキュリティ設計

Docker コンテナを以下のオプションで起動します:

```bash
docker run \
  --network none \           # ネットワーク無効
  --read-only \              # FS 読み取り専用
  --tmpfs /tmp \             # /tmp は書き込み可能
  --memory 256m \            # メモリ制限
  --cpus 1.0 \               # CPU 制限
  --security-opt no-new-privileges \
  --user nobody \            # 非 root 実行
  python:3.11-slim
```

## 実行フロー

```
コードを受信
    ↓
blocked_patterns で静的チェック
    ↓
Docker コンテナを起動
    ↓
タイムアウト監視下でコードを実行
    ↓
stdout / stderr / exit_code を取得
    ↓
コンテナを削除
    ↓
結果を返す
```

## リトライとオートフィックス

`auto_fix: true` の場合、実行失敗時に以下の処理が走ります:

1. `ErrorAnalyzer` がエラーメッセージを LLM で解析
2. 修正コードを生成（`fix_model` で指定したモデルを使用）
3. 修正コードを再実行
4. 最大 `max_attempts` 回まで繰り返す

!!! warning "auto_fix は信頼できる入力に限定"
    自動修正は外部からの入力ではなく、LLM が生成したコードにのみ適用されます。

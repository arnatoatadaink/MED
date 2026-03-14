# サンドボックス実行

Docker コンテナ内でコードを安全に実行する機能です。

## セキュリティポリシー

| 制限項目 | 設定値 |
|---------|--------|
| ネットワーク | 無効（`--network none`）|
| ファイルシステム | 読み取り専用（書き込みは `/tmp` のみ）|
| CPU | 1 コア |
| メモリ | 256 MB |
| タイムアウト | 30 秒 |
| 禁止パターン | `rm -rf /`, fork bomb, `os.system`, `subprocess` 等 |

ポリシーの詳細は `configs/sandbox_policy.yaml` で管理されます。

## 対応言語

- Python 3.11
- JavaScript (Node.js)
- Bash

## GUI での実行

1. **⚙️ サンドボックス** タブを開く
2. コード入力エリアにコードを貼り付ける
3. 言語を選択
4. 「**実行**」ボタンを押す
5. 標準出力・エラー出力・実行時間が表示される

## API での実行

```bash
curl -X POST http://localhost:8000/sandbox/execute \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(sum(range(100)))",
    "language": "python",
    "timeout": 30
  }'
```

レスポンス:

```json
{
  "stdout": "4950\n",
  "stderr": "",
  "exit_code": 0,
  "execution_time_ms": 127,
  "success": true
}
```

## 自動リトライ（コード修正）

実行が失敗した場合、LLM が自動でコードを修正して再試行します:

```
実行失敗
    ↓
Error Analyzer (LLM) がエラーを解析
    ↓
修正済みコードを生成
    ↓
ユーザーに確認（GUIの場合）
    ↓
修正版を再実行（最大 3 回）
```

自動修正は `src/sandbox/retry_handler.py` が管理します。

## チャットとの連携

チャットタブでコード生成を依頼すると、内部でサンドボックスが自動実行されます。

```
ユーザー: "Python でフィボナッチ数列を実装して"
    ↓
Teacher/Student がコードを生成
    ↓
Sandbox で自動実行
    ↓
成功した場合のみ回答として返す  ← 動作保証済みのコードを提供
```

!!! note "Docker が起動していない場合"
    サンドボックス機能は利用できませんが、チャット・メモリ機能は引き続き使えます。

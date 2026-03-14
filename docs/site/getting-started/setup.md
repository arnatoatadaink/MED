# セットアップ

## 前提条件

| 必要なもの | バージョン | 用途 |
|-----------|-----------|------|
| Python | 3.11+ | メインランタイム |
| Docker | 24+ | Sandbox 実行 |
| (オプション) Ollama | 最新 | ローカル LLM |

## インストール

```bash
git clone https://github.com/your-org/MED.git
cd MED
pip install -e ".[dev]"
```

## プロバイダー設定

### GUI から設定する（推奨）

1. GUI を起動して **🔧 設定 → プロバイダー設定 → プリセット** タブを開く
2. 使用するプロバイダーを選んで「**適用**」ボタンを押す
3. **🔧 設定 → APIキー** タブでキーを入力

### 設定ファイルを直接編集する

`configs/llm_config.yaml` の `primary_provider` を変更します:

```yaml
primary_provider: "anthropic"   # anthropic / openai / ollama / vllm / azure_openai / together

providers:
  anthropic:
    default_model: "claude-sonnet-4-20250514"
    haiku_model: "claude-haiku-4-5-20251001"
```

### 利用可能なプロバイダー

| プリセット | 必要な環境変数 | 月コスト目安 |
|-----------|--------------|------------|
| **Anthropic Claude** *(推奨)* | `ANTHROPIC_API_KEY` | ~$1–10 |
| **OpenAI GPT-4o** | `OPENAI_API_KEY` | ~$1–10 |
| **Ollama (ローカル)** | 不要 | 無料 |
| **vLLM (ローカル)** | 不要 | 無料 |
| **Together.ai** | `TOGETHER_API_KEY` | ~$1–5 |
| **Azure OpenAI** | `AZURE_OPENAI_API_KEY` | 従量制 |

## APIキーの永続化

セッションを超えてキーを保持するには `.env` ファイルを使います:

```bash
# .env (プロジェクトルートに作成)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...
```

!!! warning ".env をコミットしないでください"
    `.gitignore` に `.env` が含まれていることを確認してください。

## ローカルプロバイダーの準備

### Ollama

```bash
# Ollama をインストール後
ollama serve                    # バックグラウンドで起動
ollama pull llama3.1:8b        # モデルを取得
```

### vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8001
```

## オーケストレーター起動

```bash
uvicorn src.orchestrator.server:app --reload --port 8000
```

起動すると GUI のバナーが **接続中** に変わります。

## GUI 起動

```bash
python scripts/launch_gui.py
# または
python -m gradio src/gui/app.py
```

ブラウザで `http://localhost:7860` を開きます。

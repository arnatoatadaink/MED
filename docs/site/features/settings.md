# 設定・プロバイダー

GUI の **🔧 設定** タブで以下の設定を管理できます。

## タブ構成

| タブ | 内容 |
|-----|------|
| 🎨 テーマ | UI カラーテーマの選択 |
| 🔌 プロバイダー設定 | LLM プロバイダーのプリセット・カスタム設定 |
| 🔑 APIキー | 各プロバイダーの API キー入力 |
| 📝 YAML設定 | 設定ファイルの直接編集 |
| ℹ️ システム情報 | 現在の設定状態確認 |

---

## テーマ

5 種類のカラーテーマを切り替えられます:

| テーマ | 特徴 |
|-------|------|
| **MED Dark** *(デフォルト)* | ディープブルー系ダーク |
| **Soft Light** | ライトブルー系ライト |
| **Ocean Dark** | 深海ブルーダーク |
| **Monochrome** | グレースケールダーク |
| **Forest** | ディープグリーンダーク |

「ブラウザのカラーモードを自動検出」を ON にすると OS のダーク/ライト設定に追従します。

---

## プロバイダー設定

### プリセット

6 種類のプリセットから選んで「**適用**」ボタンを押すと `configs/llm_config.yaml` が書き換わります。

| プリセット | 必要な環境変数 | 備考 |
|-----------|--------------|------|
| **Anthropic Claude** *(推奨)* | `ANTHROPIC_API_KEY` | Sonnet / Haiku |
| **OpenAI GPT-4o** | `OPENAI_API_KEY` | GPT-4o / GPT-4o-mini |
| **Ollama (ローカル)** | 不要 | `ollama serve` が必要 |
| **vLLM (ローカル)** | 不要 | vLLM サーバーが必要 |
| **Together.ai** | `TOGETHER_API_KEY` | Llama / Mistral 等 |
| **Azure OpenAI** | `AZURE_OPENAI_API_KEY` | Azure エンドポイント必要 |

### カスタムプロバイダー

OpenAI 互換 API であれば任意のプロバイダーを追加できます:

| フィールド | 例 |
|-----------|-----|
| プロバイダー名 | `my_local_llm` |
| タイプ | `openai_compatible` |
| ベース URL | `http://localhost:11434/v1` |
| デフォルトモデル | `llama3.1:8b` |
| API Key 環境変数名 | `MY_LLM_API_KEY`（任意）|

追加後は「プリセット: カスタム」として選択できます。

---

## APIキー

各プロバイダーの API キーをセッション中に設定します。

!!! warning "永続化には .env ファイルを使用"
    GUI で入力したキーはセッション終了で消えます。
    永続化するには `.env` ファイルに記載してください:
    ```
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    ```

---

## YAML設定

`configs/llm_config.yaml` の内容をブラウザ上で直接編集できます。

変更後「**保存**」ボタンを押すとファイルに書き込まれます。

??? example "llm_config.yaml の例"
    ```yaml
    primary_provider: "anthropic"

    providers:
      anthropic:
        default_model: "claude-sonnet-4-20250514"
        haiku_model: "claude-haiku-4-5-20251001"
      openai:
        default_model: "gpt-4o"
        mini_model: "gpt-4o-mini"

    task_routing:
      query_parsing: "haiku"
      retrieval_verification: "haiku"
      response_generation: "sonnet"
      code_generation: "sonnet"
      verification: "sonnet"

    budget:
      daily_limit_usd: 10.0
      alert_threshold: 0.8
      fallback_to_local: true
    ```

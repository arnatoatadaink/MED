# llm_config.yaml リファレンス

`configs/llm_config.yaml` の全フィールドの説明です。

## 最小構成例

```yaml
primary_provider: "anthropic"

providers:
  anthropic:
    default_model: "claude-sonnet-4-20250514"
```

## 完全な設定例

```yaml
# 主要プロバイダー (必須)
primary_provider: "anthropic"

# プロバイダー設定
providers:
  anthropic:
    default_model: "claude-sonnet-4-20250514"
    haiku_model: "claude-haiku-4-5-20251001"

  openai:
    default_model: "gpt-4o"
    mini_model: "gpt-4o-mini"

  ollama:
    base_url: "http://localhost:11434"
    default_model: "llama3.1:8b"

  vllm:
    base_url: "http://localhost:8001/v1"
    default_model: "Qwen/Qwen2.5-7B-Instruct"

  azure_openai:
    endpoint: "https://your-resource.openai.azure.com"
    api_version: "2024-02-01"
    deployment_name: "gpt-4o"

  together:
    base_url: "https://api.together.xyz/v1"
    default_model: "meta-llama/Llama-3.1-8B-Instruct-Turbo"

# タスク別モデルルーティング
task_routing:
  query_parsing: "haiku"          # 軽量タスク → Haiku
  retrieval_verification: "haiku"
  response_generation: "sonnet"   # 本番タスク → Sonnet
  code_generation: "sonnet"
  verification: "sonnet"
  complex_reasoning: "opus"       # 高難度 → Opus (将来)

# 予算管理
budget:
  daily_limit_usd: 10.0           # 1日の上限金額
  alert_threshold: 0.8            # アラートを出す使用率
  fallback_to_local: true         # 予算超過時にローカルへフォールバック
```

## フィールド説明

### `primary_provider`

| 値 | プロバイダー |
|----|------------|
| `"anthropic"` | Anthropic Claude |
| `"openai"` | OpenAI GPT |
| `"ollama"` | Ollama (ローカル) |
| `"vllm"` | vLLM (ローカル) |
| `"azure_openai"` | Azure OpenAI |
| `"together"` | Together.ai |

### `task_routing`

| タスク | 説明 | 推奨モデル |
|-------|------|---------|
| `query_parsing` | クエリ意図の分類 | haiku（軽量）|
| `retrieval_verification` | 検索結果の裏どり | haiku |
| `response_generation` | メイン回答の生成 | sonnet |
| `code_generation` | コード生成 | sonnet |
| `verification` | 最終検証 | sonnet |

### `budget`

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `daily_limit_usd` | float | 10.0 | 1日の API 費用上限（USD）|
| `alert_threshold` | float | 0.8 | アラートを出す使用率（0〜1）|
| `fallback_to_local` | bool | true | 予算超過時に Ollama にフォールバック |

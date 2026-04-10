# OpenRouter 無料モデル 調査記録

最終更新: 2026-04-11

本ドキュメントは MED プロジェクトの `mature` ジョブ（MemoryReviewer / DifficultyTagger）に使用する
OpenRouter 無料モデルのベンチマーク結果と運用上の知見をまとめる。

---

## テスト方法

`scripts/openrouter_model_test.py` で全無料モデルに同一ドキュメント（FAISS IndexIVFPQ の解説）を
レビューさせ、以下を計測した。

- **speed_s**: 初回レスポンスまでの秒数
- **quality_score**: 0〜1 の品質スコア（reviewer の confidence * 正答判定）
- **overall**: `best` / `good` / `fair` / `poor` の4段階評価
- テスト日: 2026-03-31 前後

---

## 結果一覧

### ✅ 動作確認済み（overall あり）

| モデル ID | コンテキスト | speed | quality | conf | approved | overall | 備考 |
|-----------|-------------|-------|---------|------|----------|---------|------|
| `nvidia/nemotron-3-nano-30b-a3b:free` | 256K | 2.0s | 0.78 | 0.86 | ✓ | **best** | **現デフォルト**。Apr 10 バッチ実績: 237件・承認率65%・429ゼロ。速度最速 |
| `nvidia/nemotron-nano-12b-v2-vl:free` | 128K | 10.2s | 0.70 | 0.90 | ✓ | good | 強み/弱点を的確に指摘。フォールバック用途 |
| `nvidia/nemotron-3-super-120b-a12b:free` | 262K | 13.4s | 0.72 | 0.81 | ✓ | fair | Apr 10 バッチ実績: 63件・承認率33%（厳格すぎ）。429なし |
| `arcee-ai/trinity-large-preview:free` | 131K | 9.1s | 0.80 | 0.90 | ✓ | NG | 品質高いが **2026-04-03 に期限切れ**。使用不可 |
| `arcee-ai/trinity-mini:free` | 131K | 38.8s | 0.80 | 0.90 | ✓ | fair | 品質高いが速度不安定（3s〜39s）。1行理由 |
| `liquid/lfm-2.5-1.2b-thinking:free` | 32K | 14.4s | 0.60 | 0.60 | ✓ | poor | 1.2B thinking。理由が3語と極端に短い。小さすぎる |
| `liquid/lfm-2.5-1.2b-instruct:free` | 32K | 2.2s | 0.00 | 0.00 | ✓ | poor | 出力が意味不明。使用不可 |
| `z-ai/glm-4.5-air:free` | 131K | 121.6s | 0.70 | 0.90 | ✓ | poor | 品質は可だが **121秒** で実用不可 |
| `stepfun/step-3.5-flash:free` | 256K | 15.5s | 0.75 | 0.90 | ✗ | poor | コード例がないと approve しない。概念的な文書を過剰 HOLD |
| `qwen/qwen3.6-plus-preview:free` | 1M | 35.2s | 0.75 | 0.90 | ✗ | poor | 遅く（35s）、コード例がないと HOLD。バッチ mature 向きでない |

### ❌ エラー・使用不可

| モデル ID | エラー種別 | 詳細 |
|-----------|-----------|------|
| `google/gemma-3-4b-it:free` | 400 Bad Request | Google AI Studio：何らかの制約（全 Gemma 3 系で同様） |
| `google/gemma-3-12b-it:free` | 400 Bad Request | 同上 |
| `google/gemma-3-27b-it:free` | 429 Rate Limited | Google AI Studio upstream。テスト時点で率リミット |
| `google/gemma-3n-e2b-it:free` | 400 Bad Request | 同上 |
| `google/gemma-3n-e4b-it:free` | 400 Bad Request | 同上 |
| `meta-llama/llama-3.2-3b-instruct:free` | 429 Rate Limited | upstream |
| `meta-llama/llama-3.3-70b-instruct:free` | 429 Rate Limited | Venice |
| `nousresearch/hermes-3-llama-3.1-405b:free` | 429 Rate Limited | Venice |
| `qwen/qwen3-next-80b-a3b-instruct:free` | 429 Rate Limited | Venice |
| `qwen/qwen3-coder:free` | 429 Rate Limited | Venice |
| `nvidia/nemotron-nano-9b-v2:free` | TypeError (NoneType) | 不安定（テスト時クラッシュ） |
| `minimax/minimax-m2.5:free` | 429 Rate Limited | 2026-04-10 確認: OpenInference upstream 429（以前は 404 No endpoints）|
| `openai/gpt-oss-120b:free` | 404 No endpoints | 同上 |
| `openai/gpt-oss-20b:free` | 404 No endpoints | 同上 |
| `cognitivecomputations/dolphin-mistral-24b-venice-edition:free` | Venice endpoint | テスト未完 |

---

## Gemma 4 シリーズのレートリミット問題（2026-04-09〜10 確認）

### 背景

OpenRouter 経由で `google/gemma-4-*:free` を使用すると、上流の **Google AI Studio** 側で
429 Too Many Requests が返される問題が発生。

### 影響モデル

| モデル ID | 状況 |
|-----------|------|
| `google/gemma-4-31b-it:free` | 2026-04-09 以降、ほぼ全リクエストが 429 |
| `google/gemma-4-26b-a4b-it:free` | 同様に 429（2026-04-10 確認） |

### エラーメッセージ

```
google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly,
or add your own key to accumulate your rate limits:
https://openrouter.ai/settings/integrations
Provider: Google AI Studio, is_byok: False
```

### 対策として実施した設定

`configs/llm_config.yaml` に `model_rate_limits` を追加：

```yaml
openrouter:
  model_rate_limits:
    "google/gemma-4-31b-it:free": 1    # 1 RPM（最大マージン）
    "google/gemma-4-26b-a4b-it:free": 5
```

**結果**: 1 RPM に制限しても 429 が発生。Google AI Studio 側の一時的なキャパシティ制限と判断。
BYOKでなければ回避困難。

### 結論

- Gemma 4 無料枠は **利用可能な時間帯が限定的**（夜間や混雑時は使用不可）
- 安定した mature ジョブには **NVIDIA Nemotron 系** の方が適している
- Gemma 4 を使うなら BYOKでGoogle AI Studio APIキーを登録する必要がある

---

## 運用実績（OpenRouter 使用量）

| 日付 (UTC) | リクエスト数 | 上限 | 結果 |
|-----------|-------------|------|------|
| 2026-04-02 | 1,694 | 1,000 | 上限超過（旧設定） |
| 2026-04-03 | 902 | 950 | 正常 |
| 2026-04-04 | 961 | 950 | 上限超過 → DailyLimitExceeded 動作確認 |
| 2026-04-05 | 912 | 950 | 正常 |
| 2026-04-06 | 957 | 950 | 上限超過 → 自動停止 |
| 2026-04-07 | 953 | 950 | 上限超過 → 自動停止 |
| 2026-04-08 | 385 | 950 | 正常（途中停止） |
| 2026-04-09 | 437 | 950 | 正常（Gemma 4 429 問題で低調） |
| 2026-04-10 | 479 | 950 | 正常（nemotron-3-nano-30b で安定稼働） |

---

## 推奨設定（2026-04-11 時点）

```yaml
# configs/llm_config.yaml
providers:
  openrouter:
    default_model: nvidia/nemotron-3-nano-30b-a3b:free  # ← 現デフォルト
    requests_per_minute: 1   # 全モデル共通 1 RPM（安全設定）
    daily_request_limit: 950
    model_rate_limits:
      "nvidia/nemotron-3-nano-30b-a3b:free": 1
      "nvidia/nemotron-nano-12b-v2-vl:free": 1
      "nvidia/nemotron-3-super-120b-a12b:free": 1
      "google/gemma-4-31b-it:free": 1
      "minimax/minimax-m2.5:free": 1
```

### mature ジョブのプロバイダー優先順位

| 優先度 | プロバイダー | モデル | 承認率 | 備考 |
|--------|-----------|--------|--------|------|
| 1 | `fastflowlm` | `qwen3.5:9b` (NPU Q4_1) | - | ローカル。オフライン時は使用不可 |
| 2 | `openrouter` | `nvidia/nemotron-3-nano-30b-a3b:free` | **65%** | **現デフォルト**。429なし・速度最速 |
| 3 | `openrouter` | `nvidia/nemotron-nano-12b-v2-vl:free` | - | フォールバック。429なし |
| 4 | `openrouter` | `nvidia/nemotron-3-super-120b-a12b:free` | 33% | 429なしだが厳格すぎ |
| NG | `openrouter` | `google/gemma-4-*:free` | - | 429 多発。BYOKなしでは安定利用不可 |
| NG | `openrouter` | `minimax/minimax-m2.5:free` | - | OpenInference 429 |

---

## FastFlowLM（ローカル NPU）モデル評価（2026-04-09）

接続先: `http://192.168.2.104:52625/v1`

| モデル | IFBench (BF16) | IFBench (Q4_1 推定) | 評価 |
|--------|---------------|-------------------|------|
| `qwen3.5:9b` | 64.5% | ≈57% | **推奨**。JSON 品質・速度のバランス最良 |
| `qwen3.5:4b` | 59.2% | ≈50% | 許容範囲。JSON 失敗リスクあり |
| `qwen3.5:2b` | 41.3% | ≈35% | **不推奨**。単純タスク（tagger）のみ可 |
| `qwen3-it:4b` | - | - | OK（it 版のみ。thinking モード漏れがない） |
| `lfm2.5-it:1.2b` | - | - | **NG**。reason フィールドがテンプレ文字列をそのままコピー |
| `qwen3:*` 系 | - | - | **NG**。thinking モードで思考が出力に漏れ JSON 解析失敗 |

- NPU decode 速度: **2.5 tokens/s**（autoregressive の構造的制約、NPU メモリ帯域律速）
- overnight ジョブに適している（速度は遅いが電力効率は高い）

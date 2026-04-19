# TODO: FastFlowLM Server 'think' Flag Integration

## 現状
- FastFlowLM Server は `think` フラグを request payload でサポート
- ドキュメント: "Server Mode: Set the 'think' flag in the request payload."
- 現在の実装では this flag の活用状況が不明

## テスト結果
### System-level thinking instruction (2026-04-19)
- system prompt に "Think carefully..." 指示 → 承認率 60% → 40% (厳格化)
- enable_thinking (extra_params chat_template_kwargs) は効果なし

### Payload-level 'think' keyword (2026-04-19)
- payload に "think about ... carefully." 追加 → 承認率 60% (差なし)
- 全バリエーション (with_think / with_think_lowercase / with_analyze) 結果同一

## 疑問点
1. FastFlowLM Server の request payload 内 `think` フラグ形式が不明
   - chat_template_kwargs に含める?
   - extra_body に含める?
   - 独立フィールド?

2. enable_thinking (extra_params) vs think フラグの関係性
   - enable_thinking: Qwen3 chat_template_kwargs を通じて制御
   - think フラグ: FastFlowLM Server native の制御?
   - 両者の関係・優先順位不明

## Next Steps
- [ ] FastFlowLM Server の実装/ドキュメント確認
- [ ] think フラグの正確な payload 形式を特定
- [ ] enable_thinking との動作の違いを実証テスト
- [ ] reviewer.py で思考制御を think フラグで統一化（可能性の検討）

## Reference
- FastFlowLM CLI: `/think` コマンドで対話的なトグル
- FastFlowLM Server Mode: request payload の `"think"` フラグ
- 現在の実装: `extra_params.chat_template_kwargs.enable_thinking` で制御中

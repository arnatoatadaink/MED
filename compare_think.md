# Qwen3.5-9b Think / Nothink 制御検証レポート

**実施日**: 2026-04-18〜19  
**目的**: FastFlowLM (Qwen3.5-9b) の thinking ON/OFF 制御が review 品質に与える影響を検証し、
`/nothink` トークンの正しい配置と、メモリ汚染リスクを確認する。

---

## 1. 前提: /nothink の配置問題

### 旧実装（system メッセージ配置）
```python
# reviewer.py — 旧実装（誤り）
if self._provider in _NO_THINK_PROVIDERS:
    system = "/nothink\n\n" + system   # system ロールに付与
```

### 新実装（user メッセージ先頭配置）
```python
# reviewer.py — 新実装（正しい）
if self._provider in _NO_THINK_PROVIDERS:
    prompt = "/nothink\n\n" + prompt   # user 先頭に付与
```

Qwen3 系の `/nothink` ソフトスイッチは **user メッセージの先頭** に置かなければモデルが認識しない。
system ロールに配置した場合、Qwen3 はこれを通常の指示として処理し thinking を抑制しない。

---

## 2. テスト設計

### 2-A. 並列テスト（`--fragment-test`）
- 3モデル同時送信: Gemma4-Q8 / Qwen3.5 nothink / Qwen3.5 think
- FastFlowLM は NPU 単体キューのため実際は直列処理
- **問題**: elapsed_s はキュー待ち時間を含むため thinking 時間の計測に不適

### 2-B. 直列テスト（`--sequential-mode-test`）
- THINK 全件 → llama3.2:1b ダミー → nothink 全件
- ダミーリクエストでリクエスト間の状態をリセット
- seed=99 固定で再現可能

### 2-C. system prefix テスト（`--sequential-mode-test --system-prefix`）
- Phase2 の nothink で `/nothink` を system に付与（旧実装の再現）
- モデルリロード前後で各1回実施

### 2-D. インターリーブテスト（`--interleaved-test`）
- think[i] → nothink[i] を文書ごとに交互実行
- nothink → think 方向の汚染伝播を確認

---

## 3. 判定が分かれた文書

### Doc-1: `b58fe82869` ★ think/nothink の最重要差異文書
- **ソース**: Stack Overflow  
- **元ステータス**: approved q=0.80  
- **内容** (152文字):
  ```
  You can traverse from the node[5] to get the path [5]->[0], and then use the
  function path.reverseNodes() to get the nodes on the path in reverse order.
  ```
- **テスト**: sequential user prefix (2026-04-19 00:14)

| モード | 判定 | quality | reason |
|--------|------|---------|--------|
| THINK | **APPROVED** | 0.80 | "sufficiently complete as an answer to an implied question" |
| nothink | **NEEDS_SUPP** | 0.50 | "too thin and shallow; single sentence without context or explanation" |

**考察**: THINK モードは「暗黙の質問への回答」という文脈を推測して承認（憶測補完）。
nothink は文書そのものを客観的に評価して正しく NEEDS_SUPP と判定。

---

### Doc-2: `368e1586f0`
- **ソース**: arXiv  
- **元ステータス**: approved q=0.65  
- **内容** (409文字): Kalman Filter の Python 実装に関する論文 abstract
- **テスト**: 並列テスト Phase-1 (Gemma4 vs Qwen)

| モード | 判定 | quality |
|--------|------|---------|
| Gemma4-Q8 | **APPROVED** | 0.70 |
| Qwen3.5 nothink | **NEEDS_SUPP** | 0.40 |
| Qwen3.5 think | **NEEDS_SUPP** | 0.50 |

**考察**: Gemma4 が abstract を「完全」と誤判定。Qwen 両モードは一貫して薄い abstract と正しく判定。

---

### Doc-3: `0a4c466b08`
- **ソース**: GitHub Docs (tldr-pages)  
- **元ステータス**: approved q=0.80  
- **内容** (288文字): `a2disconf` コマンド — **broken link** (`More information: .`)
  ```
  tldr-pages Linux: a2disconf
  # a2disconf
  > Disable an Apache configuration file on Debian-based OSes.
  > More information: .
  ...
  ```
- **テスト**: 並列テスト

| モード | 判定 | quality |
|--------|------|---------|
| Gemma4-Q8 | **APPROVED** | 0.90 |
| Qwen3.5 nothink | **NEEDS_SUPP** | 0.50 |
| Qwen3.5 think | **NEEDS_SUPP** | 0.50 |

---

### Doc-4: `df073d7f1e`
- **ソース**: GitHub Docs (tldr-pages)  
- **元ステータス**: approved q=0.78  
- **内容** (348文字): `aa-decode` コマンド — **broken link** (`More information: .`)
- **テスト**: sequential system prefix (汚染あり)

| モード | 判定 | quality |
|--------|------|---------|
| THINK | **NEEDS_SUPP** | 0.40 | "broken reference, too shallow" |
| nothink (system prefix・汚染) | **APPROVED** | 0.80 | "concise reference for aa-decode" |

---

### Doc-5: `f7955ccaaf`
- **ソース**: GitHub Docs (tldr-pages)  
- **元ステータス**: approved q=0.70  
- **内容** (374文字): `blkid` コマンド — **broken link** (`More information: .`) + placeholder 構文
- **テスト**: sequential system prefix (汚染あり)

| モード | 判定 | quality |
|--------|------|---------|
| THINK | **NEEDS_SUPP** | 0.50 | "contains placeholders, incomplete" |
| nothink (system prefix・汚染) | **APPROVED** | 0.80 | "provides clear commands" |

---

## 4. system prefix 汚染の確認

### モデルリロード前（THINK 後の状態が残る）
| Phase | 承認率 | avg time |
|-------|--------|----------|
| THINK (Phase1) | 40% | 49.6s |
| nothink/system (Phase2) | **100%** | 48.0s |

THINK の KV キャッシュ等の状態が nothink(system) フェーズに影響し、
**本来 NEEDS_SUPP であるべき文書が全件 APPROVED** になる危険な状態。

### モデルリロード後（クリーン状態）
| Phase | 承認率 | avg time |
|-------|--------|----------|
| THINK (Phase1) | 80% | 49.4s |
| nothink/system (Phase2) | **80%** | 49.8s |

**判定 5/5 一致、時間差 0.4s** — system prefix は完全に無効であることが確定。

---

## 5. user prefix 汚染テスト（インターリーブ）

think[i] → nothink[i] を交互に直列実行した結果:

| | THINK | nothink (user) |
|--|-------|----------------|
| 承認率 | 20% | 20% |
| 平均処理時間 | 42.9s | 41.6s |
| 判定一致 | 5/5 | |

THINK 処理時間の推移: 55.2s → 33.8s → 39.1s → 46.4s → 40.1s（単調減少なし）  
→ **nothink → think 方向の汚染伝播なし**。user prefix はリクエスト間で独立動作。

---

## 6. 処理時間の考察

直列実行での実測値:

| テスト条件 | THINK avg | nothink avg | 差 |
|-----------|-----------|-------------|-----|
| sequential user prefix | 47.9s | 43.9s | **4.0s** |
| sequential system prefix (reload後) | 49.4s | 49.8s | 0.4s |
| interleaved user prefix | 42.9s | 41.6s | 1.3s |

- user prefix: 一貫して nothink が 1.3〜4.0s 速い → thinking トークンが少量生成されている証拠
- system prefix: 差なし → thinking 完全に抑制できていない

`reasoning_content` はすべてのレスポンスで `None`。FastFlowLM が thinking トークンを
専用フィールドに分離しない実装のため、`<think>...</think>` は content に含まれているか、
内部で消費されている可能性がある。

---

## 7. 結論

| 項目 | 結論 |
|------|------|
| system prefix (/nothink) | **無効**。Qwen3 は user メッセージでのみ認識 |
| user prefix (/nothink) | **有効**。承認率・処理時間ともに差が出る |
| thinking ON による弊害 | **憶測補完**: 単文断片を「文脈を推測して」承認 |
| 汚染伝播 (think→nothink) | system prefix 時のみ発生。user prefix では発生しない |
| 汚染伝播 (nothink→think) | **発生しない**（インターリーブテストで確認） |
| 推奨設定 | `reviewer.py`: user 先頭 `/nothink\n\n` + prompt を維持 |

---

## 8. 対象設問（OpenRouter nemotron 比較用）

以下の文書で nemotron 3種の speculation 傾向を比較する。

| Doc ID | 種別 | 期待判定 | speculation ポイント |
|--------|------|----------|---------------------|
| `b58fe82869` | SO 1文断片 | NEEDS_SUPP | 文脈推測による承認 |
| `368e1586f0` | arXiv abstract 薄い | NEEDS_SUPP | abstract を完全と誤認 |
| `0a4c466b08` | tldr broken link | NEEDS_SUPP | broken link 無視 |
| `df073d7f1e` | tldr broken link | NEEDS_SUPP | broken link 無視 |
| `f7955ccaaf` | tldr broken link + placeholder | NEEDS_SUPP | placeholder を実コマンドと誤認 |

**正解**: 5件すべて NEEDS_SUPP（broken link, thin content, single-sentence fragment）  
nemotron が何件を APPROVED にするかで speculation 傾向を定量化できる。

---

## 9. OpenRouter nemotron 3種 比較テスト結果

**実施日**: 2026-04-19  
**対象文書**: Section 8 の5件（全件 NEEDS_SUPP が正解）  
**テスト方法**: OpenRouter 直列実行、62秒間隔、temperature=0.0

### 9-1. モデル別サマリー

| モデル | 誤承認数 | 誤承認率 | avg quality |
|--------|----------|----------|-------------|
| nemotron-3-nano-30b-a3b:free | 3/5 | 60% | 0.592 |
| nemotron-nano-12b-v2-vl:free | 3/5 | 60% | 0.720 |
| nemotron-3-super-120b-a12b:free | 3/5 | 60% | 0.674 |
| **Qwen3.5-9b nothink (参考)** | **1/5** | **20%** | 0.480 |
| **Gemma4-26b Q8 (参考)** | **1-2/5** | **20-40%** | — |

### 9-2. 文書別の判定（正解: 全件 NEEDS_SUPP）

| 文書 | 種別 | 30b-nano | 12b-vl | 120b-super |
|------|------|:--------:|:------:|:----------:|
| `b58fe828` SO 1文断片 | ✓ | ✓ | ✓ |
| `368e1586` Kalman abstract 薄い | **✗** | **✗** | **✗** |
| `0a4c466b` tldr broken link | **✗** | **✗** | ✓ |
| `df073d7f` tldr broken link | **✗** | ✓ | **✗** |
| `f7955cca` tldr placeholder | ✓ | **✗** | **✗** |

✓ = 正解(NEEDS_SUPP)、✗ = 誤承認(APPROVED)

### 9-3. 所見

**`368e1586f0`（Kalman Filter abstract）は nemotron 3種全員が誤承認。**  
429文字の整った abstract に対し全モデルが「正確・自己完結」と判断する。
Qwen3.5-9b nothink のみが「abstract として薄い」と正しく判定できた唯一のモデル。

**broken link（`More information: .`）は全モデルが見逃す傾向。**  
nemotron 3種はいずれも「minor issue」として無視して承認する。
Qwen3.5-9b nothink は broken link を含む文書を含めて NEEDS_SUPP と判定する。

**モデルサイズ（12b/30b/120b）に関わらず誤承認率が一定（60%）。**  
パラメータ数増加が speculation 率の改善に繋がっていない。
誤承認の内訳は各モデルで異なり（一貫性なし）、size 依存の傾向は見られない。

---

## 10. モデル別 review 品質の総合評価

### 10-1. 厳格さの比較（全テスト横断）

| モデル | 傾向 | 備考 |
|--------|------|------|
| Qwen3.5-9b nothink | **最も厳格** | 今回テストで誤承認 1/5。broken link・薄い内容に対して一貫して NEEDS_SUPP |
| Gemma4-26b Q8 | **同等に厳格** | 前回調査（Apr17）では最も厳格。今回テストデータでは誤承認あり |
| Qwen3.5-9b think | 許容度高め | 文脈推測による憶測承認が発生（`b58fe828` 事例） |
| nemotron-3-nano-30b-a3b | 許容度高い | 誤承認 3/5。現行デフォルトモデルとして使用中だが speculation リスクあり |
| nemotron-nano-12b-vl | 許容度高い | 誤承認 3/5。nemotron-30b と同等 |
| nemotron-3-super-120b | 許容度高い | 誤承認 3/5。サイズ増加の効果なし |

### 10-2. Qwen3.5-9b と Gemma4 の補足

**Gemma4 の誤承認はテストデータの偏りによる影響が大きい。**  
Apr17 調査では Gemma4 が最も厳格なレビューを行い、nemotron の speculation を
最初に検出するきっかけになった。今回の5文書は tldr-pages や短い abstract 中心であり、
このカテゴリに対して Gemma4 が若干許容的に判断する傾向があった可能性がある。

**Qwen3.5-9b nothink と Gemma4-26b は同等に信頼できる reviewer。**  
両モデルが独立したアーキテクチャ・訓練データを持ちながら同等の厳格さを示すことは、
review 品質の安定性の観点から望ましい。実運用では両モデルの合意を参考にするか、
より多くの文書でクロスバリデーションを行うことが推奨される。

### 10-3. 運用推奨

| 用途 | 推奨モデル | 理由 |
|------|-----------|------|
| 日次 mature（FastFlowLM） | Qwen3.5-9b + `/nothink` user prefix | 厳格・高速・コストゼロ |
| 日次 mature（OpenRouter） | Qwen3.5-9b 系または Gemma4 系 | nemotron は speculation リスクあり |
| nemotron-3-nano-30b-a3b | 廃止検討 | speculation 率 60%、品質改善が見込めない |

---

## 11. Strict Persona 追加テスト（nemotron 3種）

**実施日**: 2026-04-19  
**方法**: Section 9 と同一 5文書に対し、system メッセージ先頭に厳格レビュアーペルソナを追加:

```
You are a strict and uncompromising reviewer. Apply the quality criteria rigorously without giving benefit of the doubt.
```

### 11-1. モデル別サマリー（正解: 全件 NEEDS_SUPP）

| モデル | 誤承認数（通常） | 誤承認数（strict） | 変化 |
|--------|:---------------:|:-----------------:|------|
| nemotron-3-nano-30b-a3b | 3/5 (60%) | **4/5 (80%)** | 悪化 |
| nemotron-nano-12b-v2-vl | 3/5 (60%) | **2/5 (40%)** | 改善 |
| nemotron-3-super-120b | 3/5 (60%) | **3/5 (60%)** | 変化なし |

### 11-2. 文書別の判定（正解: 全件 NEEDS_SUPP）

| 文書 | 種別 | 30b-nano(通常) | 30b-nano(strict) | 12b-vl(通常) | 12b-vl(strict) | 120b-super(通常) | 120b-super(strict) |
|------|------|:--------------:|:----------------:|:------------:|:--------------:|:----------------:|:-----------------:|
| `b58fe828` SO 1文断片 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `368e1586` Kalman abstract | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** |
| `0a4c466b` tldr broken link | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ |
| `df073d7f` tldr broken link | **✗** | **✗** | ✓ | ✓ | **✗** | **✗** | **✗** |
| `f7955cca` tldr placeholder | ✓ | **✗** | **✗** | **✗** | ✓ | **✗** | **✗** |

✓ = 正解(NEEDS_SUPP)、✗ = 誤承認(APPROVED)

### 11-3. 所見

**Strict persona は nemotron の speculation を安定的に抑制しない。**  
- 30b-nano では逆に悪化（60% → 80%）: ペルソナが一部文書への判定基準を変えるが、
  別文書でむしろ寛容になるという不整合が生じた
- 12b-vl では改善（60% → 40%）したが、依然 2件の誤承認が残る
- 120b-super は変化なし（60% → 60%）: ペルソナ追加の効果なし

**`368e1586f0`（Kalman Filter abstract）は strict persona でも nemotron 3種が全員誤承認。**  
429文字の整った abstract に対する「完全なドキュメント」という誤認識が
strict persona 程度では変わらない根深い傾向を示している。

**`b58fe82869`（SO 1文断片）は全モデル・全条件で正しく NEEDS_SUPP と判定。**  
152文字の明らかな断片に対しては nemotron も正しく動作する。
speculation が顕在化するのは「それなりに整った内容だが実質薄い」文書に対してである。

### 11-4. 結論

Strict persona の追加は nemotron の speculation リスクを根本的に解消しない。
判定の一貫性が低く（同一ペルソナで文書によって誤承認方向が変化）、
プロダクション利用には信頼性が不十分。**nemotron廃止推奨の結論に変更なし。**

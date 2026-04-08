# plan_translate.md — 多言語対応・翻訳・スタイル抽出 計画

## 優先度: 低（MED本体の安定後に着手）

---

## Part 1: MED 多言語埋め込みモデル移行

### 背景
- 現状: all-MiniLM-L6-v2（英語特化, 384-dim, 22M params）
- 問題: 非英語ドキュメントのベクトル品質が低い
- 方針: 多言語モデルに切り替え、英語クエリ↔多言語ドキュメントのクロスリンガル検索を可能にする

### 採用モデル
`paraphrase-multilingual-MiniLM-L12-v2`
- 384-dim（現行と同じ、FAISSインデックス設定変更不要）
- 50言語対応
- 118MB（軽量）
- sentence-transformers で利用可能

### 実装手順

#### Step 1: バックアップ（all-MiniLM-L6-v2 ベクトルを保持）
```bash
cp -r data/faiss_indices/ data/faiss_indices_minilm_backup/
```
用途: 将来の実験比較用（多言語モデルとの検索品質比較など）

#### Step 2: configs/default.yaml 変更
```yaml
embedding:
  model: "paraphrase-multilingual-MiniLM-L12-v2"
  dim: 384        # 変更不要
  batch_size: 32  # 12層モデルのためバッチサイズを半減
  device: "cpu"
  cache_dir: "data/models"
```

#### Step 3: モデルダウンロード
```bash
poetry run python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.save('data/models/paraphrase-multilingual-MiniLM-L12-v2')
"
```

#### Step 4: 一括再インデックス（推奨: バッチ方式）
- 対象: approved docs 全件（現在 ~4,600件）
- 推定時間: 3-5分（バッチ埋め込み方式）
- スクリプト: `scripts/reindex_faiss.py`（未作成）

```python
# reindex_faiss.py の概要
# 1. metadata.db から approved docs を全件取得
# 2. 新モデルで embed_batch()
# 3. FAISSインデックスを空にしてから一括追加
# 4. id_map を再構築・保存
```

#### Step 5: CLAUDE.md 更新
- 埋め込みモデル名・次元数の記述を更新

### 注意事項
- all-MiniLM-L6-v2 バックアップは `data/faiss_indices_minilm_backup/` に保持
  → 実験用途が確定したら削除判断を再検討
- StudentモデルはTinyLoRA経由でFAISSを「使う」側なので、多言語モデルへの変更は影響なし
- 将来的により高品質なモデルが必要な場合: `intfloat/multilingual-e5-base`（768-dim）
  → その際はFAISSインデックスのdim変更と全件再インデックスが必要

---

## Part 2: 翻訳パイプライン（優先度: 低）

### 目的
非英語ドキュメントを英語に翻訳してFAISSに格納することで、英語ベクトル空間の均質性を保つ。

### 現状の判断
- 現在の非英語コンテンツは <5%（手動ドキュメント5件は英訳済み）
- 多言語モデル採用後はクロスリンガル検索が機能するため翻訳は不要
- **翻訳実装は多言語モデルで対応できない品質問題が発生した場合のみ着手**

### 実装案（着手時の参考）
```
検出: langdetect（pip install langdetect）
翻訳: DeepL API（技術文書に強い）または ArgosTranslate（無料・ローカル）
ターゲット: コードブロックは翻訳しない（ルールベースで保護）
フロー: seed時に language を検出 → 非英語なら翻訳 → 英語コンテンツとしてFAISS格納
```

---

## Part 3: 多段翻訳によるスタイル抽出（別プロジェクト）

### アイデア概要
多段翻訳（EN→JA→ZH→EN）を自己符号化器として利用し、コンテンツ残差からスタイルベクトルを抽出する。

```
original_text
    ↓  EN → JA → ZH → EN  (多段翻訳 = スタイル剥離)
content_residual
    ↓
style_vector = embed(original) - embed(content_residual)
```

### 関連研究
- **STRAP** (Krishna et al., 2020): 逆翻訳でスタイル中立パラフレーズを生成
  - "Reformulating Unsupervised Style Transfer as Paraphrase Generation"
- **PAN Authorship Obfuscation** (2020-2024 shared tasks): 多段翻訳をベースライン手法として使用
- **Shen et al., 2017**: 敵対的VAEによるcontent/style潜在空間分離
- **CTRL** (Keskar et al., 2019): スタイル制御コードによる生成制御
- **StyloMetrix**: 文体特徴量の計量化（日本語対応状況は要確認）

### MED との接続点
- 抽出した style_vector は `med_hyp_style_g.md` の StyleVector（3層スタイル分解）として統合予定
- MED Phase 5 の context_emb（NEAT Context-Sensitive Search）の個人スタイル層として活用
- `src/memory/embedder.py` にスタイル剥離オプションとして差し込み可能

### 別プロジェクト化の理由
1. 翻訳APIコスト（DeepL/Google Translate）が MED 本体と分離すべき予算
2. 研究的不確実性が高い（スタイルベクトルの品質が未知）
3. `neat_trident` プロジェクトとの統合実験も候補

### 実装ステップ（着手時）
1. STRAP 論文を再現: 英語テキスト → EN→FR→EN → 差分ベクトルを可視化（t-SNE）
2. 多段翻訳（2-4ホップ）でのスタイル剥離度合いを評価
3. スタイルベクトルをMED StyleVectorの入力として検証
4. 成功した場合: `src/memory/` にスタイル剥離埋め込みオプションを追加

---

## TODO 一覧

| 項目 | 優先度 | 状態 | 備考 |
|------|--------|------|------|
| all-MiniLM-L6-v2 バックアップ | 高 | ⬜ | モデル変更前に実施 |
| 多言語モデルへの切り替え | 高 | ⬜ | `scripts/reindex_faiss.py` 作成が必要 |
| CLAUDE.md 埋め込みモデル記述更新 | 高 | ⬜ | モデル変更後 |
| 翻訳パイプライン実装 | 低 | ⬜ | 多言語モデルで解決できなければ着手 |
| スタイル抽出実験（別プロジェクト） | 低 | ⬜ | STRAP再現から開始 |
| StyleVector → MED Phase 5 統合 | 低 | ⬜ | `med_hyp_style_g.md` 参照 |

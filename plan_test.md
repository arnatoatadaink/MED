# plan_test.md — testmon + xdist Docker統合計画

## 目的

MEDフレームワークのGitHub Actions CIを以下の構成に移行する：
- **pytest-testmon**：変更ファイルに依存するテストのみ抽出
- **pytest-xdist**：抽出されたテストを並列実行
- **Docker**：再現性のある隔離環境で実行
- **.testmondataキャッシュ**：CI間で差分情報を引き継ぎ

目標：現状6時間超 → 差分実行により大幅短縮

---

## 重要な発見（タイムアウト根本原因）

### `sentence-transformers` が PyTorch (~2GB) を引き込んでいた

```
# 現状の ci.yml unit-tests ジョブ
pip install sentence-transformers  ← PyTorch ~2GB を毎回ダウンロード
```

**しかし、ユニットテストは `sentence-transformers` を一切使っていない。**
`src/memory/embedder.py` のテストはすべて `mock=True` モードで動作するため、
実際のモデルロードは発生しない。

```python
# tests/unit/test_embedder.py — モックで動作する証拠
@pytest.fixture
def mock_embedder() -> Embedder:
    config = EmbedderConfig(model_name="test-model", mock=True)
    return Embedder(config=config, mock=True)   # ← 実モデル不要
```

### 現状の無駄な処理

| 問題 | 所要時間 |
|------|---------|
| `pip install sentence-transformers`（PyTorch 2GB）| 30〜60 分 × 2回（ホスト + Docker） |
| pytest を 3 回実行（unit-tests / Run unit tests in Docker / Extract coverage） | 実質同じテストを重複実行 |
| `timeout-minutes` 未設定 → デフォルト 6h | ハングアップを 6h 許容 |

### 即効対策（Phase 0 として先行実施）

1. **`timeout-minutes` を追加**（lint: 5分 / unit-tests: 20分 / docker-tests: 30分）
2. **`sentence-transformers` を CI インストールから除外**（tests は mock=True）
3. **Docker テスト内の重複 pytest を削除**（unit-tests ジョブで完了済み）

これだけで CI 時間を **6h → ~20 分** に短縮できる。
testmon + xdist 移行（Phase 1〜4）はその後に実施する。

---

## フェーズ概要

| Phase | 作業 | 完了条件 |
|-------|------|----------|
| **0** | **即効修正**（timeout / sentence-transformers 除外 / 重複 pytest 削除） | CI が 20 分以内に完了する |
| 1 | Dockerfile.test修正 | testmon/xdist入り軽量イメージが build できる |
| 2 | ローカル動作確認 | `docker run` で testmon + xdist が動く |
| 3 | GitHub Actions修正 | CI上でキャッシュ込みのワークフローが通る |
| 4 | 週次フルランワークフロー追加 | 毎週日曜に .testmondata リセット |

---

## Phase 0：即効修正（ci.yml + Dockerfile.test）

### 0-1. `timeout-minutes` を各ジョブに追加

```yaml
jobs:
  lint:
    timeout-minutes: 5
  unit-tests:
    timeout-minutes: 20
  docker-tests:
    timeout-minutes: 30
```

### 0-2. `sentence-transformers` を CI インストールから除外

```yaml
# unit-tests ジョブ（修正後）
- name: Install dependencies
  run: |
    pip install -e ".[dev]"
    pip install faiss-cpu aiosqlite pyyaml httpx networkx docker
    # sentence-transformers は削除（tests/unit は mock=True で動作するため不要）
```

### 0-3. Dockerfile.test から `sentence-transformers` を除外

```dockerfile
# 修正後
RUN pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir \
        faiss-cpu \
        aiosqlite \
        pyyaml \
        httpx \
        networkx \
        docker
# sentence-transformers / PyTorch を除外 → イメージサイズ ~2GB 削減
```

### 0-4. docker-tests ジョブの重複 pytest を削除

```yaml
# 削除するステップ（unit-tests ジョブで実行済み）
# - name: Run unit tests in Docker   ← 削除
# - name: Extract coverage report    ← 削除（3回目の pytest を防ぐ）
```

---

## Phase 1：Dockerfile.test の修正

### 現状の想定構成

```
project/
├── Dockerfile.test        ← 修正対象
├── docker-compose.test.yml ← 必要なら修正
├── requirements-dev.txt   ← 追記対象
├── src/
└── tests/
```

### requirements-dev.txt に追記

```txt
pytest>=7.4
pytest-testmon>=2.1
pytest-xdist>=3.5
pytest-cov>=4.1
```

### Dockerfile.test 修正版

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 依存レイヤー（キャッシュ効率のため分離）
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# ソースとテストをコピー
COPY src/ ./src/
COPY tests/ ./tests/

# .testmondataはホストからマウントするため COPY しない
# CMD はワークフロー側で上書き
CMD ["pytest", "--help"]
```

### docker-compose.test.yml（新規 or 修正）

```yaml
version: "3.9"

services:
  test:
    build:
      context: .
      dockerfile: Dockerfile.test
    environment:
      - PYTHONPATH=/app
    volumes:
      # .testmondataをホスト↔コンテナで共有（★重要）
      - ./.testmondata:/app/.testmondata
    # commandはCLIで上書き
```

---

## Phase 2：ローカル動作確認手順

### 2-1. イメージビルド

```bash
docker compose -f docker-compose.test.yml build
```

### 2-2. 初回フルラン（.testmondata生成）

```bash
docker compose -f docker-compose.test.yml run --rm test \
  pytest --testmon -v --tb=short
```

実行後、プロジェクトルートに `.testmondata` が生成されることを確認。

### 2-3. ファイルを1つ変更して差分確認

```bash
# 何かソースを1行変更してから：
docker compose -f docker-compose.test.yml run --rm test \
  pytest --testmon --collect-only -q 2>/dev/null | grep "::"
```

変更ファイルに依存するテストだけ表示されれば成功。

### 2-4. xdist並列実行（テストリストを渡す方式）

```bash
# Step1: リスト収集
docker compose -f docker-compose.test.yml run --rm test \
  pytest --testmon --collect-only -q 2>/dev/null \
  | grep "::" > tests_to_run.txt

# Step2: 並列実行
docker compose -f docker-compose.test.yml run --rm test \
  pytest -n 2 --tb=short $(cat tests_to_run.txt | tr '\n' ' ')
```

---

## Phase 3：GitHub Actions ワークフロー

### ファイル：`.github/workflows/test.yml`

```yaml
name: Test (testmon + xdist + Docker)

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      # .testmondataキャッシュ復元
      - name: Cache testmon data
        uses: actions/cache@v4
        with:
          path: .testmondata
          key: testmon-${{ runner.os }}-${{ hashFiles('**/*.py') }}
          restore-keys: |
            testmon-${{ runner.os }}-

      # イメージビルド（変更なければDockerレイヤーキャッシュが効く）
      - name: Build test image
        run: docker compose -f docker-compose.test.yml build

      # Step1: 変更影響テストを抽出
      - name: Collect affected tests (testmon)
        id: collect
        run: |
          docker compose -f docker-compose.test.yml run --rm \
            -v ${{ github.workspace }}/.testmondata:/app/.testmondata \
            test \
            pytest --testmon --collect-only -q 2>/dev/null \
            | grep "::" > tests_to_run.txt || true

          COUNT=$(wc -l < tests_to_run.txt | tr -d ' ')
          echo "count=$COUNT" >> $GITHUB_OUTPUT
          echo "▶ 実行対象テスト数: $COUNT"
          cat tests_to_run.txt

      # Step2: 対象ありの場合のみ並列実行
      - name: Run affected tests in parallel (xdist)
        if: steps.collect.outputs.count != '0'
        run: |
          TEST_LIST=$(cat tests_to_run.txt | tr '\n' ' ')
          docker compose -f docker-compose.test.yml run --rm \
            -v ${{ github.workspace }}/.testmondata:/app/.testmondata \
            test \
            pytest -n 2 --tb=short --junitxml=test-results.xml $TEST_LIST

      # 対象ゼロ → スキップ
      - name: No changes detected
        if: steps.collect.outputs.count == '0'
        run: echo "✅ 変更なし。テストスキップ。"

      # testmondataを最新化
      - name: Update testmon data
        if: steps.collect.outputs.count != '0'
        run: |
          docker compose -f docker-compose.test.yml run --rm \
            -v ${{ github.workspace }}/.testmondata:/app/.testmondata \
            test \
            pytest --testmon --tb=no -q

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-results.xml
```

---

## Phase 4：週次フルランワークフロー

### ファイル：`.github/workflows/test-full.yml`

```yaml
name: Full Test (weekly reset)

on:
  schedule:
    - cron: '0 2 * * 0'   # 毎週日曜 2:00 UTC
  workflow_dispatch:        # 手動実行ボタン

jobs:
  full-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Build test image
        run: docker compose -f docker-compose.test.yml build

      # 全テスト実行 + .testmondata再生成
      - name: Full run + rebuild testmon data
        run: |
          docker compose -f docker-compose.test.yml run --rm \
            -v ${{ github.workspace }}/.testmondata:/app/.testmondata \
            test \
            pytest -n 2 --testmon --tb=short -v \
            --junitxml=test-results-full.xml

      # 新しい.testmondataをキャッシュに保存
      - name: Save fresh testmon data
        uses: actions/cache/save@v4
        with:
          path: .testmondata
          key: testmon-${{ runner.os }}-${{ hashFiles('**/*.py') }}

      - name: Upload full test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-full
          path: test-results-full.xml
```

---

## 全体フロー図

```
push / PR
    │
    ▼
.testmondataキャッシュ復元
    │
    ▼
Docker イメージビルド（レイヤーキャッシュ）
    │
    ▼
testmon: 変更影響テスト収集
    │
    ├─ 0件 ──→ ✅ スキップ（即終了）
    │
    └─ N件 ──→ xdist (-n 2): 並列実行
                    │
                    ▼
               .testmondata更新
                    │
                    ▼
               キャッシュ保存

─────────────────────────────
毎週日曜（cron）
    │
    ▼
全テスト実行 + .testmondata再生成
    │
    ▼
キャッシュ上書き保存
```

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| テストが全件実行される | .testmondataが復元されていない | キャッシュkeyを確認、週次フルランを手動実行 |
| testmon収集が0件なのに変更がある | .testmondataが古い | `workflow_dispatch`で週次フルランを手動実行 |
| Docker volumeマウントエラー | パスが違う | `${{ github.workspace }}`を絶対パスで確認 |
| xdistでテストが競合する | テスト間で共有リソースあり | `tmp_path`フィクスチャや`scope="session"`を見直す |
| .testmondataが壊れる | 並列書き込み競合 | testmon更新は単独ステップで実行（xdistと分離） |

---

## Claude Code への引き継ぎ指示

```
以下の順番でファイルを作成・修正してください：

【Phase 0：即効修正（先に実施）】
0. .github/workflows/ci.yml を修正：
   - 各ジョブに timeout-minutes 追加（lint:5 / unit-tests:20 / docker-tests:30）
   - unit-tests ジョブの pip install から sentence-transformers を削除
   - docker-tests ジョブの "Run unit tests in Docker" ステップを削除
   - docker-tests ジョブの "Extract coverage report" ステップを削除

【Phase 1〜4：testmon + xdist 移行】
1. requirements-dev.txt に pytest-testmon>=2.1, pytest-xdist>=3.5 を追記
2. Dockerfile.test を plan_test.md の Phase1 仕様に修正
   （sentence-transformers は除外済みのため含めない）
3. docker-compose.test.yml を Phase1 仕様で新規作成（または修正）
4. .github/workflows/test.yml を Phase3 の内容で作成
5. .github/workflows/test-full.yml を Phase4 の内容で作成
6. .gitignore に .testmondata を追加しない（キャッシュ対象のため）

確認方法：
- Phase 0: CI が 20 分以内に完了すること
- Phase 1〜: `docker compose -f docker-compose.test.yml build` が通ること
- `docker compose -f docker-compose.test.yml run --rm test pytest --testmon --collect-only -q` が動くこと
```

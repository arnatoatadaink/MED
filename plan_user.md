# plan_user.md — 会話履歴永続化 + ユーザー管理 実装計画（A-1）

## 背景と目的

### 解決する問題

MED の Gradio WebGUI はブラウザ上で動作するため、タブを閉じると会話履歴が失われる。
また、ユーザーの区別がなく、FAISS メモリやセッションがグローバルに共有される。

```
① 会話履歴が揮発する
   → タブを閉じると直前の会話が消失
   → 長いデバッグセッションの途中経過を失う

② ユーザー分離がない
   → 複数人が同じ FAISS インデックスに書き込む
   → プライベートな質問・回答が他ユーザーに見える

③ テスト・開発時の認証が煩雑
   → 毎回ユーザー登録が必要
   → CI / E2E テストで認証をバイパスする仕組みがない
```

### 設計方針

- **サーバーサイド永続化**: SQLite（aiosqlite）に会話履歴を保存し、JWT で認証
- **テストユーザー**: パスワードなしでトークンを発行する開発用ユーザー（localhost 限定）
- **FAISS 自動登録**: assistant ターンをユーザー専用 FAISS に自動保存し知識を蓄積
- **非破壊統合**: 既存の `/query` エンドポイントは認証なしでも動作（`get_optional_user`）

---

## アーキテクチャ

```
Client (Gradio / curl)
    │
    ├── POST /auth/register   → UserStore (users.db)
    ├── POST /auth/login      → JWT 発行
    ├── POST /auth/token/test → テストトークン発行（localhost 限定）
    │
    ├── POST /sessions        → ConversationStore (conversations.db)
    ├── GET  /sessions        → セッション一覧
    ├── DELETE /sessions/{id} → CASCADE DELETE (turns も削除)
    │
    └── POST /query
         │  session_id + Bearer token
         ▼
    ┌─────────────────────────────────────┐
    │ MEDPipeline                         │
    │                                     │
    │  Step 0: get_context_turns()        │
    │          → トークン上限内の履歴取得   │
    │                                     │
    │  Step 1-5: RAG + LLM 処理           │
    │                                     │
    │  Final: add_turn(user) + add_turn(  │
    │         assistant)                  │
    │         → FAISS 自動登録            │
    └─────────────────────────────────────┘
```

---

## モジュール構成

### 1. 認証モジュール (`src/auth/`)

| ファイル | 責務 |
|---------|------|
| `schema.py` | User / TokenPayload / LoginRequest / RegisterRequest / TestTokenRequest / TokenResponse |
| `store.py` | UserStore — aiosqlite CRUD (users テーブル) |
| `service.py` | AuthService — bcrypt ハッシュ + python-jose JWT |
| `deps.py` | FastAPI 依存注入 (get_current_user / get_optional_user / get_current_admin / require_localhost) |

#### User モデル

```python
@dataclass
class User:
    user_id: str               # UUID
    username: str              # UNIQUE
    hashed_password: str|None  # None = テストユーザー
    is_test: bool = False
    is_admin: bool = False
    is_active: bool = True
    created_at: datetime
    last_login: datetime|None
```

#### JWT ペイロード

```python
@dataclass
class TokenPayload:
    sub: str        # user_id
    username: str
    is_admin: bool = False
    is_test: bool = False
```

#### 認証フロー

```
通常ユーザー:
  POST /auth/register { username, password }
    → bcrypt ハッシュ → users.db 保存 → User 返却
  POST /auth/login { username, password }
    → bcrypt 検証 → JWT 生成 → TokenResponse 返却

テストユーザー:
  scripts/seed_test_users.py → test_alice, test_bob, test_system を登録
  POST /auth/token/test { username }  ← localhost 限定
    → パスワード不要で JWT 返却
    → 本番: allow_test_token: false で無効化
```

#### FastAPI 依存注入パターン

```python
# 必須認証
@app.get("/sessions")
async def list_sessions(user: User = Depends(get_current_user)):
    ...

# 任意認証（未認証でも動作）
@app.post("/query")
async def query(user: User | None = Depends(get_optional_user)):
    user_id = user.user_id if user else "default"
    ...

# 管理者限定
@app.get("/admin/users")
async def list_users(admin: User = Depends(get_current_admin)):
    ...

# localhost 限定
@app.post("/auth/token/test", dependencies=[Depends(require_localhost)])
async def issue_test_token(...):
    ...
```

---

### 2. 会話履歴モジュール (`src/conversation/`)

| ファイル | 責務 |
|---------|------|
| `schema.py` | Session / Turn データクラス |
| `store.py` | ConversationStore — aiosqlite CRUD (sessions + turns テーブル) |
| `manager.py` | ConversationManager — ビジネスロジック (セッション上限 / トークン窓 / FAISS 登録) |

#### DB スキーマ

```sql
-- sessions テーブル
CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    domain      TEXT NOT NULL DEFAULT 'general',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    turn_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_sessions_user ON sessions(user_id, updated_at DESC);

-- turns テーブル（CASCADE DELETE で sessions 連動）
CREATE TABLE turns (
    turn_id       TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role          TEXT NOT NULL,           -- 'user' | 'assistant'
    content       TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    token_count   INTEGER NOT NULL DEFAULT 0,
    provider      TEXT DEFAULT '',
    model         TEXT DEFAULT '',
    faiss_doc_id  TEXT,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0
);
CREATE INDEX idx_turns_session ON turns(session_id, timestamp ASC);
```

#### Turn モデル

```python
@dataclass
class Turn:
    turn_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    token_count: int = 0      # 文字数 // 4 推定
    provider: str = ""
    model: str = ""
    faiss_doc_id: str|None = None  # FAISS 登録済み doc_id
    input_tokens: int = 0
    output_tokens: int = 0
```

#### Session モデル

```python
@dataclass
class Session:
    session_id: str
    user_id: str
    title: str       # 自動生成: クエリ先頭20文字 + "…"
    domain: str
    created_at: datetime
    updated_at: datetime
    turn_count: int = 0
```

---

### 3. 主要ロジック

#### セッション上限自動削除

ユーザー毎のセッション上限（デフォルト50）を超えた場合、`updated_at` が古いものを CASCADE DELETE：

```sql
DELETE FROM sessions
WHERE session_id IN (
    SELECT session_id FROM sessions
    WHERE user_id = ?
    ORDER BY updated_at ASC
    LIMIT ?
)
```

#### トークンウィンドウ制御

LLM コンテキストに渡すターン数を `context_window_tokens`（デフォルト 2048）以内に制限：

```python
async def get_recent_turns_within_tokens(session_id, max_tokens):
    # 新しい順に取得し、token_count の合計が max_tokens を超えるまで追加
    # 最後に時系列順（古い→新しい）に反転して返す
```

#### FAISS 自動登録

assistant ターンをユーザー専用 FAISS に自動保存：

```python
async def save_to_user_faiss(turn, user_id, get_user_mm):
    mm = get_user_mm(user_id)  # user_id → MemoryManager
    doc_id = await mm.add_from_text(content=turn.content, ...)
    await store.update_turn_faiss_doc_id(turn.turn_id, doc_id)
```

パイプライン側では `asyncio.ensure_future()` でバックグラウンド実行し、レスポンス返却を遅延させない。

---

### 4. 設定 (`configs/default.yaml`)

```yaml
auth:
  users_db_path: "data/users.db"
  jwt_secret_key: "change-me-in-production"  # 本番: 環境変数 JWT_SECRET_KEY で上書き
  jwt_algorithm: "HS256"
  access_token_expire_days: 7
  allow_test_token: true         # 本番: false
  test_token_localhost_only: true
  allow_registration: true

conversation:
  db_path: "data/conversations.db"
  context_window_tokens: 2048
  auto_save_to_faiss: true
  max_sessions_per_user: 50
  max_turns_per_session: 500
  user_faiss_base: "data/faiss_indices/user_{user_id}"
  knowledge_faiss_base: "data/faiss_indices/knowledge"
```

---

### 5. API エンドポイント一覧

| メソッド | パス | 認証 | 説明 |
|---------|------|------|------|
| `POST` | `/auth/register` | なし | ユーザー登録 |
| `POST` | `/auth/login` | なし | ログイン → JWT |
| `POST` | `/auth/token/test` | localhost | テストトークン発行 |
| `GET` | `/auth/me` | Bearer | 自分のプロフィール |
| `GET` | `/sessions` | Bearer | セッション一覧 |
| `POST` | `/sessions` | Bearer | セッション作成 |
| `DELETE` | `/sessions/{id}` | Bearer | セッション削除 (CASCADE) |
| `GET` | `/sessions/{id}/turns` | Bearer | ターン一覧 |
| `GET` | `/admin/users` | admin | ユーザー一覧 |
| `DELETE` | `/admin/users/{id}` | admin | ユーザー削除 |
| `PATCH` | `/admin/users/{id}/activate` | admin | アクティブ状態変更 |
| `POST` | `/query` | optional | クエリ (session_id 付き) |

---

### 6. GUI 統合 (`src/gui/tabs/chat.py`)

Gradio Chat タブに以下を追加：

- **セッション選択ドロップダウン**: 既存セッションをドロップダウンで切り替え
- **新規セッションボタン**: 新しい会話を開始
- **更新ボタン**: セッション一覧をリロード
- **`gr.State`**: `session_id_state` / `token_state` で JWT トークンとセッション ID を保持
- セッション切替時に `GET /sessions/{id}/turns` で履歴を復元

---

### 7. テスト

| ファイル | クラス | テスト内容 |
|---------|--------|-----------|
| `tests/unit/test_auth.py` | `TestRegister` | 通常/テストユーザー登録・重複チェック・admin |
| | `TestLogin` | 正常ログイン・パスワード不一致・テストユーザー排除 |
| | `TestTestToken` | テストトークン発行・無効化・通常ユーザー排除 |
| | `TestJWT` | トークン生成・検証・不正トークン |
| | `TestUserStore` | CRUD (save / get_by_id / get_by_username / delete / set_active / list_all) |
| `tests/unit/test_conversation.py` | `TestUtils` | _auto_title / _estimate_tokens |
| | `TestSessionCRUD` | 作成・取得・一覧・削除・ユーザー分離・上限自動削除 |
| | `TestTurnCRUD` | ターン追加・turn_count更新・トークン推定・コンテキスト窓・時系列順 |
| | `TestToMessages` | Turn → LLM messages 変換 |
| | `TestConversationStore` | get_recent_turns_within_tokens 時系列順 / CASCADE DELETE |

すべて `:memory:` SQLite を使用し外部依存なし。

---

### 8. スクリプト

| ファイル | 説明 |
|---------|------|
| `scripts/seed_test_users.py` | テストユーザー3名 (test_alice / test_bob / test_system) を登録 |

```bash
# 使い方
python scripts/seed_test_users.py
python scripts/seed_test_users.py --db data/users.db

# テストトークン取得
curl -X POST http://localhost:8000/auth/token/test \
  -H 'Content-Type: application/json' \
  -d '{"username": "test_alice"}'
```

---

## 依存ライブラリ

| パッケージ | 用途 |
|-----------|------|
| `python-jose[cryptography]` | JWT 生成・検証 (HS256) |
| `passlib[bcrypt]` | パスワードハッシュ (bcrypt) |
| `aiosqlite` | 非同期 SQLite CRUD (既存依存) |

---

## セキュリティ考慮事項

1. **JWT シークレット**: `configs/default.yaml` のデフォルト値は開発用。本番では環境変数 `JWT_SECRET_KEY` で上書き必須
2. **テストトークン**: `allow_test_token: true` は開発専用。本番では `false` に設定
3. **localhost 制限**: テストトークンエンドポイントは `require_localhost` で `127.0.0.1` / `::1` のみ許可
4. **パスワードハッシュ**: bcrypt（saltありハッシュ、タイミングセーフ比較）
5. **CASCADE DELETE**: セッション削除時にターンも自動削除（孤立レコード防止）
6. **WAL モード**: aiosqlite で WAL モード有効化（同時読み書きの安全性向上）

---

## 実装ステータス

| 項目 | 状態 |
|------|------|
| `src/auth/schema.py` | ✅ 完了 |
| `src/auth/store.py` | ✅ 完了 |
| `src/auth/service.py` | ✅ 完了 |
| `src/auth/deps.py` | ✅ 完了 |
| `src/conversation/schema.py` | ✅ 完了 |
| `src/conversation/store.py` | ✅ 完了 |
| `src/conversation/manager.py` | ✅ 完了 |
| `src/common/config.py` 拡張 | ✅ 完了 |
| `configs/default.yaml` 拡張 | ✅ 完了 |
| `src/orchestrator/pipeline.py` 統合 | ✅ 完了 |
| `src/orchestrator/server.py` エンドポイント | ✅ 完了 |
| `src/gui/tabs/chat.py` UI | ✅ 完了 |
| `scripts/seed_test_users.py` | ✅ 完了 |
| `tests/unit/test_auth.py` | ✅ 完了 |
| `tests/unit/test_conversation.py` | ✅ 完了 |

---

## 後続フェーズ（オプション）

- 🟡 ブラウザ LocalStorage による JS 側キャッシュ（サーバー API で代替済みのため低優先度）
- 🟢 OAuth2 / OpenID Connect 対応（外部 IdP 連携）
- 🟢 セッション共有機能（URL ベースの会話共有）
- 🟢 会話エクスポート（JSON / Markdown 形式）
- 🟢 `ResponseGenerator.generate()` の `conversation_history` パラメータ対応確認

#!/usr/bin/env bash
# scripts/backup_data.sh — data/ の restic バックアップ + NAS同期 + B2クラウド同期
#
# バックアップ:
#   bash scripts/backup_data.sh                    # バックアップ + NAS + B2
#   bash scripts/backup_data.sh --local-only       # NAS/B2同期なし
#   bash scripts/backup_data.sh --no-cloud         # B2同期なし（ローカル+NASのみ）
#   bash scripts/backup_data.sh --tag "milestone"  # カスタムタグ追加
#
# 一覧:
#   bash scripts/backup_data.sh --list             # スナップショット一覧（ローカル）
#   bash scripts/backup_data.sh --list --from nas   # スナップショット一覧（NAS）
#   bash scripts/backup_data.sh --list --from cloud # スナップショット一覧（B2）
#
# 復元:
#   bash scripts/backup_data.sh --restore latest                   # ローカルから data/ へ
#   bash scripts/backup_data.sh --restore latest --from nas        # NASから data/ へ
#   bash scripts/backup_data.sh --restore latest --from cloud      # B2から data/ へ
#   bash scripts/backup_data.sh --restore latest --from cloud --to nas  # B2からNASへ

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESTIC_REPO="$PROJECT_ROOT/.restic/repo"
RESTIC_PASSWORD_FILE="$PROJECT_ROOT/.restic/password.txt"
CLOUD_ENV="$PROJECT_ROOT/.restic/cloud.env"
BACKUP_SOURCE="$PROJECT_ROOT/data"
NAS_TARGET="/mnt/z/med-backup"
NAS_REPO_TMP="/tmp/restic-nas-repo"
B2_BUCKET="b2:med-backup-005a1b2c3d4e5f0"

export RESTIC_REPOSITORY="$RESTIC_REPO"
export RESTIC_PASSWORD_FILE

# B2 認証情報の読み込み
load_cloud_env() {
    if [[ -f "$CLOUD_ENV" ]]; then
        # shellcheck disable=SC1090
        source "$CLOUD_ENV"
        export B2_ACCOUNT_ID B2_ACCOUNT_KEY
        return 0
    else
        echo "[backup] ERROR: $CLOUD_ENV not found."
        return 1
    fi
}

# NAS をローカル一時ディレクトリにコピー（NAS上で直接resticは使えないため）
prepare_nas_repo() {
    if [[ ! -d "$NAS_TARGET" ]]; then
        echo "[backup] ERROR: NAS not accessible at $NAS_TARGET"
        echo "[backup] Mount NAS first: sudo mount -t drvfs Z: /mnt/z"
        return 1
    fi
    echo "[backup] Copying NAS repo to local temp..."
    mkdir -p "$NAS_REPO_TMP"
    rsync -a --no-perms --no-owner --no-group --no-times \
        "$NAS_TARGET/" "$NAS_REPO_TMP/"
    echo "[backup] NAS repo ready at $NAS_REPO_TMP"
}

# 復元元のリポジトリパスを解決
resolve_repo() {
    local from="$1"
    case "$from" in
        local)  echo "$RESTIC_REPO" ;;
        nas)    echo "$NAS_REPO_TMP" ;;
        cloud)  echo "$B2_BUCKET" ;;
        *)      echo "[backup] ERROR: unknown --from value: $from"; exit 1 ;;
    esac
}

# --- 引数解析 ---
LOCAL_ONLY=false
NO_CLOUD=false
CUSTOM_TAG=""
MODE="backup"  # backup | list | restore
FROM="local"
TO="local"     # local | nas

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-only) LOCAL_ONLY=true; shift ;;
        --no-cloud)   NO_CLOUD=true; shift ;;
        --tag)        CUSTOM_TAG="$2"; shift 2 ;;
        --list)       MODE="list"; shift ;;
        --list-cloud) MODE="list"; FROM="cloud"; shift ;;  # 後方互換
        --restore)    MODE="restore"; RESTORE_TARGET="${2:-latest}"; shift 2 ;;
        --from)       FROM="$2"; shift 2 ;;
        --to)         TO="$2"; shift 2 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- スナップショット一覧 ---
if [[ "$MODE" == "list" ]]; then
    case "$FROM" in
        local)
            restic snapshots
            ;;
        nas)
            prepare_nas_repo
            restic -r "$NAS_REPO_TMP" --password-file "$RESTIC_PASSWORD_FILE" snapshots
            rm -rf "$NAS_REPO_TMP"
            ;;
        cloud)
            if load_cloud_env; then
                restic -r "$B2_BUCKET" --password-file "$RESTIC_PASSWORD_FILE" snapshots
            fi
            ;;
    esac
    exit 0
fi

# --- 復元 ---
if [[ "$MODE" == "restore" ]]; then
    # 復元先の決定
    if [[ "$TO" == "nas" ]]; then
        RESTORE_DIR="$NAS_TARGET/restored"
        mkdir -p "$RESTORE_DIR" 2>/dev/null || {
            echo "[backup] ERROR: Cannot create $RESTORE_DIR. Is NAS mounted?"
            exit 1
        }
    else
        RESTORE_DIR="$PROJECT_ROOT"
    fi

    echo "[backup] Restore: from=$FROM, snapshot=$RESTORE_TARGET, to=$RESTORE_DIR"

    case "$FROM" in
        local)
            restic restore "$RESTORE_TARGET" --target "$RESTORE_DIR"
            ;;
        nas)
            prepare_nas_repo
            restic -r "$NAS_REPO_TMP" --password-file "$RESTIC_PASSWORD_FILE" \
                restore "$RESTORE_TARGET" --target "$RESTORE_DIR"
            rm -rf "$NAS_REPO_TMP"
            ;;
        cloud)
            if load_cloud_env; then
                restic -r "$B2_BUCKET" --password-file "$RESTIC_PASSWORD_FILE" \
                    restore "$RESTORE_TARGET" --target "$RESTORE_DIR"
            else
                exit 1
            fi
            ;;
    esac

    echo "[backup] Restore complete: $RESTORE_DIR"
    exit 0
fi

# --- バックアップ ---
# Git コミットハッシュをタグに含める
GIT_HASH="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# ドキュメント数をタグに含める（FAISSメタデータDB）
DOC_COUNT="unknown"
if command -v sqlite3 &>/dev/null && [[ -f "$BACKUP_SOURCE/metadata.db" ]]; then
    DOC_COUNT="$(sqlite3 "$BACKUP_SOURCE/metadata.db" 'SELECT COUNT(*) FROM documents' 2>/dev/null || echo 'unknown')"
fi

TAGS="--tag git:$GIT_HASH --tag docs:$DOC_COUNT"
[[ -n "$CUSTOM_TAG" ]] && TAGS="$TAGS --tag $CUSTOM_TAG"

echo "[backup] Starting restic backup..."
echo "[backup]   Source: $BACKUP_SOURCE"
echo "[backup]   Repo:   $RESTIC_REPO"
echo "[backup]   Tags:   git:$GIT_HASH docs:$DOC_COUNT ${CUSTOM_TAG:+$CUSTOM_TAG}"

# shellcheck disable=SC2086
restic backup "$BACKUP_SOURCE" $TAGS

echo "[backup] Local backup complete."

# --- NAS 同期 ---
if [[ "$LOCAL_ONLY" == "true" ]]; then
    echo "[backup] --local-only: skipping NAS and cloud sync."
    exit 0
fi

if [[ ! -d "$NAS_TARGET" ]] && ! mkdir -p "$NAS_TARGET" 2>/dev/null; then
    echo "[backup] WARNING: NAS target $NAS_TARGET not accessible. Skipping NAS sync."
    echo "[backup] Mount NAS first: sudo mount -t drvfs Z: /mnt/z"
else
    echo "[backup] Syncing to NAS: $NAS_TARGET ..."
    rsync -av --no-perms --no-owner --no-group --no-times --delete \
        "$RESTIC_REPO/" "$NAS_TARGET/"
    echo "[backup] NAS sync complete."
fi

# --- B2 クラウド同期 ---
if [[ "$NO_CLOUD" == "true" ]]; then
    echo "[backup] --no-cloud: skipping B2 sync."
else
    if load_cloud_env; then
        echo "[backup] Syncing to B2: $B2_BUCKET ..."
        restic -r "$B2_BUCKET" --password-file "$RESTIC_PASSWORD_FILE" \
            copy --from-repo "$RESTIC_REPO" --from-password-file "$RESTIC_PASSWORD_FILE" \
            2>&1 || {
            echo "[backup] WARNING: B2 sync failed. Local backup is safe."
        }
        echo "[backup] B2 sync complete."
    fi
fi

echo "[backup] Done. Local snapshots:"
restic snapshots --latest 3

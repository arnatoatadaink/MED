#!/usr/bin/env bash
# scripts/backup_data.sh — data/ の restic バックアップ + NAS同期
#
# 使い方:
#   bash scripts/backup_data.sh                    # バックアップ + NAS同期
#   bash scripts/backup_data.sh --local-only       # NAS同期なし
#   bash scripts/backup_data.sh --tag "milestone"  # カスタムタグ追加
#   bash scripts/backup_data.sh --list             # スナップショット一覧
#   bash scripts/backup_data.sh --restore latest   # 最新を data/ に復元

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESTIC_REPO="$PROJECT_ROOT/.restic/repo"
RESTIC_PASSWORD_FILE="$PROJECT_ROOT/.restic/password.txt"
BACKUP_SOURCE="$PROJECT_ROOT/data"
NAS_TARGET="/mnt/z/med-backup"

export RESTIC_REPOSITORY="$RESTIC_REPO"
export RESTIC_PASSWORD_FILE

# --- 引数解析 ---
LOCAL_ONLY=false
CUSTOM_TAG=""
MODE="backup"  # backup | list | restore

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-only) LOCAL_ONLY=true; shift ;;
        --tag) CUSTOM_TAG="$2"; shift 2 ;;
        --list) MODE="list"; shift ;;
        --restore) MODE="restore"; RESTORE_TARGET="${2:-latest}"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- スナップショット一覧 ---
if [[ "$MODE" == "list" ]]; then
    restic snapshots
    exit 0
fi

# --- 復元 ---
if [[ "$MODE" == "restore" ]]; then
    echo "[backup] Restoring snapshot '$RESTORE_TARGET' to $BACKUP_SOURCE ..."
    restic restore "$RESTORE_TARGET" --target "$PROJECT_ROOT"
    echo "[backup] Restore complete."
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
    echo "[backup] --local-only: skipping NAS sync."
    exit 0
fi

if [[ ! -d "$NAS_TARGET" ]] && ! mkdir -p "$NAS_TARGET" 2>/dev/null; then
    echo "[backup] WARNING: NAS target $NAS_TARGET not accessible. Skipping sync."
    echo "[backup] Mount NAS first: sudo mount -t drvfs Z: /mnt/z"
    exit 0
fi

echo "[backup] Syncing to NAS: $NAS_TARGET ..."
rsync -av --no-perms --no-owner --no-group --no-times --delete \
    "$RESTIC_REPO/" "$NAS_TARGET/"

echo "[backup] NAS sync complete."
echo "[backup] Done. Snapshots:"
restic snapshots --latest 3

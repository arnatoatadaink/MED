#!/usr/bin/env python3
"""3モデル比較レビュースクリプト — OpenRouter nemotron 3種を同一ドキュメントで比較

対象モデル:
  - nvidia/nemotron-3-nano-30b-a3b:free   (現デフォルト。Apr17調査で speculation 疑い)
  - nvidia/nemotron-nano-12b-v2-vl:free   (Mar31テストで overall=best)
  - nvidia/nemotron-3-super-120b-a12b:free (larger model。Apr10実績: 承認率33%・厳格)

選定基準（speculation / /nothink 影響が出やすい文書を重視）:
  A. hold + 高信頼度(conf>=0.8, quality<=0.5)  → 確信して却下 → speculationでoverride?
  B. approved + 低品質(quality<=0.65)          → 疑惑のspeculation approval
  C. needs_update + 高信頼度(conf>=0.9)         → 補足必要と確信 → モデルが補完してしまうか
  D. 短断片(<300文字) で approved              → needs_supplement を見逃すか
  E. off_domain で approved                    → relevance/domain解釈の差
  F. 高品質 approved(quality>=0.85)            → 正例 control

結果は data/compare_results.db + compare_results.csv に保存。
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.common.config import get_settings
from src.llm.gateway import LLMGateway
from src.memory.metadata_store import MetadataStore
from src.memory.schema import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 設定 ──────────────────────────────────────────────────────────────────────

MODELS = [
    # Qwen3.5-9b think/nothink 比較
    ("fastflowlm",       "qwen3.5:9b"),          # /nothink ON  (thinking disabled)
    ("fastflowlm_think", "qwen3.5:9b"),          # /nothink OFF (thinking enabled)
    # OpenRouter GPT-OSS-120B テスト
    ("openrouter", "openai/gpt-oss-120b:free"),
    # Phase 1 ローカル並列テスト（必要時に有効化）
    # ("lmstudio",         "gemma-4-26b-a4b-it@q8_0"),
    # Phase 2: OpenRouter その他モデル (後回し)
    # ("openrouter", "nvidia/nemotron-3-nano-30b-a3b:free"),
    # ("openrouter", "nvidia/nemotron-nano-12b-v2-vl:free"),
    # ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
    # ("openrouter", "google/gemma-4-26b-a4b-it:free"),
]

# プロバイダ別レートリミット（秒）
# ローカルプロバイダは間隔不要。openrouter は RPM=1 で 62秒。
PROVIDER_INTERVAL: dict[str, int] = {
    "openrouter":      62,
    "fastflowlm":       1,
    "fastflowlm_think": 1,
    "lmstudio":         1,
}

# ローカルプロバイダは asyncio で並列実行
PARALLEL_PROVIDERS: frozenset[str] = frozenset({"fastflowlm", "fastflowlm_think", "lmstudio"})

DOCS_PER_CATEGORY = 2  # カテゴリあたり選出件数
FRAGMENT_TEST_N   = 5  # fragment テスト: ランダム選出件数

# thinking 制御は llm_config.local.yaml の extra_params で制御
# - fastflowlm: enable_thinking=false
# - fastflowlm_think: enable_thinking=true

DB_PATH    = _ROOT / "data" / "metadata.db"
RESULT_DB  = _ROOT / "data" / "compare_results.db"
RESULT_CSV = _ROOT / "data" / "compare_results.csv"

# ── プロンプト（reviewer.py と同一） ─────────────────────────────────────────

_REVIEW_SYSTEM = """\
You are a quality reviewer for a technical knowledge base.
Evaluate the given document and respond with ONLY valid JSON:
{
  "quality_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "approved": true/false,
  "needs_supplement": true/false,
  "reason": "brief explanation"
}

Quality criteria:
- Accuracy: Is the information correct?
- Completeness: Is it self-contained and useful?
- Clarity: Is it clear and well-written?
- Relevance: Is it relevant for technical learning?

Note on domain_flag:
- on_domain: CS/ML content. Apply standard quality criteria.
- off_domain: Non-CS/ML field (physics, math, etc.). This content is intentionally
  retained for associative memory diversity. Approve if the document is high-quality
  within its own field, even if not directly CS/ML relevant. Lower the relevance
  weight and focus on accuracy and clarity instead.

Set needs_supplement=true if the document meets ANY of these conditions:
1. Fragment / incomplete: truncated mid-sentence, missing context to be understood
   standalone, or is clearly a partial excerpt needing surrounding content.
2. Thin / shallow: fewer than ~3 meaningful sentences of substance, only contains
   a title/header/install command with no explanation, or is a navigation/UI
   description with no actual knowledge content.

When needs_supplement=true, set approved=false regardless of quality_score.
Approve if quality_score >= 0.6 AND needs_supplement=false."""

_REVIEW_SYSTEM_THINK_PAYLOAD = """\
You are a quality reviewer for a technical knowledge base.
Evaluate the given document and respond with ONLY valid JSON:
{
  "quality_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "approved": true/false,
  "needs_supplement": true/false,
  "reason": "brief explanation"
}

Quality criteria:
- Accuracy: Is the information correct?
- Completeness: Is it self-contained and useful?
- Clarity: Is it clear and well-written?
- Relevance: Is it relevant for technical learning?

Note on domain_flag:
- on_domain: CS/ML content. Apply standard quality criteria.
- off_domain: Non-CS/ML field (physics, math, etc.). This content is intentionally
  retained for associative memory diversity. Approve if the document is high-quality
  within its own field, even if not directly CS/ML relevant. Lower the relevance
  weight and focus on accuracy and clarity instead.

Set needs_supplement=true if the document meets ANY of these conditions:
1. Fragment / incomplete: truncated mid-sentence, missing context to be understood
   standalone, or is clearly a partial excerpt needing surrounding content.
2. Thin / shallow: fewer than ~3 meaningful sentences of substance, only contains
   a title/header/install command with no explanation, or is a navigation/UI
   description with no actual knowledge content.

When needs_supplement=true, set approved=false regardless of quality_score.
Approve if quality_score >= 0.6 AND needs_supplement=false."""

_REVIEW_PROMPT = """\
Document metadata:
- content_type: {content_type}
- categories: {categories}
- domain_flag: {domain_flag}

Document text:
{text}"""

# ── データクラス ──────────────────────────────────────────────────────────────

@dataclass
class TestDoc:
    doc_id: str
    category: str          # A〜F
    content: str
    content_type: str
    categories: str
    domain_flag: str
    original_status: str   # 元のreview_status
    original_quality: float
    original_confidence: float

@dataclass
class ReviewRecord:
    doc_id: str
    category: str
    provider: str
    model: str
    quality_score: float
    confidence: float
    approved: bool
    needs_supplement: bool
    reason: str
    raw_response: str
    elapsed_s: float
    error: str = ""

# ── DB 初期化 ─────────────────────────────────────────────────────────────────

def init_result_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at      TEXT,
            doc_id      TEXT,
            category    TEXT,
            provider    TEXT,
            model       TEXT,
            quality     REAL,
            confidence  REAL,
            approved    INTEGER,
            needs_supp  INTEGER,
            reason      TEXT,
            raw         TEXT,
            elapsed_s   REAL,
            error       TEXT,
            orig_status TEXT,
            orig_quality REAL,
            orig_conf   REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_docs (
            run_at      TEXT,
            doc_id      TEXT,
            category    TEXT,
            content     TEXT,
            orig_status TEXT,
            orig_quality REAL,
            orig_conf   REAL
        )
    """)
    conn.commit()
    return conn

# ── テスト文書の選出 ──────────────────────────────────────────────────────────

def select_test_docs(n: int = DOCS_PER_CATEGORY) -> list[TestDoc]:
    conn = sqlite3.connect(str(DB_PATH))
    docs: list[TestDoc] = []

    def _fetch(query: str, params: tuple, category: str) -> list[TestDoc]:
        rows = conn.execute(query, params).fetchall()
        random.shuffle(rows)
        result = []
        for r in rows[:n]:
            doc_id, content, status, quality, conf, extra_json = r
            extra = json.loads(extra_json or "{}")
            result.append(TestDoc(
                doc_id=doc_id,
                category=category,
                content=content,
                content_type=extra.get("content_type", "unknown"),
                categories=", ".join(extra.get("categories", [])) or "unknown",
                domain_flag=extra.get("domain_flag", "unknown"),
                original_status=status,
                original_quality=float(quality or 0),
                original_confidence=float(conf or 0),
            ))
        return result

    BASE = "SELECT id, content, review_status, teacher_quality, confidence, source_extra FROM documents"

    # A: hold + 高信頼度（モデルが確信して却下 → speculation で覆すか）
    docs += _fetch(
        BASE + " WHERE review_status='hold' AND confidence>=0.8 AND teacher_quality<=0.5"
              " AND length(content) BETWEEN 200 AND 1200",
        (), "A_hold_highconf"
    )

    # B: approved + 低品質（疑惑の speculation approval）
    docs += _fetch(
        BASE + " WHERE review_status='approved' AND teacher_quality>0 AND teacher_quality<=0.65"
              " AND length(content) BETWEEN 200 AND 1200",
        (), "B_approved_lowqual"
    )

    # C: needs_update + 高信頼度（補足が必要と確信 → 補完してしまうか）
    docs += _fetch(
        BASE + " WHERE review_status='needs_update' AND confidence>=0.9"
              " AND length(content) BETWEEN 300 AND 1200",
        (), "C_needs_update_highconf"
    )

    # D: 短断片 (<300文字) で approved（needs_supplement 見逃し候補）
    docs += _fetch(
        BASE + " WHERE review_status='approved' AND length(content)<300 AND length(content)>=80",
        (), "D_short_fragment"
    )

    # E: off_domain で approved（relevance/domain 解釈の差）
    docs += _fetch(
        BASE + " WHERE review_status='approved' AND source_extra LIKE '%off_domain%'"
              " AND length(content) BETWEEN 200 AND 1200",
        (), "E_off_domain"
    )

    # F: 高品質 approved (quality>=0.85) — 正例 control
    docs += _fetch(
        BASE + " WHERE review_status='approved' AND teacher_quality>=0.85"
              " AND length(content) BETWEEN 300 AND 1200",
        (), "F_highqual_control"
    )

    conn.close()
    logger.info(f"テスト文書選出: {len(docs)}件")
    for d in docs:
        logger.info(f"  [{d.category}] {d.doc_id[:8]}... orig={d.original_status} "
                    f"quality={d.original_quality:.2f} conf={d.original_confidence:.2f} "
                    f"len={len(d.content)}")
    return docs


def select_fragment_docs(n: int = FRAGMENT_TEST_N, seed: int = 99) -> list[TestDoc]:
    """Truncated/fragment 文書に特化した選出 — think/nothink 差分が出やすい文書。

    選出戦略:
      T1: approved かつ末尾が文末記号なし（truncate mid-sentence）
      T2: approved かつ文頭が小文字 or 接続詞（文の途中から始まる断片）
      T3: needs_update + 高信頼度（reviewer が「補足必要」と確信済み）
      T4: 末尾が "..." で終わる明示的 truncation
      T5: 短断片(100-250文字) で approved（needs_supplement を見逃した可能性）

    pool を合算してシャッフル → 重複排除 → n 件をランダム選出。
    """
    random.seed(seed)
    conn = sqlite3.connect(str(DB_PATH))

    def _fetch_pool(query: str, params: tuple, category: str, pool_limit: int = 100) -> list[TestDoc]:
        rows = conn.execute(query + f" LIMIT {pool_limit}", params).fetchall()
        result = []
        for r in rows:
            doc_id, content, status, quality, conf, extra_json = r
            extra = json.loads(extra_json or "{}")
            result.append(TestDoc(
                doc_id=doc_id,
                category=category,
                content=content,
                content_type=extra.get("content_type", "unknown"),
                categories=", ".join(extra.get("categories", [])) or "unknown",
                domain_flag=extra.get("domain_flag", "unknown"),
                original_status=status,
                original_quality=float(quality or 0),
                original_confidence=float(conf or 0),
            ))
        return result

    BASE = "SELECT id, content, review_status, teacher_quality, confidence, source_extra FROM documents"

    pool: list[TestDoc] = []

    # T1: 末尾が文末記号なし (truncated mid-sentence) — approved で憶測承認の疑い
    pool += _fetch_pool(
        BASE + """
        WHERE review_status='approved'
          AND length(content) BETWEEN 120 AND 600
          AND content NOT LIKE '%.'
          AND content NOT LIKE '%!'
          AND content NOT LIKE '%?'
          AND content NOT LIKE '%)'
          AND content NOT LIKE '%]'
          AND content NOT LIKE '%"'
          AND content NOT LIKE "%'"
          AND content NOT LIKE '%`'
        ORDER BY RANDOM()
        """, (), "T1_truncated_approved"
    )

    # T2: 文頭が小文字（前の文章の途中から始まる断片）
    pool += _fetch_pool(
        BASE + """
        WHERE review_status IN ('approved', 'needs_update')
          AND length(content) BETWEEN 150 AND 700
          AND (
            content GLOB '[a-z]*'
            OR content LIKE 'and %' OR content LIKE 'or %'
            OR content LIKE 'but %' OR content LIKE 'also %'
            OR content LIKE 'however, %' OR content LIKE 'furthermore, %'
            OR content LIKE 'additionally, %' OR content LIKE 'moreover, %'
          )
        ORDER BY RANDOM()
        """, (), "T2_midsentence_start"
    )

    # T3: needs_update + 高信頼度（補足必要と確信 → thinking モデルが補完するか）
    pool += _fetch_pool(
        BASE + """
        WHERE review_status='needs_update'
          AND confidence >= 0.85
          AND length(content) BETWEEN 200 AND 900
        ORDER BY RANDOM()
        """, (), "T3_needs_update_highconf"
    )

    # T4: 末尾が "..." "…" — 明示的な truncation marker
    pool += _fetch_pool(
        BASE + """
        WHERE (content LIKE '%...' OR content LIKE '%…')
          AND length(content) BETWEEN 100 AND 700
        ORDER BY RANDOM()
        """, (), "T4_ellipsis_end"
    )

    # T5: 短断片 (100-250文字) で approved — needs_supplement 見逃し候補
    pool += _fetch_pool(
        BASE + """
        WHERE review_status='approved'
          AND length(content) BETWEEN 100 AND 250
        ORDER BY RANDOM()
        """, (), "T5_short_approved"
    )

    conn.close()

    # 重複排除 → シャッフル → n 件選出
    seen: set[str] = set()
    deduped: list[TestDoc] = []
    random.shuffle(pool)
    for d in pool:
        if d.doc_id not in seen:
            seen.add(d.doc_id)
            deduped.append(d)

    selected = deduped[:n]
    logger.info(f"[Fragment Test] 文書選出: {len(selected)}件 (pool={len(deduped)})")
    for d in selected:
        preview = d.content[:80].replace("\n", " ")
        logger.info(f"  [{d.category}] {d.doc_id[:8]}... orig={d.original_status} "
                    f"q={d.original_quality:.2f} conf={d.original_confidence:.2f} "
                    f"len={len(d.content)} | {preview!r}")
    return selected


# ── LLM 呼び出し ─────────────────────────────────────────────────────────────

async def review_doc(
    gateway: LLMGateway,
    doc: TestDoc,
    provider: str,
    model: str,
) -> ReviewRecord:
    prompt = _REVIEW_PROMPT.format(
        content_type=doc.content_type,
        categories=doc.categories,
        domain_flag=doc.domain_flag,
        text=doc.content[:1200],
    )
    t0 = time.time()
    error = ""
    raw = ""
    quality = 0.0
    confidence = 0.0
    approved = False
    needs_supp = False
    reason = ""

    # thinking 制御は llm_config.local.yaml の extra_params で制御される
    # - fastflowlm: enable_thinking=false (thinking OFF)
    # - fastflowlm_think: enable_thinking=true (thinking ON)
    system = _REVIEW_SYSTEM

    for attempt in range(2):  # 429 時は1回リトライ
        try:
            resp = await gateway.complete(
                prompt,
                system=system,
                provider=provider,
                model=model,
                temperature=0.0,
            )
            raw = resp.content if hasattr(resp, "content") else str(resp)
            # thinking トークンの有無を確認（think/nothink 動作確認用）
            thinking = getattr(resp, "thinking_text", None)
            thinking_tokens = getattr(resp, "thinking_tokens", 0)
            logger.debug("  [%s/%s] thinking=%s tokens=%d",
                         provider, model, bool(thinking), thinking_tokens or 0)
            if thinking:
                logger.info("  [%s] THINKING detected (%d chars): %s...",
                            provider, len(thinking), thinking[:80])
            # JSON 抽出
            m = __import__("re").search(r"\{[\s\S]*\}", raw)
            if m:
                data = json.loads(m.group())
                quality    = float(data.get("quality_score", 0))
                confidence = float(data.get("confidence", 0))
                approved   = bool(data.get("approved", False))
                needs_supp = bool(data.get("needs_supplement", False))
                reason     = str(data.get("reason", ""))
            else:
                error = "JSON not found in response"
            break  # 成功
        except BaseException as e:
            error = str(e)[:200]
            is_429 = "429" in error or "rate" in error.lower()
            if attempt == 0 and is_429:
                wait = 90
                logger.warning(f"  429 [{provider}/{model}] {doc.doc_id[:8]} → {wait}s 待機後リトライ")
                await asyncio.sleep(wait)
            else:
                logger.warning(f"  エラー [{provider}/{model}] {doc.doc_id[:8]}: {e}")

    elapsed = time.time() - t0
    return ReviewRecord(
        doc_id=doc.doc_id,
        category=doc.category,
        provider=provider,
        model=model,
        quality_score=quality,
        confidence=confidence,
        approved=approved,
        needs_supplement=needs_supp,
        reason=reason,
        raw_response=raw[:500],
        elapsed_s=elapsed,
        error=error,
    )


_NEMOTRON_MODELS = [
    ("openrouter", "nvidia/nemotron-3-nano-30b-a3b:free"),
    ("openrouter", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("openrouter", "openai/gpt-oss-120b:free"),
]

# compare_think.md Section8 の固定文書 IDs
_NEMOTRON_DOC_IDS = [
    "b58fe82869a34b1f94d508cdb83ad72c",  # SO 1文断片 — THINK承認・nothink拒否
    "368e1586f05845fba2cf038ec2ac0b6c",  # arXiv abstract 薄い — Gemma承認・Qwen拒否
    "0a4c466b082f451d855a61e94e224369",  # tldr broken link — Gemma承認・Qwen拒否
    "df073d7f1e2c452482c756c058ee2af7",  # tldr broken link — system汚染承認・THINK拒否
    "f7955ccaafaa465f8a25b8543a8b563f",  # tldr broken link + placeholder — system汚染承認・THINK拒否
]


def select_nemotron_docs() -> list[TestDoc]:
    """compare_think.md Section8 の5文書を固定で選出。"""
    conn = sqlite3.connect(str(DB_PATH))
    docs: list[TestDoc] = []
    for doc_id in _NEMOTRON_DOC_IDS:
        row = conn.execute(
            "SELECT id, content, review_status, teacher_quality, confidence, source_extra "
            "FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
        if row is None:
            logger.warning("Doc not found: %s", doc_id)
            continue
        d_id, content, status, quality, conf, extra_json = row
        extra = json.loads(extra_json or "{}")
        docs.append(TestDoc(
            doc_id=d_id,
            category="nemotron_target",
            content=content,
            content_type=extra.get("content_type", "unknown"),
            categories=", ".join(extra.get("categories", [])) or "unknown",
            domain_flag=extra.get("domain_flag", "unknown"),
            original_status=status,
            original_quality=float(quality or 0),
            original_confidence=float(conf or 0),
        ))
    conn.close()
    logger.info("Nemotron 対象文書: %d件", len(docs))
    for d in docs:
        logger.info("  %s [%s] orig=%s q=%.2f len=%d | %s",
                    d.doc_id[:10], d.category, d.original_status,
                    d.original_quality, len(d.content), d.content[:60].replace("\n"," "))
    return docs


_STRICT_PERSONA = "You are a strict and uncompromising reviewer. Apply the quality criteria rigorously without giving benefit of the doubt."


def _parse_response(content: str) -> dict:
    """JSON レスポンスを抽出・パースする。"""
    import re
    m = re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return {}
    return {}


_PROMPT_VARIATIONS = {
    "baseline": _REVIEW_PROMPT,
    "with_think": "think about the following document carefully.\n\n" + _REVIEW_PROMPT,
    "with_think_lowercase": "think about the following document.\n\n" + _REVIEW_PROMPT,
    "with_analyze": "analyze the following document carefully.\n\n" + _REVIEW_PROMPT,
}


async def run_payload_think_test(gateway: LLMGateway, result_conn: sqlite3.Connection,
                                 seed: int = 99) -> None:
    """payload に 'think' キーワード を含める場合の4パターン比較。

    - baseline: 通常プロンプト
    - with_think: "think about the following document carefully."
    - with_think_lowercase: "think about the following document." (carefully なし)
    - with_analyze: "analyze the following document carefully." (think なし)

    'carefully' vs 'think' キーワードの効果を分離。
    """
    run_at = datetime.now(UTC).isoformat()
    test_docs = select_fragment_docs(n=5, seed=seed)
    if not test_docs:
        logger.error("テスト文書が見つかりません")
        return

    print(f"\n{'='*80}")
    print("Payload 'think' Keyword Test: 'think' / 'carefully' の効果分離")
    print(f"{'='*80}\n")

    all_records: list[ReviewRecord] = []
    total = len(test_docs) * len(_PROMPT_VARIATIONS)
    counter = 0

    def _save(rec: ReviewRecord, doc: TestDoc) -> None:
        result_conn.execute(
            "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_at, rec.doc_id, rec.category, rec.provider, rec.model,
             rec.quality_score, rec.confidence,
             int(rec.approved), int(rec.needs_supplement),
             rec.reason, rec.raw_response, rec.elapsed_s, rec.error,
             doc.original_status, doc.original_quality, doc.original_confidence)
        )
        result_conn.commit()

    def _print_rec(rec: ReviewRecord, idx: int, total: int, label: str) -> None:
        status = "APPROVED" if rec.approved else ("NEEDS_SUPP" if rec.needs_supplement else "HOLD")
        print(f"[{idx:2d}/{total}] {label:30s} {rec.doc_id[:8]}... "
              f"q={rec.quality_score:.2f} conf={rec.confidence:.2f} {status} ({rec.elapsed_s:.1f}s)", flush=True)
        if rec.error:
            print(f"         ERROR: {rec.error[:200]}", flush=True)

    # 4つのプロンプトバリエーションをテスト
    for var_label, prompt_template in _PROMPT_VARIATIONS.items():
        print(f"\n--- {var_label:30s} ---\n")

        for i, doc in enumerate(test_docs):
            counter += 1
            prompt = prompt_template.format(
                content_type=doc.content_type,
                categories=doc.categories,
                domain_flag=doc.domain_flag,
                text=doc.content[:1200],
            )
            t0 = time.time()
            error = ""
            raw = ""
            quality = 0.0
            confidence = 0.0
            approved = False
            needs_supp = False
            reason = ""

            try:
                resp = await gateway.complete(
                    prompt,
                    system=_REVIEW_SYSTEM,
                    provider="fastflowlm",
                    model="qwen3.5:9b",
                    temperature=0.0,
                )
                raw = resp.content if hasattr(resp, "content") else str(resp)
                parsed = _parse_response(raw)
                quality = float(parsed.get("quality_score", 0.0))
                confidence = float(parsed.get("confidence", 0.5))
                approved = bool(parsed.get("approved", False))
                needs_supp = bool(parsed.get("needs_supplement", False))
                reason = str(parsed.get("reason", ""))
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)[:150]}"
                logger.exception("Review failed for doc=%s", doc.id)

            elapsed = time.time() - t0
            rec = ReviewRecord(
                doc_id=doc.doc_id,
                category=doc.category,
                provider="fastflowlm",
                model="qwen3.5:9b",
                quality_score=quality,
                confidence=confidence,
                approved=approved,
                needs_supplement=needs_supp,
                reason=reason,
                raw_response=raw[:500],
                elapsed_s=elapsed,
                error=error,
            )
            all_records.append(rec)
            _print_rec(rec, counter, total, var_label)
            _save(rec, doc)

    # ── 比較サマリー ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Payload 'think' Keyword Test — 比較サマリー")
    print(f"{'='*80}\n")

    print(f"{'Prompt Variation':<30s} {'Avg Time':>10s} {'Approval':>10s} {'Avg Quality':>12s}")
    print("-" * 65)

    for var_label in _PROMPT_VARIATIONS.keys():
        group = [r for r in all_records if r.provider == "fastflowlm"]
        # グループを分割
        idx = list(_PROMPT_VARIATIONS.keys()).index(var_label)
        group = group[idx * len(test_docs):(idx + 1) * len(test_docs)]

        if group:
            avg_time = sum(r.elapsed_s for r in group) / len(group)
            approval = sum(1 for r in group if r.approved) / len(group) * 100
            avg_quality = sum(r.quality_score for r in group) / len(group)
            print(f"{var_label:<30s} {avg_time:>10.1f}s {approval:>9.1f}% {avg_quality:>12.2f}")

    print(f"{'='*80}\n")
    print(f"結果保存: {RESULT_DB}\n")


async def run_system_thinking_test(gateway: LLMGateway, result_conn: sqlite3.Connection,
                                   seed: int = 99) -> None:
    """system prompt に thinking 指示を含める場合と含めない場合を比較。

    4パターン:
    1. No thinking instruction + thinking OFF (fastflowlm)
    2. No thinking instruction + thinking ON (fastflowlm_think)
    3. With thinking instruction + thinking OFF (fastflowlm)
    4. With thinking instruction + thinking ON (fastflowlm_think)
    """
    run_at = datetime.now(UTC).isoformat()
    test_docs = select_fragment_docs(n=5, seed=seed)
    if not test_docs:
        logger.error("テスト文書が見つかりません")
        return

    print(f"\n{'='*80}")
    print("System Context Thinking Test: thinking指示 有無 × enable_thinking ON/OFF")
    print(f"{'='*80}\n")

    all_records: list[ReviewRecord] = []
    patterns = [
        ("No thinking", _REVIEW_SYSTEM, False),           # (label, system, has_thinking_instruction)
        ("With thinking", _REVIEW_SYSTEM_WITH_THINKING, True),
    ]
    providers = [
        ("fastflowlm", "qwen3.5:9b", False),             # (provider, model, thinking_enabled)
        ("fastflowlm_think", "qwen3.5:9b", True),
    ]

    total = len(test_docs) * len(patterns) * len(providers)
    counter = 0

    def _save(rec: ReviewRecord, doc: TestDoc) -> None:
        result_conn.execute(
            "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_at, rec.doc_id, rec.category, rec.provider, rec.model,
             rec.quality_score, rec.confidence,
             int(rec.approved), int(rec.needs_supplement),
             rec.reason, rec.raw_response, rec.elapsed_s, rec.error,
             doc.original_status, doc.original_quality, doc.original_confidence)
        )
        result_conn.commit()

    def _print_rec(rec: ReviewRecord, idx: int, total: int, context: str) -> None:
        status = "APPROVED" if rec.approved else ("NEEDS_SUPP" if rec.needs_supplement else "HOLD")
        print(f"[{idx:2d}/{total}] {context:30s} {rec.doc_id[:8]}... "
              f"q={rec.quality_score:.2f} conf={rec.confidence:.2f} {status} ({rec.elapsed_s:.1f}s)", flush=True)
        if rec.error:
            print(f"         ERROR: {rec.error[:200]}", flush=True)
        elif rec.reason:
            print(f"         {rec.reason[:100]}", flush=True)

    # 4パターンの組み合わせで実行
    for pattern_label, system_prompt, _ in patterns:
        for prov, model, _ in providers:
            thinking_label = "THINK" if prov == "fastflowlm_think" else "NOTHINK"
            print(f"\n--- {pattern_label:15s} + {thinking_label} (fastflowlm) ---\n")

            for i, doc in enumerate(test_docs):
                counter += 1
                prompt = _REVIEW_PROMPT.format(
                    content_type=doc.content_type,
                    categories=doc.categories,
                    domain_flag=doc.domain_flag,
                    text=doc.content[:1200],
                )
                t0 = time.time()
                error = ""
                raw = ""
                quality = 0.0
                confidence = 0.0
                approved = False
                needs_supp = False
                reason = ""

                try:
                    resp = await gateway.complete(
                        prompt,
                        system=system_prompt,
                        provider=prov,
                        model=model,
                        temperature=0.0,
                    )
                    raw = resp.content if hasattr(resp, "content") else str(resp)
                    parsed = _parse_response(raw)
                    quality = float(parsed.get("quality_score", 0.0))
                    confidence = float(parsed.get("confidence", 0.5))
                    approved = bool(parsed.get("approved", False))
                    needs_supp = bool(parsed.get("needs_supplement", False))
                    reason = str(parsed.get("reason", ""))
                except Exception as e:
                    error = f"{type(e).__name__}: {str(e)[:150]}"
                    logger.exception("Review failed for doc=%s", doc.id)

                elapsed = time.time() - t0
                rec = ReviewRecord(
                    doc_id=doc.doc_id,
                    category=doc.category,
                    provider=prov,
                    model=model,
                    quality_score=quality,
                    confidence=confidence,
                    approved=approved,
                    needs_supplement=needs_supp,
                    reason=reason,
                    raw_response=raw[:500],
                    elapsed_s=elapsed,
                    error=error,
                )
                all_records.append(rec)
                context = f"{pattern_label} + {thinking_label}"
                _print_rec(rec, counter, total, context)
                _save(rec, doc)

    # ── 比較サマリー ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("System Context Thinking Test — 比較サマリー")
    print(f"{'='*80}\n")

    # 4グループに分割
    no_think_off = [r for r in all_records if r.provider == "fastflowlm"][:len(test_docs)]
    no_think_on = [r for r in all_records if r.provider == "fastflowlm_think"][:len(test_docs)]
    with_think_off = [r for r in all_records if r.provider == "fastflowlm"][len(test_docs):len(test_docs)*2]
    with_think_on = [r for r in all_records if r.provider == "fastflowlm_think"][len(test_docs):len(test_docs)*2]

    # 各グループの統計
    print(f"{'Pattern':<30s} {'Avg Time':>10s} {'Approval':>10s} {'Avg Quality':>12s}")
    print("-" * 65)

    for label, group in [
        ("No thinking + OFF", no_think_off),
        ("No thinking + ON", no_think_on),
        ("With thinking + OFF", with_think_off),
        ("With thinking + ON", with_think_on),
    ]:
        if group:
            avg_time = sum(r.elapsed_s for r in group) / len(group)
            approval = sum(1 for r in group if r.approved) / len(group) * 100
            avg_quality = sum(r.quality_score for r in group) / len(group)
            print(f"{label:<30s} {avg_time:>10.1f}s {approval:>9.1f}% {avg_quality:>12.2f}")

    print(f"{'='*80}\n")
    print(f"結果保存: {RESULT_DB}\n")


async def run_nemotron_test(gateway: LLMGateway, result_conn: sqlite3.Connection,
                             strict_persona: bool = False) -> None:
    """compare_think.md Section8 の5文書で nemotron 3種を直列評価。

    OpenRouter RPM=1 のため 62秒間隔。3モデル × 5文書 = 15リクエスト ≈ 15分。
    正解は全件 NEEDS_SUPP。何件を APPROVED にするかで speculation 傾向を定量化。
    strict_persona=True の場合: system に厳格レビュアーペルソナを付与。
    """
    run_at = datetime.now(UTC).isoformat()
    test_docs = select_nemotron_docs()
    if not test_docs:
        logger.error("対象文書が見つかりません")
        return

    persona_label = "strict persona あり" if strict_persona else "ベースライン（前回と同条件）"
    total = len(test_docs) * len(_NEMOTRON_MODELS)
    print(f"\n{'='*70}")
    print(f"Nemotron 3種 Speculation テスト — {persona_label}")
    print(f"文書: {len(test_docs)}件 × {len(_NEMOTRON_MODELS)}モデル = {total}リクエスト")
    print(f"予想時間: {total * 62 // 60}分 (OpenRouter 62秒間隔)")
    if strict_persona:
        print(f"Persona: \"{_STRICT_PERSONA}\"")
    print(f"正解: 全5件 NEEDS_SUPP (broken link / thin / single-sentence fragment)")
    print(f"{'='*70}\n")

    all_records: list[ReviewRecord] = []

    def _save(rec: ReviewRecord, doc: TestDoc) -> None:
        result_conn.execute(
            "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_at, rec.doc_id, rec.category, rec.provider, rec.model,
             rec.quality_score, rec.confidence,
             int(rec.approved), int(rec.needs_supplement),
             rec.reason, rec.raw_response, rec.elapsed_s, rec.error,
             doc.original_status, doc.original_quality, doc.original_confidence)
        )
        result_conn.commit()

    # strict_persona: system の先頭にペルソナ行を追加
    # review_doc は _REVIEW_SYSTEM をモジュール変数で参照するため一時上書き
    global _REVIEW_SYSTEM
    _orig_system = _REVIEW_SYSTEM
    if strict_persona:
        _REVIEW_SYSTEM = _STRICT_PERSONA + "\n\n" + _REVIEW_SYSTEM

    idx = 0
    for provider, model in _NEMOTRON_MODELS:
        short = model.split("/")[-1]
        print(f"\n--- {short} ---")
        for doc in test_docs:
            idx += 1
            rec = await review_doc(gateway, doc, provider, model)
            all_records.append(rec)
            _save(rec, doc)
            verdict = "APPROVED  " if rec.approved else ("NEEDS_SUPP" if rec.needs_supplement else "HOLD      ")
            correct = "✓" if not rec.approved else "✗ SPECULATION?"
            print(f"[{idx:2d}/{total}] {doc.doc_id[:8]}... q={rec.quality_score:.2f} "
                  f"conf={rec.confidence:.2f} {verdict} {correct} ({rec.elapsed_s:.1f}s)", flush=True)
            if rec.error:
                print(f"        ERROR: {rec.error[:150]}", flush=True)
            else:
                print(f"        {rec.reason[:120]}", flush=True)
            if idx < total:
                interval = PROVIDER_INTERVAL.get(provider, 62)
                if interval > 5:
                    print(f"        待機 {interval}秒...", flush=True)
                await asyncio.sleep(interval)

    # _REVIEW_SYSTEM を元に戻す
    _REVIEW_SYSTEM = _orig_system

    # ── サマリー ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Nemotron 3種 — Speculation サマリー [{persona_label}]（正解: 全件 NEEDS_SUPP）")
    print(f"{'='*70}")
    print(f"{'モデル':<42} {'承認率':>6} {'avg q':>6} {'誤承認数':>8}")
    print("-" * 65)
    for provider, model in _NEMOTRON_MODELS:
        recs = [r for r in all_records if r.model == model and not r.error]
        if not recs:
            continue
        app_rate = sum(1 for r in recs if r.approved) / len(recs) * 100
        avg_q = sum(r.quality_score for r in recs) / len(recs)
        false_app = sum(1 for r in recs if r.approved)
        short = model.split("/")[-1][:40]
        print(f"{short:<42} {app_rate:5.1f}% {avg_q:6.3f} {false_app:8d}/5")

    print(f"\n{'文書別':}")
    print(f"{'doc_id':<12}", end="")
    for _, model in _NEMOTRON_MODELS:
        print(f"  {model.split('/')[-1][:20]:>22}", end="")
    print()
    for doc in test_docs:
        print(f"{doc.doc_id[:10]:<12}", end="")
        for _, model in _NEMOTRON_MODELS:
            recs = [r for r in all_records if r.doc_id == doc.doc_id and r.model == model]
            if recs and not recs[0].error:
                r = recs[0]
                v = "APPROVED" if r.approved else "NEEDS_SUPP"
                print(f"  {v+' q='+str(round(r.quality_score,2)):>22}", end="")
            else:
                print(f"  {'ERROR':>22}", end="")
        print()
    print(f"{'='*70}\n")
    print(f"結果保存: {RESULT_DB}")


async def run_interleaved_test(gateway: LLMGateway, result_conn: sqlite3.Connection,
                               seed: int = 99) -> None:
    """think → nothink を文書ごとに交互に直列実行。

    各文書について THINK → nothink(user prefix) の順で評価し、
    nothink の /nothink が次の THINK リクエストに伝播（メモリ汚染）するか確認する。

    実行順: think[1] → nothink[1] → think[2] → nothink[2] → ... → think[n] → nothink[n]
    """
    run_at = datetime.now(UTC).isoformat()
    test_docs = select_fragment_docs(n=5, seed=seed)
    if not test_docs:
        logger.error("テスト文書が見つかりません")
        return

    print(f"\n{'='*70}")
    print("Interleaved Test: think[i] → nothink[i] per document")
    print(f"文書: {len(test_docs)}件 / /nothink 配置: user メッセージ先頭（新実装）")
    print(f"{'='*70}\n")

    all_records: list[ReviewRecord] = []
    total = len(test_docs) * 2
    step = 0

    def _save(rec: ReviewRecord, doc: TestDoc) -> None:
        result_conn.execute(
            "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_at, rec.doc_id, rec.category, rec.provider, rec.model,
             rec.quality_score, rec.confidence,
             int(rec.approved), int(rec.needs_supplement),
             rec.reason, rec.raw_response, rec.elapsed_s, rec.error,
             doc.original_status, doc.original_quality, doc.original_confidence)
        )
        result_conn.commit()

    def _print_rec(rec: ReviewRecord, idx: int, label: str) -> None:
        think_tag = f"[{label:<8}]"
        status = "APPROVED  " if rec.approved else ("NEEDS_SUPP" if rec.needs_supplement else "HOLD      ")
        print(f"[{idx:2d}/{total}] {think_tag} [{rec.category}] {rec.doc_id[:8]}... "
              f"q={rec.quality_score:.2f} conf={rec.confidence:.2f} {status} ({rec.elapsed_s:.1f}s)", flush=True)
        if rec.error:
            print(f"           ERROR: {rec.error[:200]}", flush=True)
        else:
            print(f"           reason: {rec.reason}", flush=True)

    think_recs:   list[ReviewRecord] = []
    nothink_recs: list[ReviewRecord] = []

    for doc in test_docs:
        # THINK
        step += 1
        print(f"--- doc {step}/{len(test_docs)}: {doc.doc_id[:8]}... [{doc.category}] len={len(doc.content)} ---")
        rec_think = await review_doc(gateway, doc, "fastflowlm_think", "qwen3.5:9b")
        all_records.append(rec_think)
        think_recs.append(rec_think)
        _print_rec(rec_think, step * 2 - 1, "THINK")
        _save(rec_think, doc)

        # nothink (user prefix)
        rec_nothink = await review_doc(gateway, doc, "fastflowlm", "qwen3.5:9b")
        all_records.append(rec_nothink)
        nothink_recs.append(rec_nothink)
        _print_rec(rec_nothink, step * 2, "nothink")
        _save(rec_nothink, doc)
        print()

    # ── 比較サマリー ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Interleaved Test — THINK vs nothink(user) 処理時間・判定比較")
    print(f"{'='*70}")
    print(f"{'文書':>10}  {'THINK time':>12}  {'nothink time':>12}  {'判定一致':>8}  {'quality差':>10}")
    print("-" * 70)
    for tr, nr in zip(think_recs, nothink_recs):
        match = (tr.approved == nr.approved and tr.needs_supplement == nr.needs_supplement)
        qdiff = tr.quality_score - nr.quality_score
        print(f"{tr.doc_id[:10]}  {tr.elapsed_s:>12.1f}s  {nr.elapsed_s:>12.1f}s  "
              f"{'✓' if match else '✗':>8}  {qdiff:>+10.2f}")

    avg_think   = sum(r.elapsed_s for r in think_recs)   / len(think_recs)
    avg_nothink = sum(r.elapsed_s for r in nothink_recs) / len(nothink_recs)
    think_app   = sum(1 for r in think_recs   if r.approved) / len(think_recs)   * 100
    nothink_app = sum(1 for r in nothink_recs if r.approved) / len(nothink_recs) * 100
    print("-" * 70)
    print(f"{'平均':>10}  {avg_think:>12.1f}s  {avg_nothink:>12.1f}s")
    print(f"{'承認率':>10}  {think_app:>11.1f}%  {nothink_app:>11.1f}%")
    print(f"\n汚染判定: THINK の処理時間が nothink 直後に短縮していれば thinking 抑制が伝播している可能性")
    for i, (tr, nr) in enumerate(zip(think_recs, nothink_recs)):
        prev_think = think_recs[i-1].elapsed_s if i > 0 else None
        diff = f"(前回比 {tr.elapsed_s - prev_think:+.1f}s)" if prev_think else "(初回)"
        print(f"  think[{i+1}] {tr.elapsed_s:.1f}s {diff}")
    print(f"{'='*70}\n")
    print(f"結果保存: {RESULT_DB}")


# ── メイン ────────────────────────────────────────────────────────────────────

async def run_sequential_mode_test(gateway: LLMGateway, result_conn: sqlite3.Connection,
                                    seed: int = 99) -> None:
    """think → llama3.2:1b ダミー → nothink の順で直列実行。

    FastFlowLM がリクエスト間でモード状態を保持しているか確認するテスト。
    同一文書を think/nothink の両方で評価し、処理時間と判定を比較する。

    thinking 制御は llm_config.local.yaml の extra_params で行われる:
    - fastflowlm: enable_thinking=false
    - fastflowlm_think: enable_thinking=true
    """
    run_at = datetime.now(UTC).isoformat()
    test_docs = select_fragment_docs(n=5, seed=seed)
    if not test_docs:
        logger.error("テスト文書が見つかりません")
        return

    print(f"\n{'='*70}")
    print("Sequential Mode Test: think → dummy → nothink")
    print(f"文書: {len(test_docs)}件 / thinking制御: extra_params")
    print(f"{'='*70}\n")

    all_records: list[ReviewRecord] = []
    total = len(test_docs) * 2  # think + nothink

    def _save(rec: ReviewRecord, doc: TestDoc) -> None:
        result_conn.execute(
            "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_at, rec.doc_id, rec.category, rec.provider, rec.model,
             rec.quality_score, rec.confidence,
             int(rec.approved), int(rec.needs_supplement),
             rec.reason, rec.raw_response, rec.elapsed_s, rec.error,
             doc.original_status, doc.original_quality, doc.original_confidence)
        )
        result_conn.commit()

    def _print_rec(rec: ReviewRecord, idx: int, label: str) -> None:
        think_tag = "[THINK]  " if rec.provider == "fastflowlm_think" else "[nothink]"
        status = "APPROVED" if rec.approved else ("NEEDS_SUPP" if rec.needs_supplement else "HOLD")
        print(f"[{idx:2d}/{total}] {think_tag} [{rec.category}] {rec.doc_id[:8]}... "
              f"q={rec.quality_score:.2f} conf={rec.confidence:.2f} {status} ({rec.elapsed_s:.1f}s)", flush=True)
        if rec.error:
            print(f"         ERROR: {rec.error[:200]}", flush=True)
        elif rec.reason:
            print(f"         reason: {rec.reason}", flush=True)

    # ── Phase 1: THINK (fastflowlm_think) ────────────────────────────────────
    print("--- Phase 1: qwen3.5:9b [THINK] (5件 直列) ---\n")
    for i, doc in enumerate(test_docs):
        rec = await review_doc(gateway, doc, "fastflowlm_think", "qwen3.5:9b")
        all_records.append(rec)
        _print_rec(rec, i + 1, "THINK")
        _save(rec, doc)

    # ── ダミーリクエスト (llama3.2:1b → FastFlowLM) ─────────────────────────
    print(f"\n--- Dummy request: llama3.2:1b → FastFlowLM (モード状態クリア確認) ---")
    dummy_t0 = time.time()
    dummy_error = ""
    try:
        dummy_resp = await gateway.complete(
            "Reply with one word: OK",
            provider="fastflowlm",
            model="llama3.2:1b",
            temperature=0.0,
            max_tokens=10,
        )
        dummy_content = dummy_resp.content if hasattr(dummy_resp, "content") else str(dummy_resp)
        dummy_elapsed = time.time() - dummy_t0
        print(f"  dummy response ({dummy_elapsed:.1f}s): {dummy_content[:80]!r}", flush=True)
    except Exception as e:
        dummy_error = str(e)[:200]
        dummy_elapsed = time.time() - dummy_t0
        print(f"  dummy ERROR ({dummy_elapsed:.1f}s): {dummy_error}", flush=True)
    print()

    # ── Phase 2: NOTHINK (fastflowlm) ────────────────────────────────────────
    print(f"--- Phase 2: qwen3.5:9b [NOTHINK] (5件 直列) ---\n")
    for i, doc in enumerate(test_docs):
        rec = await review_doc(gateway, doc, "fastflowlm", "qwen3.5:9b")
        all_records.append(rec)
        _print_rec(rec, len(test_docs) + i + 1, "NOTHINK")
        _save(rec, doc)

    # ── 比較サマリー ─────────────────────────────────────────────────────────
    think_recs   = [r for r in all_records if r.provider == "fastflowlm_think"]
    nothink_recs = [r for r in all_records if r.provider == "fastflowlm"]

    print(f"\n{'='*70}")
    print("Sequential Mode Test — 処理時間・判定 比較")
    print(f"{'='*70}")
    print(f"{'文書':>10}  {'THINK time':>12}  {'nothink time':>12}  {'判定一致':>8}  {'quality差':>10}")
    print("-" * 70)
    for td, nr in zip(think_recs, nothink_recs):
        match = (td.approved == nr.approved and td.needs_supplement == nr.needs_supplement)
        qdiff = td.quality_score - nr.quality_score
        print(f"{td.doc_id[:10]}  {td.elapsed_s:>12.1f}s  {nr.elapsed_s:>12.1f}s  "
              f"{'✓' if match else '✗':>8}  {qdiff:>+10.2f}")

    avg_think   = sum(r.elapsed_s for r in think_recs)   / len(think_recs)   if think_recs   else 0
    avg_nothink = sum(r.elapsed_s for r in nothink_recs) / len(nothink_recs) if nothink_recs else 0
    think_app   = sum(1 for r in think_recs   if r.approved) / len(think_recs)   * 100
    nothink_app = sum(1 for r in nothink_recs if r.approved) / len(nothink_recs) * 100
    print("-" * 70)
    print(f"{'平均':>10}  {avg_think:>12.1f}s  {avg_nothink:>12.1f}s")
    print(f"{'承認率':>10}  {think_app:>11.1f}%  {nothink_app:>11.1f}%")
    print(f"{'='*70}\n")
    print(f"結果保存: {RESULT_DB}")


async def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment-test", action="store_true",
                    help="truncated/fragment 文書に特化した think vs nothink テスト")
    ap.add_argument("--sequential-mode-test", action="store_true",
                    help="think全件→llama3.2:1bダミー→nothink全件 直列実行（モード状態確認）")
    ap.add_argument("--interleaved-test", action="store_true",
                    help="think[i]→nothink[i] を文書ごとに交互実行（nothink伝播の確認）")
    ap.add_argument("--system-thinking-test", action="store_true",
                    help="system prompt に thinking指示 有無 × enable_thinking ON/OFF の4パターン比較")
    ap.add_argument("--payload-think-test", action="store_true",
                    help="payload に 'think' キーワード を含める場合の4パターン比較（'think'/'carefully' 効果分離）")
    ap.add_argument("--nemotron-test", action="store_true",
                    help="nemotron 3種 OpenRouter テスト（compare_think.md の5文書固定）")
    ap.add_argument("--strict-persona", action="store_true",
                    help="--nemotron-test と併用: system に厳格レビュアーペルソナを付与")
    ap.add_argument("--seed", type=int, default=None,
                    help="乱数シード (fragment-test 時のデフォルト: 99, 通常: 42)")
    args = ap.parse_args()

    fragment_mode   = args.fragment_test
    sequential_mode = args.sequential_mode_test
    seed = args.seed if args.seed is not None else (99 if (fragment_mode or sequential_mode) else 42)
    random.seed(seed)
    run_at = datetime.now(UTC).isoformat()

    config = get_settings()
    gateway = LLMGateway(config)
    result_conn = init_result_db(RESULT_DB)

    if sequential_mode:
        await run_sequential_mode_test(gateway, result_conn, seed=seed)
        result_conn.close()
        return

    if args.interleaved_test:
        await run_interleaved_test(gateway, result_conn, seed=seed)
        result_conn.close()
        return

    if args.system_thinking_test:
        await run_system_thinking_test(gateway, result_conn, seed=seed)
        result_conn.close()
        return

    if args.payload_think_test:
        await run_payload_think_test(gateway, result_conn, seed=seed)
        result_conn.close()
        return

    if args.nemotron_test:
        await run_nemotron_test(gateway, result_conn, strict_persona=args.strict_persona)
        result_conn.close()
        return

    test_docs = select_fragment_docs(seed=seed) if fragment_mode else select_test_docs()
    if not test_docs:
        logger.error("テスト文書が見つかりません")
        return

    # test_docs を DB に記録
    result_conn.executemany(
        "INSERT INTO test_docs VALUES (?,?,?,?,?,?,?)",
        [(run_at, d.doc_id, d.category, d.content[:300],
          d.original_status, d.original_quality, d.original_confidence)
         for d in test_docs]
    )
    result_conn.commit()

    all_records: list[ReviewRecord] = []
    total = len(test_docs) * len(MODELS)

    # プロバイダをローカル並列 / OpenRouter 直列に分類
    local_models  = [(p, m) for p, m in MODELS if p in PARALLEL_PROVIDERS]
    remote_models = [(p, m) for p, m in MODELS if p not in PARALLEL_PROVIDERS]

    est_remote = len(remote_models) * len(test_docs) * 62 // 60
    print(f"\n{'='*70}")
    print(f"比較開始: {len(test_docs)}件 × {len(MODELS)}モデル = {total}リクエスト")
    print(f"  ローカル並列: {[m.split('/')[-1].split('@')[0] for _,m in local_models]}")
    print(f"  OpenRouter直列: {[m.split('/')[-1] for _,m in remote_models]}")
    print(f"想定所要時間: ローカル数分 + OpenRouter約{est_remote}分")
    print(f"{'='*70}\n")

    def _save(rec: ReviewRecord, doc: TestDoc) -> None:
        result_conn.execute(
            "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_at, rec.doc_id, rec.category, rec.provider, rec.model,
             rec.quality_score, rec.confidence,
             int(rec.approved), int(rec.needs_supplement),
             rec.reason, rec.raw_response, rec.elapsed_s, rec.error,
             doc.original_status, doc.original_quality, doc.original_confidence)
        )
        result_conn.commit()

    def _print_rec(rec: ReviewRecord, idx: int, total: int) -> None:
        short = rec.model.split("/")[-1].split("@")[0][:25]
        think_tag = "[THINK]" if rec.provider == "fastflowlm_think" else "[nothink]" if rec.provider == "fastflowlm" else ""
        status = "APPROVED" if rec.approved else ("NEEDS_SUPP" if rec.needs_supplement else "HOLD")
        print(f"[{idx:2d}/{total}] [{rec.category}] {rec.doc_id[:8]}... → {short} {think_tag}", flush=True)
        print(f"       quality={rec.quality_score:.2f} conf={rec.confidence:.2f} {status} ({rec.elapsed_s:.1f}s)", flush=True)
        if rec.error:
            print(f"       ERROR: {rec.error[:200]}", flush=True)
        elif rec.reason:
            # fragment テストでは reason を全文表示（憶測補完の検出のため）
            print(f"       reason: {rec.reason}", flush=True)

    # ── ローカル並列処理 ───────────────────────────────────────────────────────
    if local_models:
        print(f"--- ローカル並列実行 ({len(local_models)}モデル × {len(test_docs)}件) ---\n")
        idx = 0
        for doc in test_docs:
            tasks = [review_doc(gateway, doc, p, m) for p, m in local_models]
            results = await asyncio.gather(*tasks)
            for rec in results:
                idx += 1
                all_records.append(rec)
                _print_rec(rec, idx, total)
                _save(rec, doc)

    # ── OpenRouter 直列処理 ────────────────────────────────────────────────────
    if remote_models:
        print(f"\n--- OpenRouter 直列実行 ({len(remote_models)}モデル × {len(test_docs)}件) ---\n")
        done = len(local_models) * len(test_docs)
        for doc in test_docs:
            for provider, model in remote_models:
                done += 1
                rec = await review_doc(gateway, doc, provider, model)
                all_records.append(rec)
                _print_rec(rec, done, total)
                _save(rec, doc)
                if done < total:
                    interval = PROVIDER_INTERVAL.get(provider, 62)
                    if interval > 5:
                        print(f"       待機 {interval}秒...", flush=True)
                    await asyncio.sleep(interval)

    # ── サマリー表示 ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("結果サマリー")
    print(f"{'='*70}")
    print(f"{'モデル':<40} {'承認率':>6} {'avg quality':>11} {'avg conf':>8} {'errors':>6}")
    print("-" * 70)

    for provider, model in MODELS:
        recs = [r for r in all_records if r.model == model]
        if not recs:
            continue
        valid = [r for r in recs if not r.error]
        approve_rate = sum(1 for r in valid if r.approved) / len(valid) * 100 if valid else 0
        avg_q = sum(r.quality_score for r in valid) / len(valid) if valid else 0
        avg_c = sum(r.confidence for r in valid) / len(valid) if valid else 0
        short = model.split("/")[-1][:38]
        print(f"{short:<40} {approve_rate:5.1f}% {avg_q:11.3f} {avg_c:8.3f} {len(recs)-len(valid):6d}")

    print(f"\n{'カテゴリ別':}")
    print(f"{'カテゴリ':<28} ", end="")
    for _, model in MODELS:
        print(f"{model.split('/')[-1][:18]:>20}", end="")
    print()
    print("-" * (28 + 20 * len(MODELS)))

    categories = sorted(set(r.category for r in all_records))
    for cat in categories:
        print(f"{cat:<28}", end="")
        for provider, model in MODELS:
            recs = [r for r in all_records if r.category == cat and r.model == model]
            valid = [r for r in recs if not r.error]
            if valid:
                n_app = sum(1 for r in valid if r.approved)
                label = f"{n_app}/{len(valid)} ({sum(r.quality_score for r in valid)/len(valid):.2f})"
            else:
                label = "ERROR"
            print(f"{label:>20}", end="")
        print()

    # ── CSV 保存 ──────────────────────────────────────────────────────────────
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_at", "doc_id", "category", "orig_status", "orig_quality", "orig_conf",
            "provider", "model", "quality", "confidence", "approved", "needs_supplement",
            "reason", "elapsed_s", "error"
        ])
        for rec in all_records:
            doc = next(d for d in test_docs if d.doc_id == rec.doc_id)
            writer.writerow([
                run_at, rec.doc_id, rec.category,
                doc.original_status, doc.original_quality, doc.original_confidence,
                rec.provider, rec.model,
                rec.quality_score, rec.confidence, rec.approved, rec.needs_supplement,
                rec.reason[:200], round(rec.elapsed_s, 1), rec.error
            ])

    result_conn.close()
    print(f"\n結果保存: {RESULT_DB}")
    print(f"CSV保存:  {RESULT_CSV}")
    print(f"合計リクエスト: {total}件 (エラー: {sum(1 for r in all_records if r.error)}件)")


if __name__ == "__main__":
    asyncio.run(main())

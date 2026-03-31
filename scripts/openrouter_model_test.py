"""OpenRouter free モデル評価スクリプト

Usage:
  # DB初期化 & モデル一覧取得
  poetry run python scripts/openrouter_model_test.py --init

  # 未テストモデルをテスト
  poetry run python scripts/openrouter_model_test.py --test

  # 結果一覧表示
  poetry run python scripts/openrouter_model_test.py --show

  # 特定モデルのログ表示
  poetry run python scripts/openrouter_model_test.py --log <model_id>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（scripts/ から実行した場合に src を解決）
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import re
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/openrouter_test.db")

REVIEW_SYSTEM = """Evaluate the given document and respond with ONLY valid JSON:
{
  "quality_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "approved": true/false,
  "needs_supplement": true/false,
  "reason": "brief explanation"
}
Approve if quality_score >= 0.6 AND needs_supplement=false."""

REVIEW_PROMPT = """Document:
title: How to implement FAISS IndexIVFPQ for billion-scale search
content: FAISS IVF (Inverted File Index) with PQ (Product Quantization) is designed for billion-scale approximate nearest neighbor search. The key parameters are nlist (number of clusters, typically sqrt(N)) and m (number of sub-quantizers). Training requires a representative sample: index.train(sample_vectors). After training, add vectors and search with nprobe controlling the accuracy-speed tradeoff. Higher nprobe = better recall but slower search.
source: stackoverflow"""

# 除外モデル（音楽・画像・汎用ルーター等）
EXCLUDE_IDS = {
    "google/lyria-3-clip-preview",
    "google/lyria-3-pro-preview",
    "openrouter/free",
}

# 既テスト結果（前セッションの実測値）
PREVIOUS_RESULTS = {
    "nvidia/nemotron-3-super-120b-a12b:free": {
        "speed_s": 13.4, "quality_score": 0.72, "confidence": 0.81,
        "approved": True, "needs_supplement": False,
        "reason": "The document provides a clear, accurate overview of FAISS IndexIVFPQ parameters and usage, sufficient for a high-level understanding.",
        "raw_response": '{"quality_score": 0.72, "confidence": 0.81, "approved": true, "needs_supplement": false, "reason": "The document provides a clear, accurate overview of FAISS IndexIVFPQ parameters and usage, sufficient for a high-level understanding."}',
        "reason_quality": "balanced",
        "judgment_criteria": "lenient",
        "overall": "good",
        "notes": "Brief but nuanced. Accepts high-level docs.",
    },
    "nvidia/nemotron-nano-12b-v2-vl:free": {
        "speed_s": 10.2, "quality_score": 0.70, "confidence": 0.90,
        "approved": True, "needs_supplement": False,
        "reason": "The document clearly explains FAISS IVFPQ's purpose, key parameters (nlist, m), training process, and search tradeoffs. However, it lacks depth on practical implementation steps (e.g., code examples, parameter tuning guidelines) which could enhance usability for developers.",
        "raw_response": '{"quality_score": 0.7, "confidence": 0.9, "approved": true, "needs_supplement": false, "reason": "The document clearly explains FAISS IVFPQ\'s purpose, key parameters (nlist, m), training process, and search tradeoffs. However, it lacks depth on practical implementation steps (e.g., code examples, parameter tuning guidelines) which could enhance usability for developers."}',
        "reason_quality": "detailed",
        "judgment_criteria": "balanced",
        "overall": "best",
        "notes": "Acknowledges strengths and gaps. Best balance of speed and reason quality.",
    },
    "arcee-ai/trinity-mini:free": {
        "speed_s": 38.8, "quality_score": 0.80, "confidence": 0.90,
        "approved": True, "needs_supplement": False,
        "reason": "Comprehensive explanation of FAISS IVF with PQ parameters and usage.",
        "raw_response": '{"quality_score":0.8, "confidence": 0.9, "approved": true, "needs_supplement": false, "reason": "Comprehensive explanation of FAISS IVF with PQ parameters and usage."}',
        "reason_quality": "thin",
        "judgment_criteria": "lenient",
        "overall": "fair",
        "notes": "High quality score but 1-line reason. Speed inconsistent (3s→39s).",
    },
    "nvidia/nemotron-3-nano-30b-a3b:free": {
        "speed_s": 2.0, "quality_score": 0.78, "confidence": 0.86,
        "approved": True, "needs_supplement": False,
        "reason": "Clear explanation of FAISS IVF-PQ setup and parameters.",
        "raw_response": '{"quality_score": 0.78, "confidence": 0.86, "approved": true, "needs_supplement": false, "reason": "Clear explanation of FAISS IVF-PQ setup and parameters."}',
        "reason_quality": "thin",
        "judgment_criteria": "lenient",
        "overall": "fair",
        "notes": "Fastest model (1.7-2s) but reason too brief, confidence slightly lower.",
    },
    "arcee-ai/trinity-large-preview:free": {
        "speed_s": 9.1, "quality_score": 0.80, "confidence": 0.90,
        "approved": True, "needs_supplement": False,
        "reason": "The document provides a clear and concise explanation of FAISS IndexIVFPQ implementation for billion-scale search, covering key parameters, training, and search process. The information is accurate and well-structured.",
        "raw_response": '{"quality_score": 0.8, "confidence": 0.9, "approved": true, "needs_supplement": false, "reason": "The document provides a clear and concise explanation of FAISS IndexIVFPQ implementation for billion-scale search, covering key parameters, training, and search process. The information is accurate and well-structured."}',
        "reason_quality": "detailed",
        "judgment_criteria": "lenient",
        "overall": "good",
        "notes": "Good quality but EXPIRES 2026-04-03. Do not use as default.",
    },
    "stepfun/step-3.5-flash:free": {
        "speed_s": 15.5, "quality_score": 0.75, "confidence": 0.90,
        "approved": False, "needs_supplement": True,
        "reason": "Accurate high-level overview but lacks implementation details like code snippets, parameter tuning, and practical examples for full deployment.",
        "raw_response": '{"quality_score": 0.75, "confidence": 0.9, "approved": false, "needs_supplement": true, "reason": "Accurate high-level overview but lacks implementation details like code snippets, parameter tuning, and practical examples for full deployment."}',
        "reason_quality": "detailed",
        "judgment_criteria": "strict",
        "overall": "poor",
        "notes": "Requires code examples to approve. Will over-HOLD conceptual docs.",
    },
    "qwen/qwen3.6-plus-preview:free": {
        "speed_s": 35.2, "quality_score": 0.75, "confidence": 0.90,
        "approved": False, "needs_supplement": True,
        "reason": "Technically accurate and concise, but lacks code examples, memory considerations, and detailed parameter tuning guidance expected for an implementation guide.",
        "raw_response": '{"quality_score": 0.75, "confidence": 0.9, "approved": false, "needs_supplement": true, "reason": "Technically accurate and concise, but lacks code examples, memory considerations, and detailed parameter tuning guidance expected for an implementation guide."}',
        "reason_quality": "detailed",
        "judgment_criteria": "strict",
        "overall": "poor",
        "notes": "Slow (35s) and strict. Requires code examples. Not suitable for batch mature.",
    },
    "z-ai/glm-4.5-air:free": {
        "speed_s": 121.6, "quality_score": 0.70, "confidence": 0.90,
        "approved": True, "needs_supplement": False,
        "reason": "The document provides accurate and relevant information about FAISS IndexIVFPQ for large-scale search.",
        "raw_response": '{"quality_score": 0.70, "confidence": 0.9, "approved": true, "needs_supplement": false, "reason": "The document provides accurate and relevant information about FAISS IndexIVFPQ for large-scale search."}',
        "reason_quality": "thin",
        "judgment_criteria": "lenient",
        "overall": "poor",
        "notes": "Way too slow (121s). Not usable for batch processing.",
    },
}

ERROR_MODELS = {
    "minimax/minimax-m2.5:free": "404 No endpoints (data policy)",
    "openai/gpt-oss-120b:free": "404 No endpoints (data policy)",
    "openai/gpt-oss-20b:free": "404 No endpoints (data policy)",
    "qwen/qwen3-next-80b-a3b-instruct:free": "429 Rate limited (Venice)",
    "nvidia/nemotron-nano-9b-v2:free": "TypeError NoneType (unstable)",
    "qwen/qwen3-coder:free": "429 Rate limited (Venice)",
    "meta-llama/llama-3.3-70b-instruct:free": "429 Rate limited (Venice)",
    "nousresearch/hermes-3-llama-3.1-405b:free": "429 Rate limited (Venice)",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free": "Venice endpoint (untested)",
}


def init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id                  TEXT PRIMARY KEY,
            name                TEXT,
            created_ts          INTEGER,
            created_date        TEXT,
            expires             TEXT,
            knowledge_cutoff    TEXT,
            context_length      INTEGER,
            architecture_json   TEXT,
            per_request_limits  TEXT,
            description         TEXT,
            -- test state
            tested              INTEGER DEFAULT 0,
            test_error          TEXT,
            tested_at           TEXT,
            -- performance
            speed_s             REAL,
            quality_score       REAL,
            confidence          REAL,
            approved            INTEGER,
            needs_supplement    INTEGER,
            reason              TEXT,
            raw_response        TEXT,
            -- manual ratings
            reason_quality      TEXT,
            judgment_criteria   TEXT,
            overall             TEXT,
            notes               TEXT
        )
    """)
    con.commit()
    return con


def fetch_free_models() -> list[dict]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    models = data.get("data", [])
    return [
        m for m in models
        if str(m.get("pricing", {}).get("prompt", "1")) == "0"
        and m.get("id", "") not in EXCLUDE_IDS
    ]


def upsert_model(con: sqlite3.Connection, m: dict) -> None:
    created_ts = m.get("created", 0)
    created_date = (
        datetime.fromtimestamp(created_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if created_ts else ""
    )
    con.execute("""
        INSERT INTO models (
            id, name, created_ts, created_date, expires, knowledge_cutoff,
            context_length, architecture_json, per_request_limits, description
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            created_ts=excluded.created_ts,
            created_date=excluded.created_date,
            expires=excluded.expires,
            knowledge_cutoff=excluded.knowledge_cutoff,
            context_length=excluded.context_length,
            architecture_json=excluded.architecture_json,
            per_request_limits=excluded.per_request_limits,
            description=excluded.description
    """, (
        m["id"],
        m.get("name", ""),
        created_ts,
        created_date,
        m.get("expiration_date") or "",
        m.get("knowledge_cutoff") or "",
        m.get("context_length", 0),
        json.dumps(m.get("architecture") or {}),
        json.dumps(m.get("per_request_limits") or {}),
        (m.get("description") or "")[:500],
    ))


def apply_previous_results(con: sqlite3.Connection) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for model_id, r in PREVIOUS_RESULTS.items():
        con.execute("""
            UPDATE models SET
                tested=1, tested_at=?,
                speed_s=?, quality_score=?, confidence=?,
                approved=?, needs_supplement=?,
                reason=?, raw_response=?,
                reason_quality=?, judgment_criteria=?, overall=?, notes=?
            WHERE id=?
        """, (
            now, r["speed_s"], r["quality_score"], r["confidence"],
            int(r["approved"]), int(r["needs_supplement"]),
            r["reason"], r["raw_response"],
            r["reason_quality"], r["judgment_criteria"], r["overall"], r["notes"],
            model_id,
        ))
    for model_id, err in ERROR_MODELS.items():
        con.execute("""
            UPDATE models SET tested=1, tested_at=?, test_error=?
            WHERE id=?
        """, (now, err, model_id))
    con.commit()


async def run_test(model_id: str) -> dict:
    from src.llm.gateway import LLMGateway
    gw = LLMGateway()
    t = time.time()
    resp = await gw.complete(
        REVIEW_PROMPT, system=REVIEW_SYSTEM,
        provider="openrouter", model=model_id,
        max_tokens=400, temperature=0.0,
    )
    elapsed = time.time() - t
    content = resp.content.strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    parsed = json.loads(match.group()) if match else {}
    return {
        "speed_s": round(elapsed, 2),
        "quality_score": parsed.get("quality_score"),
        "confidence": parsed.get("confidence"),
        "approved": parsed.get("approved"),
        "needs_supplement": parsed.get("needs_supplement"),
        "reason": parsed.get("reason", ""),
        "raw_response": content,
    }


def show_results(con: sqlite3.Connection) -> None:
    print(f"\n{'ID':50s} {'created':10s} {'expires':10s} {'tested':6s} {'spd':5s} {'qs':4s} {'ap':3s} {'rq':10s} {'jc':10s} {'overall':8s}")
    print("-" * 130)
    rows = con.execute("""
        SELECT id, created_date, expires, tested, test_error,
               speed_s, quality_score, approved,
               reason_quality, judgment_criteria, overall, notes
        FROM models ORDER BY created_ts DESC
    """).fetchall()
    for r in rows:
        mid, cdate, exp, tested, err, spd, qs, ap, rq, jc, ov, notes = r
        if not tested:
            status = "  -   "
        elif err:
            status = f"ERR   "
        else:
            status = "  ✓   "
        spd_s  = f"{spd:.1f}s" if spd else "-    "
        qs_s   = f"{qs:.2f}" if qs is not None else "-   "
        ap_s   = ("✅" if ap else "❌") if ap is not None else "-"
        exp_s  = exp[:10] if exp else "-"
        print(f"{mid:50s} {cdate:10s} {exp_s:10s} {status:6s} {spd_s:5s} {qs_s:4s} {ap_s:3s} {(rq or '-'):10s} {(jc or '-'):10s} {(ov or '-'):8s}")
        if notes:
            print(f"  {'':50s} ↳ {notes}")


def show_log(con: sqlite3.Connection, model_id: str) -> None:
    row = con.execute(
        "SELECT * FROM models WHERE id=?", (model_id,)
    ).fetchone()
    if not row:
        print(f"Not found: {model_id}")
        return
    cols = [d[0] for d in con.execute("SELECT * FROM models LIMIT 0").description]
    print(f"\n=== {model_id} ===")
    for col, val in zip(cols, row):
        if val not in (None, "", "{}"):
            print(f"  {col:22s}: {val}")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--init",  action="store_true", help="DB初期化 & モデル一覧取得")
    parser.add_argument("--test",  action="store_true", help="未テストモデルをテスト")
    parser.add_argument("--show",  action="store_true", help="結果一覧表示")
    parser.add_argument("--log",   type=str,            help="特定モデルの詳細表示")
    args = parser.parse_args()

    con = init_db()

    if args.init:
        print("Fetching free models from OpenRouter...")
        models = fetch_free_models()
        print(f"  Found {len(models)} free language models")
        for m in models:
            upsert_model(con, m)
        con.commit()
        print("Applying previous test results...")
        apply_previous_results(con)
        print("Done.")
        show_results(con)

    elif args.test:
        untested = con.execute(
            "SELECT id FROM models WHERE tested=0 ORDER BY created_ts DESC"
        ).fetchall()
        print(f"Untested models: {len(untested)}")

        async def run_all():
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for (model_id,) in untested:
                print(f"\nTesting: {model_id}")
                try:
                    r = await run_test(model_id)
                    print(f"  speed={r['speed_s']}s  quality={r['quality_score']}  "
                          f"approved={r['approved']}  needs_supplement={r['needs_supplement']}")
                    print(f"  reason: {r['reason'][:100]}")
                    con.execute("""
                        UPDATE models SET
                            tested=1, tested_at=?,
                            speed_s=?, quality_score=?, confidence=?,
                            approved=?, needs_supplement=?,
                            reason=?, raw_response=?
                        WHERE id=?
                    """, (
                        now, r["speed_s"], r["quality_score"], r["confidence"],
                        int(bool(r["approved"])), int(bool(r["needs_supplement"])),
                        r["reason"], r["raw_response"],
                        model_id,
                    ))
                    con.commit()
                except Exception as e:
                    err = str(e)[:200]
                    print(f"  ERROR: {err}")
                    con.execute(
                        "UPDATE models SET tested=1, tested_at=?, test_error=? WHERE id=?",
                        (now, err, model_id),
                    )
                    con.commit()

        asyncio.run(run_all())
        show_results(con)

    elif args.show:
        show_results(con)

    elif args.log:
        show_log(con, args.log)

    else:
        parser.print_help()

    con.close()


if __name__ == "__main__":
    main()

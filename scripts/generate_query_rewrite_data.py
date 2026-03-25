"""scripts/generate_query_rewrite_data.py — CRAG 訓練データ生成

Teacher LLM を使って (ユーザー質問, 最適検索クエリ) ペアを生成する。
生成したデータは FLAN-T5 / Qwen の SFT + RL fine-tune に使用する。

使い方:
    # dry-run: 生成されるプロンプトを確認
    poetry run python scripts/generate_query_rewrite_data.py --dry-run --count 5

    # 実行: Teacher LLM でデータ生成
    poetry run python scripts/generate_query_rewrite_data.py --count 200 --output data/training/query_rewrite_pairs.jsonl

    # FAISS から質問を抽出して使用
    poetry run python scripts/generate_query_rewrite_data.py --from-faiss --count 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 定数 ──────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a search query optimization expert. Given a user question about programming or technology,
generate 3-5 optimal search queries that would retrieve the most relevant documents from a code/docs knowledge base.

Rules:
- Each query should target a different aspect of the question
- Include both specific (exact terms) and broad (conceptual) queries
- Use technical keywords, library names, and function names where appropriate
- Output as JSON: {"question": "...", "queries": ["q1", "q2", "q3", ...]}
- No explanation, only valid JSON"""

_SEED_QUESTIONS = [
    "How do I use FAISS for similarity search in Python?",
    "What is the difference between IVF and HNSW indexes in FAISS?",
    "How to implement RAG with LangChain and OpenAI?",
    "Explain knowledge distillation for small language models",
    "How to fine-tune a language model with LoRA?",
    "What is GRPO and how does it compare to PPO for RLHF?",
    "How to set up a Docker sandbox for code execution?",
    "Implement a cross-encoder reranker for search results",
    "How to use sentence-transformers for embedding generation?",
    "What is Corrective RAG and how does it improve retrieval?",
    "How to build a knowledge graph with NetworkX?",
    "Explain the TinyLoRA approach for efficient fine-tuning",
    "How to implement iterative retrieval with HyDE?",
    "What is Reciprocal Rank Fusion for combining search results?",
    "How to use vLLM for efficient LLM inference?",
    "Implement a Learning to Rank model for document reranking",
    "How to design a multi-turn conversation system?",
    "What are the best practices for prompt engineering?",
    "How to implement a query expander for search systems?",
    "Explain the difference between SFT, DPO, and PPO for alignment",
    "How to use aiosqlite for async database operations?",
    "Implement token-based authentication with JWT in FastAPI",
    "How to build a Gradio interface with multiple tabs?",
    "What is the Star attention mechanism for long context?",
    "How to implement memory consolidation for a RAG system?",
    "Explain entity extraction using LLM APIs",
    "How to set up CI/CD with GitHub Actions for Python projects?",
    "Implement a difficulty-based curriculum for model training",
    "How to use restic for incremental data backups?",
    "What is the CURIO reward for curiosity-driven exploration?",
]


async def generate_with_teacher(
    questions: list[str],
    provider: str | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Teacher LLM で質問→検索クエリペアを生成する。"""
    if dry_run:
        logger.info("DRY RUN: %d questions would be sent to Teacher", len(questions))
        for i, q in enumerate(questions[:5]):
            logger.info("  [%d] %s", i, q)
        return []

    from src.llm.gateway import LLMGateway

    gateway = LLMGateway()

    pairs: list[dict] = []
    errors = 0

    for i, question in enumerate(questions):
        try:
            response = await gateway.complete(
                f"Generate search queries for:\n{question}",
                system=_SYSTEM_PROMPT,
                provider=provider,
                temperature=0.3,
            )
            text = response.content.strip()
            # JSON パース
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            if "queries" in data and isinstance(data["queries"], list):
                pair = {
                    "question": question,
                    "queries": data["queries"],
                    "source": "teacher_generated",
                }
                pairs.append(pair)
                logger.info("[%d/%d] OK: %d queries for '%s'", i + 1, len(questions), len(data["queries"]), question[:50])
            else:
                logger.warning("[%d/%d] Invalid format: %s", i + 1, len(questions), text[:100])
                errors += 1
        except json.JSONDecodeError:
            logger.warning("[%d/%d] JSON parse error", i + 1, len(questions))
            errors += 1
        except Exception:
            logger.exception("[%d/%d] Failed", i + 1, len(questions))
            errors += 1

    logger.info("Generated %d pairs (%d errors)", len(pairs), errors)
    return pairs


async def load_questions_from_faiss(count: int) -> list[str]:
    """FAISS メタデータから既存の質問/タイトルを抽出する。"""
    try:
        from src.memory.metadata_store import MetadataStore

        store = MetadataStore()
        await store.initialize()
        docs = await store.list_documents(limit=count * 2)
        questions = []
        for doc in docs:
            title = doc.get("title", "") or doc.get("content", "")[:100]
            if title:
                questions.append(f"How to {title.lower().strip('.')}?")
        await store.close()
        return questions[:count]
    except Exception:
        logger.exception("Failed to load from FAISS, using seed questions")
        return _SEED_QUESTIONS[:count]


def save_pairs(pairs: list[dict], output_path: Path) -> None:
    """JSONL 形式で保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Saved %d pairs to %s", len(pairs), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRAG 訓練データ生成")
    parser.add_argument("--count", type=int, default=30, help="生成する質問数")
    parser.add_argument("--output", type=str, default="data/training/query_rewrite_pairs.jsonl", help="出力パス")
    parser.add_argument("--provider", type=str, default=None, help="Teacher プロバイダー")
    parser.add_argument("--from-faiss", action="store_true", help="FAISS から質問を抽出")
    parser.add_argument("--dry-run", action="store_true", help="実行せずプロンプトを表示")
    args = parser.parse_args()

    if args.from_faiss:
        questions = asyncio.run(load_questions_from_faiss(args.count))
    else:
        # シード質問をサイクルして必要数に
        questions = []
        for i in range(args.count):
            questions.append(_SEED_QUESTIONS[i % len(_SEED_QUESTIONS)])

    pairs = asyncio.run(generate_with_teacher(
        questions[:args.count],
        provider=args.provider,
        dry_run=args.dry_run,
    ))

    if pairs:
        save_pairs(pairs, Path(args.output))


if __name__ == "__main__":
    main()

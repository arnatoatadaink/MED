#!/usr/bin/env python3
"""scripts/test_teacher.py — Teacher モデルの接続テスト＆品質評価 CLI

LM Studio 等に設定した Teacher モデルの動作確認と品質ベンチマークを実行する。
テスト合格データは FAISS メモリへの投入にも対応。

使い方:
    # 接続確認（ping）
    poetry run python scripts/test_teacher.py --ping --provider lmstudio

    # 全テスト実行
    poetry run python scripts/test_teacher.py --provider lmstudio

    # JSON出力信頼性テスト
    poetry run python scripts/test_teacher.py --provider lmstudio --test json

    # コード生成テスト
    poetry run python scripts/test_teacher.py --provider lmstudio --test code

    # Entity抽出テスト
    poetry run python scripts/test_teacher.py --provider lmstudio --test entity

    # 自由プロンプト
    poetry run python scripts/test_teacher.py --provider lmstudio --prompt "FAISSの仕組みを説明して"

    # テスト結果をFAISSメモリに投入
    poetry run python scripts/test_teacher.py --provider lmstudio --test code --ingest

    # 利用可能プロバイダー一覧
    poetry run python scripts/test_teacher.py --list-providers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ============================================================================
# テストケース定義
# ============================================================================

JSON_TEST_CASES = [
    {
        "name": "simple_json",
        "system": "You are a helpful assistant. Always respond in valid JSON.",
        "prompt": 'Return a JSON object with keys "name", "language", "purpose" describing the FAISS library.',
        "validate": lambda r: _validate_json(r, required_keys=["name", "language", "purpose"]),
    },
    {
        "name": "json_array",
        "system": "You are a helpful assistant. Always respond in valid JSON.",
        "prompt": "Return a JSON array of 3 objects, each with keys \"name\" and \"category\", listing popular Python ML libraries.",
        "validate": lambda r: _validate_json_array(r, min_items=3),
    },
    {
        "name": "nested_json",
        "system": "You are a helpful assistant. Always respond in valid JSON.",
        "prompt": (
            'Return a JSON object with key "entities" (array of objects with "name" and "type") '
            'and key "relations" (array of objects with "source", "target", "type"). '
            "Describe the relationship between Python, NumPy, and machine learning."
        ),
        "validate": lambda r: _validate_json(r, required_keys=["entities", "relations"]),
    },
]

CODE_TEST_CASES = [
    {
        "name": "python_function",
        "system": "You are an expert Python programmer. Return only code, no explanation.",
        "prompt": "Write a Python function `binary_search(arr: list[int], target: int) -> int` that returns the index or -1.",
        "validate": lambda r: _validate_code(r, must_contain=["def binary_search", "return"]),
    },
    {
        "name": "python_class",
        "system": "You are an expert Python programmer. Return only code, no explanation.",
        "prompt": "Write a Python class `LRUCache` with `get(key)` and `put(key, value)` methods using OrderedDict.",
        "validate": lambda r: _validate_code(r, must_contain=["class LRUCache", "def get", "def put"]),
    },
    {
        "name": "code_with_docstring",
        "system": "You are an expert Python programmer.",
        "prompt": "Write a Python function `cosine_similarity(a, b)` that computes cosine similarity between two lists of floats. Include a docstring.",
        "validate": lambda r: _validate_code(r, must_contain=["def cosine_similarity", '"""']),
    },
]

ENTITY_TEST_CASES = [
    {
        "name": "entity_extraction",
        "system": (
            "You are a knowledge graph entity extractor. "
            "Given text, extract entities and relations. "
            "Return valid JSON with keys: entities (array of {name, type}), "
            "relations (array of {source, target, type, weight})."
        ),
        "prompt": (
            "Extract entities and relations from this text:\n\n"
            "FAISS is a library developed by Facebook AI Research for efficient "
            "similarity search. It uses vector quantization techniques like PQ "
            "(Product Quantization) and supports GPU acceleration via CUDA. "
            "It is commonly used in RAG systems alongside embedding models "
            "like sentence-transformers."
        ),
        "validate": lambda r: _validate_entity_extraction(r),
    },
    {
        "name": "causal_relations",
        "system": (
            "You are a knowledge graph entity extractor. "
            "Extract entities and causal/hierarchical relations. "
            "Return valid JSON with keys: entities (array of {name, type}), "
            "relations (array of {source, target, type, weight})."
        ),
        "prompt": (
            "Extract entities and relations from this text:\n\n"
            "TinyLoRA achieves 91% accuracy on GSM8K by using only 13 trainable "
            "parameters. It works by freezing the base model weights and learning "
            "a tiny projection matrix through GRPO reinforcement learning. "
            "The key insight is that knowledge already exists in the pretrained model; "
            "RL only needs to teach the model how to use it."
        ),
        "validate": lambda r: _validate_entity_extraction(r, min_entities=3, min_relations=2),
    },
]

QA_TEST_CASES = [
    {
        "name": "factual_qa",
        "system": "You are a technical expert. Answer concisely and accurately.",
        "prompt": "What is the difference between FAISS IndexFlatIP and IndexIVFFlat? When should you use each?",
        "validate": lambda r: _validate_length(r, min_chars=100, max_chars=2000),
    },
    {
        "name": "reasoning",
        "system": "You are a technical expert. Think step by step.",
        "prompt": (
            "A RAG system retrieves 10 documents per query. The FAISS index has 50,000 documents "
            "with 768-dimensional embeddings. Should we use IndexFlatIP or IndexIVFPQ? "
            "Calculate the approximate memory usage for each and recommend one."
        ),
        "validate": lambda r: _validate_length(r, min_chars=200, max_chars=3000),
    },
]


# ============================================================================
# バリデーション関数
# ============================================================================


def _extract_json(text: str) -> str:
    """マークダウンコードブロックからJSONを抽出する。"""
    # ```json ... ``` パターン
    if "```" in text:
        lines = text.split("\n")
        in_block = False
        json_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                json_lines.append(line)
        if json_lines:
            return "\n".join(json_lines)
    return text.strip()


def _validate_json(text: str, required_keys: list[str] | None = None) -> tuple[bool, str]:
    try:
        data = json.loads(_extract_json(text))
        if required_keys:
            if not isinstance(data, dict):
                return False, f"Expected dict, got {type(data).__name__}"
            missing = [k for k in required_keys if k not in data]
            if missing:
                return False, f"Missing keys: {missing}"
        return True, "Valid JSON"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"


def _validate_json_array(text: str, min_items: int = 1) -> tuple[bool, str]:
    try:
        data = json.loads(_extract_json(text))
        if not isinstance(data, list):
            # オブジェクト内の配列もチェック
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list) and len(v) >= min_items:
                        return True, f"Found array with {len(v)} items (in object)"
            return False, f"Expected array, got {type(data).__name__}"
        if len(data) < min_items:
            return False, f"Expected >={min_items} items, got {len(data)}"
        return True, f"Valid JSON array with {len(data)} items"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"


def _validate_code(text: str, must_contain: list[str] | None = None) -> tuple[bool, str]:
    # コードブロックから抽出
    code = text
    if "```" in text:
        code = _extract_json(text)  # 同じロジックで抽出可能

    if must_contain:
        missing = [p for p in must_contain if p not in code]
        if missing:
            return False, f"Missing patterns: {missing}"
    return True, "Code contains expected patterns"


def _validate_entity_extraction(
    text: str, min_entities: int = 2, min_relations: int = 1
) -> tuple[bool, str]:
    ok, msg = _validate_json(text, required_keys=["entities", "relations"])
    if not ok:
        return False, msg

    data = json.loads(_extract_json(text))
    entities = data.get("entities", [])
    relations = data.get("relations", [])

    issues = []
    if len(entities) < min_entities:
        issues.append(f"entities: {len(entities)} < {min_entities}")
    if len(relations) < min_relations:
        issues.append(f"relations: {len(relations)} < {min_relations}")

    # Entity 構造チェック
    for e in entities:
        if "name" not in e:
            issues.append(f"Entity missing 'name': {e}")
            break

    # Relation 構造チェック
    for r in relations:
        if "source" not in r or "target" not in r:
            issues.append(f"Relation missing source/target: {r}")
            break

    if issues:
        return False, "; ".join(issues)
    return True, f"{len(entities)} entities, {len(relations)} relations"


def _validate_length(text: str, min_chars: int = 50, max_chars: int = 5000) -> tuple[bool, str]:
    length = len(text.strip())
    if length < min_chars:
        return False, f"Too short: {length} < {min_chars} chars"
    if length > max_chars:
        return False, f"Too long: {length} > {max_chars} chars"
    return True, f"{length} chars"


# ============================================================================
# テスト実行
# ============================================================================


async def ping_provider(provider: str, model: str | None, timeout: float) -> bool:
    """Provider への接続テスト。"""
    from src.llm.gateway import LLMGateway

    gw = LLMGateway()
    available = gw.available_providers()
    print(f"Available providers: {available}")

    if provider not in available:
        print(f"[ERROR] Provider '{provider}' not available")
        return False

    print(f"\n[ping] Sending test request to {provider}...")
    start = time.monotonic()
    try:
        resp = await gw.complete(
            "Say 'hello' in one word.",
            provider=provider,
            model=model,
            max_tokens=32,
            temperature=0.0,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        print(f"[ping] OK — {elapsed:.1f}s")
        print(f"  Model : {resp.model}")
        print(f"  Tokens: in={resp.input_tokens} out={resp.output_tokens}")
        print(f"  Reply : {resp.content[:200]}")
        return True
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"[ping] FAILED after {elapsed:.1f}s — {e}")
        return False


async def run_test_suite(
    provider: str,
    model: str | None,
    test_cases: list[dict],
    suite_name: str,
    timeout: float,
    max_tokens: int,
) -> list[dict]:
    """テストスイートを実行し結果を返す。"""
    from src.llm.gateway import LLMGateway

    gw = LLMGateway()
    results = []

    print(f"\n{'='*60}")
    print(f"  {suite_name} ({len(test_cases)} tests)")
    print(f"{'='*60}")

    for tc in test_cases:
        name = tc["name"]
        print(f"\n[{name}] Running...", end=" ", flush=True)

        start = time.monotonic()
        try:
            resp = await gw.complete(
                tc["prompt"],
                system=tc.get("system"),
                provider=provider,
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,
                timeout=timeout,
            )
            elapsed = time.monotonic() - start

            passed, detail = tc["validate"](resp.content)
            status = "PASS" if passed else "FAIL"
            print(f"{status} ({elapsed:.1f}s, {resp.output_tokens} tok)")

            if not passed:
                print(f"  Reason: {detail}")
                print(f"  Output: {resp.content[:300]}...")

            results.append({
                "name": name,
                "suite": suite_name,
                "passed": passed,
                "detail": detail,
                "elapsed_s": round(elapsed, 2),
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "content": resp.content,
                "model": resp.model,
            })

        except Exception as e:
            elapsed = time.monotonic() - start
            print(f"ERROR ({elapsed:.1f}s)")
            print(f"  {e}")
            results.append({
                "name": name,
                "suite": suite_name,
                "passed": False,
                "detail": f"Exception: {e}",
                "elapsed_s": round(elapsed, 2),
                "input_tokens": 0,
                "output_tokens": 0,
                "content": "",
                "model": "",
            })

    return results


async def run_free_prompt(
    provider: str, model: str | None, prompt: str, timeout: float, max_tokens: int
) -> None:
    """自由プロンプトを実行する。"""
    from src.llm.gateway import LLMGateway

    gw = LLMGateway()

    print(f"\n[prompt] Provider: {provider}")
    print(f"[prompt] Sending: {prompt[:100]}...")
    print()

    start = time.monotonic()
    resp = await gw.complete(
        prompt,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start

    print(resp.content)
    print(f"\n--- {resp.model} | {elapsed:.1f}s | in={resp.input_tokens} out={resp.output_tokens} ---")


async def ingest_results(results: list[dict], domain: str) -> int:
    """合格したテスト結果をFAISSメモリに投入する。"""
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.memory.schema import Document, Domain, SourceMeta, SourceType

    passed = [r for r in results if r["passed"] and r["content"]]
    if not passed:
        print("[ingest] No passed results to ingest")
        return 0

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    count = 0
    for r in passed:
        try:
            doc = Document(
                content=r["content"],
                domain=Domain(domain),
                source=SourceMeta(
                    source_type=SourceType.TEACHER,
                    title=f"teacher_test:{r['name']}",
                ).set_teacher(
                    teacher_id=r.get("model", "unknown"),
                    provider=r["name"],
                ),
            )
            await mm.add(doc)
            count += 1
        except Exception as e:
            print(f"  [warn] Failed to ingest {r['name']}: {e}")

    await mm.close()
    return count


def print_summary(all_results: list[dict]) -> None:
    """テスト結果サマリーを表示する。"""
    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    failed = total - passed

    total_time = sum(r["elapsed_s"] for r in all_results)
    total_in = sum(r["input_tokens"] for r in all_results)
    total_out = sum(r["output_tokens"] for r in all_results)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total : {total} tests")
    print(f"  Passed: {passed}  Failed: {failed}")
    print(f"  Rate  : {passed/total*100:.0f}%" if total else "  Rate  : N/A")
    print(f"  Time  : {total_time:.1f}s total, {total_time/total:.1f}s avg" if total else "")
    print(f"  Tokens: in={total_in} out={total_out} total={total_in+total_out}")

    if failed:
        print(f"\n  Failed tests:")
        for r in all_results:
            if not r["passed"]:
                print(f"    - [{r['suite']}] {r['name']}: {r['detail']}")

    # Teacher適格性判定
    if total >= 4:
        rate = passed / total
        if rate >= 0.8:
            print(f"\n  >> Teacher適格: PASS ({rate*100:.0f}% >= 80%)")
        else:
            print(f"\n  >> Teacher適格: FAIL ({rate*100:.0f}% < 80%)")


# ============================================================================
# メインエントリーポイント
# ============================================================================

TEST_SUITES = {
    "json": ("JSON Output (IFEval-like)", JSON_TEST_CASES),
    "code": ("Code Generation", CODE_TEST_CASES),
    "entity": ("Entity Extraction (KG)", ENTITY_TEST_CASES),
    "qa": ("QA / Reasoning", QA_TEST_CASES),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher model connection test & quality benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", type=str, default=None, help="Provider name (e.g. lmstudio)")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument("--test", type=str, default=None,
                        choices=list(TEST_SUITES.keys()) + ["all"],
                        help="Test suite to run (default: all)")
    parser.add_argument("--ping", action="store_true", help="Connection test only")
    parser.add_argument("--prompt", type=str, default=None, help="Free-form prompt")
    parser.add_argument("--list-providers", action="store_true", help="List available providers")
    parser.add_argument("--ingest", action="store_true", help="Ingest passed results to FAISS")
    parser.add_argument("--domain", default="code", choices=["code", "academic", "general"])
    parser.add_argument("--timeout", type=float, default=300, help="Request timeout (seconds)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # --list-providers
    if args.list_providers:
        from src.llm.gateway import LLMGateway
        gw = LLMGateway()
        print("Available providers:")
        for name in gw.available_providers():
            prov = gw._providers[name]
            info = ""
            if hasattr(prov, "_base_url"):
                info = f" ({prov._base_url})"
            if hasattr(prov, "_default_model"):
                info += f" model={prov._default_model}"
            print(f"  - {name}{info}")
        return

    if not args.provider:
        parser.error("--provider is required (use --list-providers to see options)")

    # --ping
    if args.ping:
        ok = asyncio.run(ping_provider(args.provider, args.model, args.timeout))
        sys.exit(0 if ok else 1)

    # --prompt (free form)
    if args.prompt:
        asyncio.run(run_free_prompt(args.provider, args.model, args.prompt, args.timeout, args.max_tokens))
        return

    # テスト実行
    suites_to_run = list(TEST_SUITES.keys()) if args.test in (None, "all") else [args.test]

    # まず ping
    if not asyncio.run(ping_provider(args.provider, args.model, args.timeout)):
        print("\n[ABORT] Connection failed. Check provider and try --ping first.")
        sys.exit(1)

    all_results: list[dict] = []
    for suite_key in suites_to_run:
        suite_name, cases = TEST_SUITES[suite_key]
        results = asyncio.run(
            run_test_suite(args.provider, args.model, cases, suite_name, args.timeout, args.max_tokens)
        )
        all_results.extend(results)

    print_summary(all_results)

    # --output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[output] Results saved to {out_path}")

    # --ingest
    if args.ingest:
        count = asyncio.run(ingest_results(all_results, args.domain))
        print(f"[ingest] Added {count} docs to FAISS memory (domain={args.domain})")


if __name__ == "__main__":
    main()

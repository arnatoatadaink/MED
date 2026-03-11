#!/usr/bin/env python3
"""scripts/evaluate_student.py — Student モデル評価スクリプト

学習済み Student モデルをベンチマークで評価し、Teacher との比較を行う。

使い方:
    python scripts/evaluate_student.py
    python scripts/evaluate_student.py --benchmarks qa_retrieval code_generation
    python scripts/evaluate_student.py --compare-teacher
    python scripts/evaluate_student.py --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


async def run_evaluation(
    benchmark_names: list[str],
    compare_teacher: bool,
    output_path: str | None,
) -> None:
    from src.common.config import get_settings
    from src.llm.gateway import LLMGateway
    from src.training.evaluation.benchmark_suite import BenchmarkSuite
    from src.training.evaluation.teacher_comparison import TeacherComparison

    settings = get_settings()
    gateway = LLMGateway(settings)

    # ベンチマーク実行
    suite = BenchmarkSuite(gateway)
    print(f"[eval] Running benchmarks: {benchmark_names or 'all'}...")
    report = await suite.run(benchmark_names=benchmark_names if benchmark_names else None)

    print(f"\n[eval] Overall score: {report.overall_score:.4f}")
    print(f"[eval] Elapsed: {report.elapsed_seconds:.1f}s\n")

    summary = report.summary()
    for name, metrics in summary.get("benchmarks", {}).items():
        print(f"  {name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # Teacher 比較
    if compare_teacher:
        print("\n[eval] Running Teacher comparison...")
        comparison = TeacherComparison(gateway)
        test_queries = [
            ("What is FAISS?", "FAISS is a vector search library."),
            ("Explain TinyLoRA.", "TinyLoRA uses 13 parameters for efficient RL."),
        ]
        results = []
        for query, student_answer in test_queries:
            teacher_answers = await comparison.generate_teacher_answers([query])
            result = await comparison.compare(query, student_answer, teacher_answers[0])
            results.append(result)
            print(f"  Q: {query[:50]}")
            print(f"    Student: {result.student_score:.3f} | Teacher: {result.teacher_score:.3f} | Winner: {result.winner}")

        win_rate = sum(1 for r in results if r.winner == "student") / len(results)
        print(f"\n  Student win rate: {win_rate:.1%}")

    # 結果出力
    if output_path:
        out = {
            "overall_score": report.overall_score,
            "benchmarks": summary.get("benchmarks", {}),
            "elapsed_seconds": report.elapsed_seconds,
        }
        Path(output_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n[eval] Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Student model performance")
    parser.add_argument("--benchmarks", nargs="*", metavar="NAME",
                        help="Benchmark names (omit for all)")
    parser.add_argument("--compare-teacher", action="store_true",
                        help="Run Teacher comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output file path")
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        benchmark_names=args.benchmarks or [],
        compare_teacher=args.compare_teacher,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()

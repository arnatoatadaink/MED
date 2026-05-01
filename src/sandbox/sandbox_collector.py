"""src/sandbox/sandbox_collector.py — サンドボックス実行による体験知収集

DBのドキュメントからコードブロックを抽出し、DockerSandboxで実行して
結果を知識ドキュメントとしてFAISSメモリに格納する。

使い方:
    from src.memory.memory_manager import MemoryManager
    from src.memory.embedder import Embedder
    from src.sandbox.sandbox_collector import SandboxCollector

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    collector = SandboxCollector(mm)
    stats = await collector.collect(languages=["python", "bash"], limit=50)
    print(stats)
    await mm.close()
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field

from src.memory.memory_manager import MemoryManager
from src.memory.schema import Domain, SourceMeta, SourceType
from src.sandbox.code_extractor import CodeBlock, CodeExtractor, _EXECUTABLE
from src.sandbox.docker_runtime import DockerRuntime

logger = logging.getLogger(__name__)

# ---- 定数 -------------------------------------------------------

_MAX_OUTPUT_CHARS = 600   # ドキュメントに格納する出力の最大文字数
_MAX_CODE_CHARS   = 1500  # コード本文の最大文字数

# 「インストール不足」エラーのみの場合はスキップ
_SKIP_ERRORS = [
    "ModuleNotFoundError",
    "ImportError",
    "command not found",
    "No module named",
]


# ---- データクラス -----------------------------------------------

@dataclass
class ExecResult:
    """1コードブロックの実行結果。"""

    block: CodeBlock
    stdout: str
    stderr: str
    exit_code: str    # "0" = success, それ以外 = fail/-1
    duration_ms: float

    @property
    def success(self) -> bool:
        return self.exit_code == "0"

    @property
    def output(self) -> str:
        """stdout + stderr を結合したもの（上限付き）。"""
        combined = (self.stdout + self.stderr).strip()
        return combined[:_MAX_OUTPUT_CHARS]

    def worth_storing(self) -> bool:
        """保存価値があるか。"""
        # インストール不足のみの失敗はスキップ
        if not self.success:
            err = self.stderr + self.stdout
            if any(pat in err for pat in _SKIP_ERRORS) and not self.stdout.strip():
                return False
        return True

    def to_content(self) -> str:
        """FAISSドキュメントのコンテンツ文字列を生成する。"""
        status = "SUCCESS" if self.success else "FAIL"
        code = self.block.code[:_MAX_CODE_CHARS]
        out = self.output or "(no output)"

        return (
            f"[sandbox:{self.block.block_hash}] "
            f"{self.block.language} from: {self.block.title}\n\n"
            f"Code:\n{code}\n\n"
            f"Result [{status}, {self.duration_ms:.0f}ms]:\n{out}"
        )

    def to_source_extra(self) -> dict:
        return {
            "domain_flag":        "practical_reference",
            "content_type":       "sandbox_result",
            "block_hash":         self.block.block_hash,
            "original_doc_id":    self.block.doc_id,
            "original_source":    self.block.source_type,
            "language":           self.block.language,
            "exit_code":          self.exit_code,
            "execution_time_ms":  self.duration_ms,
            "success":            self.success,
        }


@dataclass
class CollectionStats:
    """収集処理の統計。"""

    extracted:  int = 0   # DB から抽出したブロック数
    new_blocks: int = 0   # 未収集ブロック数
    executed:   int = 0   # 実行したブロック数
    stored:     int = 0   # FAISS に格納したブロック数
    skipped:    int = 0   # スキップしたブロック数
    errors:     int = 0   # 実行エラー数
    by_lang:    dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lang_str = "  ".join(f"{k}:{v}" for k, v in self.by_lang.items())
        return (
            f"extracted={self.extracted}  new={self.new_blocks}  "
            f"executed={self.executed}  stored={self.stored}  "
            f"skipped={self.skipped}  errors={self.errors}  [{lang_str}]"
        )


# ---- SandboxCollector -------------------------------------------

class SandboxCollector:
    """DB のドキュメントからコードを実行して体験知を収集する。

    Args:
        memory_manager: 初期化済みの MemoryManager。
        db_path:        metadata.db のパス。
        network_disabled: Docker のネットワークを無効化するか。
                          pip install が必要な場合は False を指定。
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        db_path: str = "data/metadata.db",
        network_disabled: bool = False,
    ) -> None:
        self._mm = memory_manager
        self._extractor = CodeExtractor(db_path=db_path)
        self._db_path = db_path
        self._network_disabled = network_disabled

    # ---- public ---------------------------------------------------

    async def collect(
        self,
        languages: list[str] | None = None,
        source_types: list[str] | None = None,
        limit: int = 50,
        cmd_timeout: int = 30,
    ) -> CollectionStats:
        """コードブロックを抽出・実行し、結果をメモリに格納する。

        Args:
            languages:    対象言語リスト（None = ["python", "bash"]）。
            source_types: 対象ソース種別（None = 全種別）。
            limit:        格納する最大件数。
            cmd_timeout:  1コマンドのタイムアウト秒数。

        Returns:
            CollectionStats。
        """
        if languages is None:
            languages = ["python", "bash"]

        stats = CollectionStats()

        # 1. コードブロック抽出
        blocks = self._extractor.extract_from_db(
            languages=languages,
            source_types=source_types,
            limit=limit * 5,   # 重複除去後に limit 件残るよう多めに取る
            deduplicate=True,
        )
        stats.extracted = len(blocks)
        logger.info("Extracted %d code blocks", stats.extracted)

        # 2. 未収集ブロックのみに絞り込む
        new_blocks = await asyncio.to_thread(self._filter_existing, blocks)
        new_blocks = new_blocks[:limit]
        stats.new_blocks = len(new_blocks)
        logger.info("New blocks (not yet in sandbox): %d", stats.new_blocks)

        if not new_blocks:
            logger.info("Nothing to collect.")
            return stats

        # 3. docker_image 別にグループ化して実行
        by_image: dict[str, list[CodeBlock]] = {}
        for b in new_blocks:
            img = b.docker_image or _EXECUTABLE.get(b.language, "python:3.11-slim")
            by_image.setdefault(img, []).append(b)

        results: list[ExecResult] = []
        for image, image_blocks in by_image.items():
            batch = await self._execute_batch(image, image_blocks, cmd_timeout)
            results.extend(batch)
            stats.executed += len(batch)
            stats.errors   += sum(1 for r in batch if r.exit_code == "-1")

        # 4. FAISS に格納
        for result in results:
            if not result.worth_storing():
                stats.skipped += 1
                continue
            try:
                await self._store(result)
                stats.stored += 1
                stats.by_lang[result.block.language] = (
                    stats.by_lang.get(result.block.language, 0) + 1
                )
            except Exception:
                logger.exception(
                    "Failed to store sandbox result for block %s",
                    result.block.block_hash,
                )
                stats.errors += 1

        logger.info("Collection done: %s", stats)
        return stats

    # ---- private --------------------------------------------------

    def _filter_existing(self, blocks: list[CodeBlock]) -> list[CodeBlock]:
        """すでに DB に収録済みの block_hash を除外する（同期）。"""
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT json_extract(source_extra, '$.block_hash') "
                "FROM documents WHERE source_type = 'sandbox'"
            ).fetchall()
            conn.close()
            existing = {r[0] for r in rows if r[0]}
        except Exception:
            logger.warning("Could not query existing sandbox hashes; skipping filter")
            existing = set()

        return [b for b in blocks if b.block_hash not in existing]

    async def _execute_batch(
        self,
        docker_image: str,
        blocks: list[CodeBlock],
        cmd_timeout: int,
    ) -> list[ExecResult]:
        """1つのコンテナで blocks を順番に実行する。"""
        results: list[ExecResult] = []

        logger.info(
            "Starting container: image=%s blocks=%d", docker_image, len(blocks)
        )
        try:
            runtime = await asyncio.to_thread(
                DockerRuntime.create,
                docker_image,
                network_disabled=self._network_disabled,
            )
        except Exception:
            logger.exception("Failed to start container: %s", docker_image)
            return results

        try:
            for i, block in enumerate(blocks):
                logger.debug(
                    "[%d/%d] Running %s block: %s",
                    i + 1, len(blocks), block.language, block.code[:60],
                )
                t0 = time.monotonic()
                try:
                    stdout, stderr, ec = await runtime.async_demux_run(
                        _build_run_command(block),
                        timeout=cmd_timeout,
                    )
                except Exception as exc:
                    stdout, stderr, ec = "", str(exc), "-1"

                duration_ms = (time.monotonic() - t0) * 1000
                results.append(ExecResult(
                    block=block,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=ec,
                    duration_ms=duration_ms,
                ))
        finally:
            await runtime.aclose()

        return results

    async def _store(self, result: ExecResult) -> None:
        """ExecResult を Document に変換して MemoryManager に追加する。"""
        domain = (
            Domain.CODE if result.block.language == "python" else Domain.GENERAL
        )
        source = SourceMeta(
            source_type=SourceType.SANDBOX,
            url=result.block.source_url,
            title=f"[sandbox] {result.block.title[:80]}",
            language=result.block.language,
            extra=result.to_source_extra(),
        )

        from src.memory.schema import Document

        doc = Document(
            content=result.to_content(),
            domain=domain,
            source=source,
            is_executable=True,
            execution_verified=True,
            last_execution_success=result.success,
        )
        await self._mm.add(doc)


# ---- コマンド文字列ビルダー -----------------------------------

def _build_run_command(block: CodeBlock) -> str:
    """CodeBlock に対してコンテナ内で実行するシェルコマンドを返す。"""
    if block.language == "python":
        # python -c を使うと indent がずれる場合があるため
        # echo でスクリプトを書き出してから実行する
        escaped = block.code.replace("'", "'\\''")
        return f"echo '{escaped}' > /tmp/snippet.py && python3 /tmp/snippet.py"

    if block.language == "bash":
        return block.code

    if block.language in ("javascript", "typescript"):
        escaped = block.code.replace("'", "'\\''")
        ext = "ts" if block.language == "typescript" else "js"
        runner = "npx tsx" if block.language == "typescript" else "node"
        return f"echo '{escaped}' > /tmp/snippet.{ext} && {runner} /tmp/snippet.{ext}"

    # fallback: bash として実行
    return block.code

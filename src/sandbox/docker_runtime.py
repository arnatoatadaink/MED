"""src/sandbox/docker_runtime.py — 永続 Docker コンテナによるコード実行

ベースライン: https://github.com/llm-in-sandbox/llm-in-sandbox (Apache 2.0)
MEDへの変更点:
  - Tsinghua pip ミラー設定を削除（中国向け設定のため）
  - git init を削除（不要）
  - async_run / async_demux_run を追加（asyncio.to_thread ラップ）
  - MED の型ヒント規約に合わせて整形
  - コンテキストマネージャー (sync/async) 対応

ephemeral方式 (executor.py) との違い:
  - コンテナを1度起動して exec_run を繰り返す → バルク実行で高速
  - 状態が実行間で共有される（pip install 結果が次の実行に引き継がれる）
  - 使用後は必ず close() / async with を呼ぶこと

使い方:
    # 同期
    runtime = DockerRuntime("python:3.11-slim")
    out, code = runtime.run("print('hello')")
    runtime.close()

    # async context manager
    async with DockerRuntime.create("python:3.11-slim") as rt:
        out, code = await rt.async_run("import sys; print(sys.version)")
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import io
import logging
import os
import re
import shlex
import tarfile
import time
from typing import Optional

import docker

logger = logging.getLogger(__name__)

# --- 定数 -------------------------------------------------------

_CMD_TIMEOUT = 120  # デフォルトタイムアウト秒

_DOCKER_PATH = (
    "/root/.local/bin:/root/.cargo/bin"
    ":/usr/local/sbin:/usr/local/bin"
    ":/usr/sbin:/usr/bin:/sbin:/bin"
)

_DEFAULT_ENV = {
    "PATH": _DOCKER_PATH,
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_ROOT_USER_ACTION": "ignore",
    "PIP_NO_WARN_SCRIPT_LOCATION": "1",
}


# --- DockerRuntime -----------------------------------------------

class DockerRuntime:
    """永続 Docker コンテナ上でコマンドを実行するランタイム。

    Args:
        docker_image: 使用する Docker イメージ名。
        workdir: コンテナ内の作業ディレクトリ。
        command: コンテナ起動コマンド（デフォルト: sleep infinity）。
        network_disabled: ネットワークを無効化するか。
        mem_limit: メモリ上限（例: "256m"）。None で無制限。
    """

    def __init__(
        self,
        docker_image: str,
        workdir: str = "/testbed",
        command: str = "sleep infinity",
        network_disabled: bool = True,
        mem_limit: Optional[str] = "512m",
    ) -> None:
        self.docker_image = docker_image
        self.workdir = workdir
        self.command = command
        self._network_disabled = network_disabled
        self._mem_limit = mem_limit

        self._client = docker.from_env(timeout=120)
        self._container: docker.models.containers.Container | None = None
        self._container_name = self._make_container_name(docker_image)

        self._start_container()
        self._setup_env()
        logger.info(
            "DockerRuntime ready: image=%s container=%s",
            docker_image, self._container_name,
        )

    # ---- ファクトリ -----------------------------------------------

    @classmethod
    def create(
        cls,
        docker_image: str = "python:3.11-slim",
        workdir: str = "/testbed",
        network_disabled: bool = True,
        mem_limit: Optional[str] = "512m",
    ) -> "DockerRuntime":
        """インスタンスを生成する（async with 用）。"""
        return cls(docker_image, workdir=workdir,
                   network_disabled=network_disabled, mem_limit=mem_limit)

    # ---- 初期化ヘルパー -------------------------------------------

    @staticmethod
    def _make_container_name(image_name: str) -> str:
        unique = str(datetime.datetime.now()) + str(os.getpid())
        h = hashlib.sha256(unique.encode()).hexdigest()[:10]
        safe = image_name.replace("/", "-").replace(":", "-")
        return f"med-sandbox-{safe}-{h}"

    def _start_container(self) -> None:
        """イメージを pull してコンテナを起動する。"""
        try:
            self._client.images.get(self.docker_image)
        except docker.errors.ImageNotFound:
            logger.info("Pulling Docker image: %s", self.docker_image)
            self._client.images.pull(self.docker_image)

        run_kwargs: dict = dict(
            command=self.command,
            name=self._container_name,
            detach=True,
            stdin_open=True,
            tty=True,
            environment=_DEFAULT_ENV,
            working_dir=self.workdir,
            network_disabled=self._network_disabled,
        )
        if self._mem_limit:
            run_kwargs["mem_limit"] = self._mem_limit

        self._container = self._client.containers.run(self.docker_image, **run_kwargs)
        logger.debug("Container started: %s", self._container_name)

    def _setup_env(self) -> None:
        """作業ディレクトリを作成する（pip mirror・git は設定しない）。"""
        self.run(f"mkdir -p {self.workdir}")

    # ---- コマンド実行 ---------------------------------------------

    def run(
        self,
        code: str,
        timeout: int = _CMD_TIMEOUT,
        workdir: Optional[str] = None,
    ) -> tuple[str, str]:
        """コンテナ内でコマンドを実行する（stdout+stderr 混在）。

        Returns:
            (output, exit_code_str)  exit_code_str は "0" または "Error: ..."
        """
        if self._container is None:
            return "Error: container not running", "-1"

        exec_workdir = workdir or self.workdir
        cmd = ["bash", "-c", f"timeout {timeout} bash -c {shlex.quote(code)}"]

        try:
            result = self._container.exec_run(
                cmd, workdir=exec_workdir, environment=_DEFAULT_ENV
            )
            output = result.output.decode("utf-8", errors="replace")
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            ec = result.exit_code

            if ec == 124:
                return f"Timed out (>{timeout}s)", "-1"
            return output, str(ec)

        except Exception as exc:
            return f"Error: {repr(exc)}", "-1"

    def demux_run(
        self,
        code: str,
        timeout: int = _CMD_TIMEOUT,
        workdir: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """stdout / stderr を分離して実行する。

        Returns:
            (stdout, stderr, exit_code_str)
        """
        if self._container is None:
            return "", "Error: container not running", "-1"

        exec_workdir = workdir or self.workdir
        cmd = ["bash", "-c", f"timeout {timeout} bash -c {shlex.quote(code)}"]

        try:
            result = self._container.exec_run(
                cmd, workdir=exec_workdir, demux=True, environment=_DEFAULT_ENV
            )
            stdout_data, stderr_data = result.output
            ec = result.exit_code

            stdout = stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
            stderr = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
            stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
            stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)

            if ec == 124:
                return f"Timed out (>{timeout}s)", "", "-1"
            return stdout, stderr, str(ec)

        except Exception as exc:
            msg = f"Error: {repr(exc)}"
            return "", msg, "-1"

    # ---- async ラッパー -------------------------------------------

    async def async_run(
        self,
        code: str,
        timeout: int = _CMD_TIMEOUT,
        workdir: Optional[str] = None,
    ) -> tuple[str, str]:
        """run() の非同期バージョン。"""
        return await asyncio.to_thread(self.run, code, timeout, workdir)

    async def async_demux_run(
        self,
        code: str,
        timeout: int = _CMD_TIMEOUT,
        workdir: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """demux_run() の非同期バージョン。"""
        return await asyncio.to_thread(self.demux_run, code, timeout, workdir)

    # ---- pip インストール -----------------------------------------

    def pip_install(self, *packages: str, timeout: int = 120) -> bool:
        """パッケージを pip install する。成功なら True。"""
        pkgs = " ".join(shlex.quote(p) for p in packages)
        _, ec = self.run(f"pip install -q {pkgs}", timeout=timeout)
        return ec == "0"

    async def async_pip_install(self, *packages: str, timeout: int = 120) -> bool:
        """pip_install の非同期バージョン。"""
        return await asyncio.to_thread(self.pip_install, *packages, timeout=timeout)

    # ---- ファイル転送 ---------------------------------------------

    def copy_to_container(self, src_path: str, dest_path: str) -> None:
        """ローカルファイルをコンテナにコピーする。"""
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            self.run(f"mkdir -p {dest_dir}")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        buf.seek(0)
        self._container.put_archive(dest_dir or "/", buf)

    def copy_from_container(self, container_path: str, local_path: str) -> None:
        """コンテナからローカルにファイルをコピーする。"""
        bits, _ = self._container.get_archive(container_path)
        os.makedirs(local_path, exist_ok=True)

        buf = io.BytesIO()
        for chunk in bits:
            buf.write(chunk)
        buf.seek(0)

        base = os.path.basename(container_path.rstrip("/"))
        with tarfile.open(fileobj=buf, mode="r") as tar:
            for member in tar.getmembers():
                if member.name == base:
                    continue
                if member.name.startswith(base + "/"):
                    member.name = member.name[len(base) + 1:]
                if not member.name:
                    continue
                os.makedirs(
                    os.path.dirname(os.path.join(local_path, member.name)),
                    exist_ok=True,
                )
                tar.extract(member, path=local_path)

    # ---- ライフサイクル -------------------------------------------

    def close(self) -> None:
        """コンテナを停止・削除する。"""
        if self._container is not None:
            try:
                self._container.stop(timeout=5)
                self._container.remove(force=True)
                logger.info("Container removed: %s", self._container_name)
            except Exception as exc:
                logger.warning("Error stopping container: %s", exc)
            finally:
                self._container = None

    async def aclose(self) -> None:
        """close() の非同期バージョン。"""
        await asyncio.to_thread(self.close)

    # ---- コンテキストマネージャー ---------------------------------

    def __enter__(self) -> "DockerRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> "DockerRuntime":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def __del__(self) -> None:
        if self._container is not None:
            try:
                self._container.stop(timeout=2)
                self._container.remove(force=True)
            except Exception:
                pass

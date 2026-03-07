"""src/gui/tabs/settings.py — 設定パネルタブ。

configs/*.yaml の現在値を表示・編集し、APIキーの設定と
設定のリロードを提供する。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_ENV_FILE = _PROJECT_ROOT / ".env"


# ────────────────────────────────────────────────────────────────
# ヘルパー
# ────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-4:]


def _read_config_yaml(name: str) -> str:
    """configs/<name>.yaml の内容を文字列で返す。"""
    path = _CONFIGS_DIR / f"{name}.yaml"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"# {name}.yaml が見つかりません"


def _write_config_yaml(name: str, content: str) -> str:
    path = _CONFIGS_DIR / f"{name}.yaml"
    try:
        parsed = yaml.safe_load(content)
        if parsed is None:
            return "❌ YAMLが空です"
        _save_yaml(path, parsed)
        return f"✅ {name}.yaml を保存しました"
    except yaml.YAMLError as e:
        return f"❌ YAML解析エラー: {e}"


# ────────────────────────────────────────────────────────────────
# タブ UI 構築
# ────────────────────────────────────────────────────────────────

def build_tab() -> None:
    """Gradio Blocks コンテキスト内で設定タブを描画する。"""

    with gr.Tabs():

        # ── APIキー ─────────────────────────────────────────────
        with gr.TabItem("APIキー"):
            gr.Markdown(
                "### APIキー設定\n"
                "キーは `.env` ファイルまたは環境変数で管理します。\n"
                "ここでの入力は**セッション中のみ有効**です（永続化しません）。"
            )

            def _key_row(label: str, env_key: str):
                with gr.Row():
                    current = _get_env(env_key)
                    display = _mask(current) if current else "_(未設定)_"
                    gr.Markdown(f"**{label}**  現在値: `{display}`", scale=2)
                    new_key = gr.Textbox(
                        placeholder=f"{env_key} を入力…",
                        type="password",
                        show_label=False,
                        scale=3,
                    )
                    apply_btn = gr.Button("適用", scale=1)
                    result_md = gr.Markdown(scale=2)

                    def _apply(value, _env=env_key):
                        if value.strip():
                            os.environ[_env] = value.strip()
                            return f"✅ `{_env}` をセッションに設定しました"
                        return "❌ 空のキーは設定できません"

                    apply_btn.click(fn=_apply, inputs=[new_key], outputs=[result_md])

            _key_row("Anthropic API Key", "ANTHROPIC_API_KEY")
            _key_row("OpenAI API Key", "OPENAI_API_KEY")
            _key_row("GitHub Token", "GITHUB_TOKEN")
            _key_row("Tavily API Key", "TAVILY_API_KEY")
            _key_row("Stack Overflow Key", "STACKOVERFLOW_API_KEY")

            gr.Markdown(
                "---\n"
                "**永続化するには** `.env` ファイルに記載してください:\n"
                "```\nANTHROPIC_API_KEY=sk-ant-...\n```"
            )

        # ── YAMLエディター ──────────────────────────────────────
        with gr.TabItem("YAML設定"):
            gr.Markdown("### 設定ファイル編集")

            config_names = [
                "default",
                "llm_config",
                "faiss_config",
                "training",
                "sandbox_policy",
                "retrievers",
            ]

            file_selector = gr.Dropdown(
                choices=config_names,
                value="default",
                label="設定ファイル",
            )
            yaml_editor = gr.Code(
                value=_read_config_yaml("default"),
                language="yaml",
                label="YAML内容",
                lines=25,
            )
            with gr.Row():
                save_btn = gr.Button("保存", variant="primary")
                reload_btn = gr.Button("再読み込み", variant="secondary")
            save_result = gr.Markdown()

            def _on_file_change(name):
                return _read_config_yaml(name)

            file_selector.change(
                fn=_on_file_change,
                inputs=[file_selector],
                outputs=[yaml_editor],
            )
            reload_btn.click(
                fn=_on_file_change,
                inputs=[file_selector],
                outputs=[yaml_editor],
            )

            def _on_save(name, content):
                return _write_config_yaml(name, content)

            save_btn.click(
                fn=_on_save,
                inputs=[file_selector, yaml_editor],
                outputs=[save_result],
            )

        # ── システム情報 ─────────────────────────────────────────
        with gr.TabItem("システム情報"):
            gr.Markdown("### 現在の設定サマリー")

            def _get_system_info() -> str:
                default_cfg = _load_yaml(_CONFIGS_DIR / "default.yaml")
                llm_cfg = _load_yaml(_CONFIGS_DIR / "llm_config.yaml")

                anthropic_key = _get_env("ANTHROPIC_API_KEY")
                openai_key = _get_env("OPENAI_API_KEY")

                lines = [
                    "| 項目 | 値 |",
                    "|------|-----|",
                    f"| アプリ名 | `{default_cfg.get('app', {}).get('name', '—')}` |",
                    f"| バージョン | `{default_cfg.get('app', {}).get('version', '—')}` |",
                    f"| ホスト | `{default_cfg.get('app', {}).get('host', '—')}:{default_cfg.get('app', {}).get('port', '—')}` |",
                    f"| 埋め込みモデル | `{default_cfg.get('embedding', {}).get('model', '—')}` |",
                    f"| 埋め込み次元 | `{default_cfg.get('embedding', {}).get('dim', '—')}` |",
                    f"| Anthropicデフォルトモデル | `{llm_cfg.get('providers', {}).get('anthropic', {}).get('default_model', '—')}` |",
                    f"| OpenAIデフォルトモデル | `{llm_cfg.get('providers', {}).get('openai', {}).get('default_model', '—')}` |",
                    f"| ANTHROPIC_API_KEY | `{'設定済 ' + _mask(anthropic_key) if anthropic_key else '未設定'}` |",
                    f"| OPENAI_API_KEY | `{'設定済 ' + _mask(openai_key) if openai_key else '未設定'}` |",
                ]
                return "\n".join(lines)

            info_md = gr.Markdown(_get_system_info())
            gr.Button("更新", variant="secondary").click(fn=_get_system_info, outputs=[info_md])

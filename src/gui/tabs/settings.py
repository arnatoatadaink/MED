"""src/gui/tabs/settings.py — 設定パネルタブ。

configs/*.yaml の現在値を表示・編集し、APIキーの設定と
設定のリロードを提供する。プロバイダープリセット・カスタムプロバイダー
管理機能を含む。
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import yaml

from src.gui.utils import get_all_provider_choices

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_ENV_FILE = _PROJECT_ROOT / ".env"

# ────────────────────────────────────────────────────────────────
# プロバイダープリセット定義
# ────────────────────────────────────────────────────────────────

_PROVIDER_PRESETS: dict[str, dict] = {
    "Anthropic Claude": {
        "description": "Claude Sonnet / Haiku を使用（デフォルト推奨）",
        "required_env": "ANTHROPIC_API_KEY",
        "summary": [
            ("主要プロバイダー", "Anthropic"),
            ("高精度タスク", "claude-sonnet-4-20250514"),
            ("軽量タスク",   "claude-haiku-4-5-20251001"),
            ("必要なAPIキー", "ANTHROPIC_API_KEY"),
            ("月額目安",     "$10 制限 / fallback: Ollama"),
        ],
        "config": {
            "primary_provider": "anthropic",
            "providers": {
                "anthropic": {
                    "default_model": "claude-sonnet-4-20250514",
                    "haiku_model":   "claude-haiku-4-5-20251001",
                },
                "openai": {
                    "default_model": "gpt-4o",
                    "mini_model":    "gpt-4o-mini",
                },
                "ollama": {
                    "base_url":      "http://localhost:11434",
                    "default_model": "llama3.1:8b",
                },
            },
            "task_routing": {
                "query_parsing":       "haiku",
                "query_rewrite":       "haiku",
                "hyde":                "haiku",
                "verification":        "sonnet",
                "response_generation": "sonnet",
                "code_generation":     "sonnet",
                "error_analysis":      "sonnet",
                "feedback_analysis":   "haiku",
            },
            "budget": {
                "daily_limit_usd":   10.0,
                "alert_threshold":   0.8,
                "fallback_to_local": True,
            },
        },
    },
    "OpenAI GPT-4o": {
        "description": "GPT-4o / GPT-4o-mini を使用",
        "required_env": "OPENAI_API_KEY",
        "summary": [
            ("主要プロバイダー", "OpenAI"),
            ("高精度タスク", "gpt-4o"),
            ("軽量タスク",   "gpt-4o-mini"),
            ("必要なAPIキー", "OPENAI_API_KEY"),
            ("月額目安",     "$10 制限 / fallback: Ollama"),
        ],
        "config": {
            "primary_provider": "openai",
            "providers": {
                "openai": {
                    "default_model": "gpt-4o",
                    "mini_model":    "gpt-4o-mini",
                },
                "anthropic": {
                    "default_model": "claude-sonnet-4-20250514",
                    "haiku_model":   "claude-haiku-4-5-20251001",
                },
                "ollama": {
                    "base_url":      "http://localhost:11434",
                    "default_model": "llama3.1:8b",
                },
            },
            "task_routing": {
                "query_parsing":       "mini",
                "query_rewrite":       "mini",
                "hyde":                "mini",
                "verification":        "gpt4o",
                "response_generation": "gpt4o",
                "code_generation":     "gpt4o",
                "error_analysis":      "gpt4o",
                "feedback_analysis":   "mini",
            },
            "budget": {
                "daily_limit_usd":   10.0,
                "alert_threshold":   0.8,
                "fallback_to_local": True,
            },
        },
    },
    "Ollama (ローカル)": {
        "description": "ローカル Ollama サーバーのみ使用（APIキー不要）",
        "required_env": None,
        "summary": [
            ("主要プロバイダー", "Ollama (ローカル)"),
            ("高精度タスク", "llama3.1:70b または llama3.3:70b"),
            ("軽量タスク",   "llama3.1:8b"),
            ("必要なAPIキー", "なし"),
            ("ベースURL",    "http://localhost:11434"),
        ],
        "config": {
            "primary_provider": "ollama",
            "providers": {
                "ollama": {
                    "base_url":      "http://localhost:11434",
                    "default_model": "llama3.1:8b",
                    "large_model":   "llama3.1:70b",
                },
                "anthropic": {
                    "default_model": "claude-sonnet-4-20250514",
                    "haiku_model":   "claude-haiku-4-5-20251001",
                },
                "openai": {
                    "default_model": "gpt-4o",
                    "mini_model":    "gpt-4o-mini",
                },
            },
            "task_routing": {
                "query_parsing":       "ollama",
                "query_rewrite":       "ollama",
                "hyde":                "ollama",
                "verification":        "ollama",
                "response_generation": "ollama",
                "code_generation":     "ollama",
                "error_analysis":      "ollama",
                "feedback_analysis":   "ollama",
            },
            "budget": {
                "daily_limit_usd":   0.0,
                "alert_threshold":   1.0,
                "fallback_to_local": True,
            },
        },
    },
    "vLLM (ローカル)": {
        "description": "ローカル vLLM サーバー (OpenAI互換エンドポイント)",
        "required_env": None,
        "summary": [
            ("主要プロバイダー", "vLLM (OpenAI互換)"),
            ("高精度タスク", "Qwen2.5-7B-Instruct または任意"),
            ("軽量タスク",   "同上"),
            ("必要なAPIキー", "なし (VLLM_API_KEY 任意)"),
            ("ベースURL",    "http://localhost:8001/v1"),
        ],
        "config": {
            "primary_provider": "vllm",
            "providers": {
                "vllm": {
                    "base_url":      "http://localhost:8001/v1",
                    "api_key_env":   "VLLM_API_KEY",
                    "default_model": "Qwen/Qwen2.5-7B-Instruct",
                    "type":          "openai_compatible",
                },
                "ollama": {
                    "base_url":      "http://localhost:11434",
                    "default_model": "llama3.1:8b",
                },
                "anthropic": {
                    "default_model": "claude-sonnet-4-20250514",
                    "haiku_model":   "claude-haiku-4-5-20251001",
                },
                "openai": {
                    "default_model": "gpt-4o",
                    "mini_model":    "gpt-4o-mini",
                },
            },
            "task_routing": {
                "query_parsing":       "vllm",
                "query_rewrite":       "vllm",
                "hyde":                "vllm",
                "verification":        "vllm",
                "response_generation": "vllm",
                "code_generation":     "vllm",
                "error_analysis":      "vllm",
                "feedback_analysis":   "vllm",
            },
            "budget": {
                "daily_limit_usd":   0.0,
                "alert_threshold":   1.0,
                "fallback_to_local": True,
            },
        },
    },
    "Azure OpenAI": {
        "description": "Azure OpenAI Service エンドポイントを使用",
        "required_env": "AZURE_OPENAI_API_KEY",
        "summary": [
            ("主要プロバイダー", "Azure OpenAI"),
            ("高精度タスク", "gpt-4o (deployment名を要設定)"),
            ("軽量タスク",   "gpt-4o-mini (deployment名を要設定)"),
            ("必要なAPIキー", "AZURE_OPENAI_API_KEY"),
            ("追加設定",     "AZURE_OPENAI_ENDPOINT / API version"),
        ],
        "config": {
            "primary_provider": "azure_openai",
            "providers": {
                "azure_openai": {
                    "endpoint_env":       "AZURE_OPENAI_ENDPOINT",
                    "api_key_env":        "AZURE_OPENAI_API_KEY",
                    "api_version":        "2024-08-01-preview",
                    "default_deployment": "gpt-4o",
                    "mini_deployment":    "gpt-4o-mini",
                    "type":               "azure",
                },
                "ollama": {
                    "base_url":      "http://localhost:11434",
                    "default_model": "llama3.1:8b",
                },
            },
            "task_routing": {
                "query_parsing":       "azure_mini",
                "query_rewrite":       "azure_mini",
                "hyde":                "azure_mini",
                "verification":        "azure",
                "response_generation": "azure",
                "code_generation":     "azure",
                "error_analysis":      "azure",
                "feedback_analysis":   "azure_mini",
            },
            "budget": {
                "daily_limit_usd":   20.0,
                "alert_threshold":   0.8,
                "fallback_to_local": True,
            },
        },
    },
    "Together.ai": {
        "description": "Together.ai の OpenAI互換 API を使用",
        "required_env": "TOGETHER_API_KEY",
        "summary": [
            ("主要プロバイダー", "Together.ai"),
            ("高精度タスク", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
            ("軽量タスク",   "meta-llama/Llama-3.1-8B-Instruct-Turbo"),
            ("必要なAPIキー", "TOGETHER_API_KEY"),
            ("ベースURL",    "https://api.together.xyz/v1"),
        ],
        "config": {
            "primary_provider": "together",
            "providers": {
                "together": {
                    "base_url":      "https://api.together.xyz/v1",
                    "api_key_env":   "TOGETHER_API_KEY",
                    "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "mini_model":    "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                    "type":          "openai_compatible",
                },
                "ollama": {
                    "base_url":      "http://localhost:11434",
                    "default_model": "llama3.1:8b",
                },
            },
            "task_routing": {
                "query_parsing":       "together_mini",
                "query_rewrite":       "together_mini",
                "hyde":                "together_mini",
                "verification":        "together",
                "response_generation": "together",
                "code_generation":     "together",
                "error_analysis":      "together",
                "feedback_analysis":   "together_mini",
            },
            "budget": {
                "daily_limit_usd":   5.0,
                "alert_threshold":   0.8,
                "fallback_to_local": True,
            },
        },
    },
}


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


def _preset_summary_md(name: str) -> str:
    """プリセットの概要を Markdown テーブルで返す。"""
    if name not in _PROVIDER_PRESETS:
        return ""
    p = _PROVIDER_PRESETS[name]
    lines = [
        f"**{p['description']}**\n",
        "| 項目 | 値 |",
        "|------|-----|",
    ]
    for k, v in p["summary"]:
        lines.append(f"| {k} | `{v}` |")

    env_key = p.get("required_env")
    if env_key:
        current = _get_env(env_key)
        status = f"設定済 {_mask(current)}" if current else "⚠️ 未設定"
        lines.append(f"| APIキー状態 | {status} |")
    return "\n".join(lines)


def _apply_preset(name: str) -> str:
    """プリセットを llm_config.yaml に書き込む。"""
    if name not in _PROVIDER_PRESETS:
        return f"❌ プリセット '{name}' が見つかりません"
    llm_path = _CONFIGS_DIR / "llm_config.yaml"
    _save_yaml(llm_path, _PROVIDER_PRESETS[name]["config"])
    return f"✅ プリセット「{name}」を llm_config.yaml に適用しました"


_KNOWN_PROVIDERS = {"anthropic", "openai", "ollama", "vllm", "azure_openai", "together"}


def _get_custom_providers() -> dict[str, dict]:
    """llm_config.yaml から組み込み以外のプロバイダーを返す。"""
    cfg = _load_yaml(_CONFIGS_DIR / "llm_config.yaml")
    return {k: v for k, v in cfg.get("providers", {}).items()
            if k not in _KNOWN_PROVIDERS}


def _add_custom_provider(
    name: str,
    ptype: str,
    base_url: str,
    default_model: str,
    api_key_env: str,
    api_key_value: str,
) -> str:
    name = name.strip()
    if not name:
        return "❌ プロバイダー名を入力してください"
    if not base_url.strip():
        return "❌ ベースURLを入力してください"
    if not default_model.strip():
        return "❌ デフォルトモデルを入力してください"

    if api_key_value.strip():
        env_name = api_key_env.strip() or f"{name.upper()}_API_KEY"
        os.environ[env_name] = api_key_value.strip()
        api_key_env = env_name

    entry: dict = {
        "type":          ptype,
        "base_url":      base_url.strip(),
        "default_model": default_model.strip(),
    }
    if api_key_env.strip():
        entry["api_key_env"] = api_key_env.strip()

    llm_path = _CONFIGS_DIR / "llm_config.yaml"
    cfg = _load_yaml(llm_path)
    cfg.setdefault("providers", {})[name] = entry
    _save_yaml(llm_path, cfg)
    return f"✅ カスタムプロバイダー「{name}」を追加/更新しました"


def _delete_custom_provider(name: str) -> tuple[str, list[str]]:
    if not name:
        return "❌ プロバイダー名を選択してください", _custom_provider_names()
    llm_path = _CONFIGS_DIR / "llm_config.yaml"
    cfg = _load_yaml(llm_path)
    providers = cfg.get("providers", {})
    if name not in providers:
        return f"❌ 「{name}」が見つかりません", _custom_provider_names()
    del providers[name]
    cfg["providers"] = providers
    _save_yaml(llm_path, cfg)
    return f"✅ 「{name}」を削除しました", _custom_provider_names()


def _custom_provider_names() -> list[str]:
    return list(_get_custom_providers().keys())


def _custom_providers_table() -> str:
    providers = _get_custom_providers()
    if not providers:
        return "_カスタムプロバイダーは登録されていません_"
    lines = [
        "| 名前 | タイプ | ベースURL | モデル |",
        "|------|--------|-----------|--------|",
    ]
    for k, v in providers.items():
        lines.append(
            f"| `{k}` | `{v.get('type', '—')}` "
            f"| `{v.get('base_url', '—')}` "
            f"| `{v.get('default_model', '—')}` |"
        )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# タブ UI 構築
# ────────────────────────────────────────────────────────────────

_THEME_NAMES = ["MED Dark", "Soft Light", "Ocean Dark", "Monochrome", "Forest"]


def build_tab(provider_dd: "gr.Dropdown | None" = None) -> None:
    """Gradio Blocks コンテキスト内で設定タブを描画する。

    Args:
        provider_dd: チャットタブのプロバイダードロップダウン。
                     渡すとカスタムプロバイダーの追加/削除時に自動で選択肢が同期される。
    """

    with gr.Tabs():

        # ── テーマ ──────────────────────────────────────────────
        with gr.TabItem("🎨 テーマ"):
            gr.Markdown(
                "### テーマ設定\n"
                "テーマはブラウザの `localStorage` に保存されます（再読み込み後も維持）。\n"
                "ページ初回ロード時に一瞬デフォルトの MED Dark が表示されることがあります。"
            )

            theme_dd = gr.Dropdown(
                choices=_THEME_NAMES,
                value="MED Dark",
                label="テーマ選択",
                interactive=True,
            )

            system_chk = gr.Checkbox(
                value=False,
                label="OSのテーマ設定に従う（prefers-color-scheme）",
                info="チェックするとOS/ブラウザのダーク/ライト設定を自動検出します",
            )

            gr.HTML("""
<div id="med-syspref-box" style="
    padding: 10px 14px; border-radius: 6px;
    border: 1px solid var(--block-border-color, #444);
    font-size: 0.88rem; margin: 4px 0 8px;">
  <strong>ブラウザ検出カラーモード:</strong>
  <span id="med-syspref-label" style="margin-left: 8px; font-weight: 600;">—</span>
  <span style="color: var(--block-label-text-color, #888); font-size: 0.82rem; margin-left: 12px;">
    (OS / ブラウザ設定から <code>prefers-color-scheme</code> を読み取り)
  </span>
</div>
<script>
(function () {
    function update() {
        var el = document.getElementById("med-syspref-label");
        if (!el) return false;
        var dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        el.textContent = dark ? "🌙 ダーク" : "☀️ ライト";
        window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
            el.textContent = e.matches ? "🌙 ダーク" : "☀️ ライト";
        });
        return true;
    }
    var poll = setInterval(function () { if (update()) clearInterval(poll); }, 150);
})();
</script>
""")

            gr.Markdown("""
**ブラウザのシステム設定について:**
- OS の外観設定（Windows: テーマ色 / macOS: 外観モード / Linux: GTK テーマ）を `prefers-color-scheme` CSS メディアクエリで読み取ります
- Gradio はサーバーサイドで動くため、Python がブラウザ設定を直接取得することは**できません**
- 代わりにクライアントサイド JavaScript で検出し、CSS カスタムプロパティをリアルタイムに切り替えます
- ダーク → **MED Dark** / ライト → **Soft Light** に自動マッピング
- OS 設定が変わると（例: macOS の自動モード）リアルタイムで追従します
""")

            theme_dd.change(
                fn=None,
                inputs=[theme_dd],
                js="(t) => { window.MED_applyTheme && window.MED_applyTheme(t); }",
            )

            system_chk.change(
                fn=None,
                inputs=[system_chk],
                js="""(use) => {
    if (use) {
        window.MED_applySystemTheme && window.MED_applySystemTheme();
    } else {
        try {
            var saved = localStorage.getItem("med-theme") || "MED Dark";
            window.MED_applyTheme && window.MED_applyTheme(saved);
            localStorage.setItem("med-use-system", "false");
        } catch(e) {}
    }
}""",
            )

        # ── プロバイダー設定 ──────────────────────────────────────
        with gr.TabItem("🔌 プロバイダー設定"):
            with gr.Tabs():

                # ── プリセット ───────────────────────────────────
                with gr.TabItem("プリセット"):
                    gr.Markdown(
                        "### プロバイダープリセット\n"
                        "プリセットを選択して適用すると `configs/llm_config.yaml` が上書きされます。"
                    )
                    with gr.Row():
                        preset_dd = gr.Dropdown(
                            choices=list(_PROVIDER_PRESETS.keys()),
                            value="Anthropic Claude",
                            label="プリセット",
                            scale=2,
                        )
                        preset_apply_btn = gr.Button(
                            "適用", variant="primary", scale=1
                        )
                    preset_preview = gr.Markdown(
                        _preset_summary_md("Anthropic Claude")
                    )
                    preset_result = gr.Markdown()

                    preset_dd.change(
                        fn=_preset_summary_md,
                        inputs=[preset_dd],
                        outputs=[preset_preview],
                    )
                    preset_apply_btn.click(
                        fn=_apply_preset,
                        inputs=[preset_dd],
                        outputs=[preset_result],
                    )

                # ── カスタムプロバイダー ─────────────────────────
                with gr.TabItem("カスタムプロバイダー"):
                    gr.Markdown(
                        "### カスタムプロバイダー追加\n"
                        "OpenAI互換エンドポイントや独自サーバーを登録します。"
                    )

                    with gr.Group():
                        with gr.Row():
                            cp_name = gr.Textbox(
                                label="プロバイダー名",
                                placeholder="例: my_llm",
                                scale=2,
                            )
                            cp_type = gr.Dropdown(
                                choices=["openai_compatible", "ollama", "other"],
                                value="openai_compatible",
                                label="タイプ",
                                scale=1,
                            )
                        cp_base_url = gr.Textbox(
                            label="ベースURL",
                            placeholder="例: https://api.example.com/v1",
                        )
                        cp_model = gr.Textbox(
                            label="デフォルトモデル",
                            placeholder="例: meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        )
                        with gr.Row():
                            cp_env_name = gr.Textbox(
                                label="APIキー 環境変数名",
                                placeholder="例: MY_LLM_API_KEY",
                                scale=2,
                            )
                            cp_api_key = gr.Textbox(
                                label="APIキー値 (セッション中のみ有効)",
                                placeholder="sk-...",
                                type="password",
                                scale=3,
                            )

                    cp_add_btn = gr.Button("追加 / 更新", variant="primary")
                    cp_add_result = gr.Markdown()

                    gr.Markdown("---\n### 登録済みカスタムプロバイダー")
                    cp_table = gr.Markdown(_custom_providers_table())

                    with gr.Row():
                        cp_del_dd = gr.Dropdown(
                            choices=_custom_provider_names(),
                            label="削除するプロバイダー",
                            scale=3,
                        )
                        cp_del_btn = gr.Button("削除", variant="stop", scale=1)
                    cp_del_result = gr.Markdown()

                    if provider_dd is not None:
                        def _on_add(name, ptype, url, model, env_name, key):
                            msg = _add_custom_provider(name, ptype, url, model, env_name, key)
                            names = _custom_provider_names()
                            all_choices = get_all_provider_choices()
                            return (
                                msg,
                                _custom_providers_table(),
                                gr.update(choices=names, value=names[-1] if names else None),
                                gr.update(choices=all_choices),
                            )

                        cp_add_btn.click(
                            fn=_on_add,
                            inputs=[cp_name, cp_type, cp_base_url, cp_model,
                                    cp_env_name, cp_api_key],
                            outputs=[cp_add_result, cp_table, cp_del_dd, provider_dd],
                        )

                        def _on_delete(name):
                            msg, names = _delete_custom_provider(name)
                            all_choices = get_all_provider_choices()
                            return (
                                msg,
                                _custom_providers_table(),
                                gr.update(choices=names, value=names[0] if names else None),
                                gr.update(choices=all_choices),
                            )

                        cp_del_btn.click(
                            fn=_on_delete,
                            inputs=[cp_del_dd],
                            outputs=[cp_del_result, cp_table, cp_del_dd, provider_dd],
                        )
                    else:
                        def _on_add(name, ptype, url, model, env_name, key):
                            msg = _add_custom_provider(name, ptype, url, model, env_name, key)
                            names = _custom_provider_names()
                            return (
                                msg,
                                _custom_providers_table(),
                                gr.update(choices=names, value=names[-1] if names else None),
                            )

                        cp_add_btn.click(
                            fn=_on_add,
                            inputs=[cp_name, cp_type, cp_base_url, cp_model,
                                    cp_env_name, cp_api_key],
                            outputs=[cp_add_result, cp_table, cp_del_dd],
                        )

                        def _on_delete(name):
                            msg, names = _delete_custom_provider(name)
                            return (
                                msg,
                                _custom_providers_table(),
                                gr.update(choices=names, value=names[0] if names else None),
                            )

                        cp_del_btn.click(
                            fn=_on_delete,
                            inputs=[cp_del_dd],
                            outputs=[cp_del_result, cp_table, cp_del_dd],
                        )

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
                    with gr.Column(scale=2):
                        gr.Markdown(f"**{label}**  現在値: `{display}`")
                    new_key = gr.Textbox(
                        placeholder=f"{env_key} を入力…",
                        type="password",
                        show_label=False,
                        scale=3,
                    )
                    apply_btn = gr.Button("適用", scale=1)
                    with gr.Column(scale=2):
                        result_md = gr.Markdown()

                    def _apply(value, _env=env_key):
                        if value.strip():
                            os.environ[_env] = value.strip()
                            return f"✅ `{_env}` をセッションに設定しました"
                        return "❌ 空のキーは設定できません"

                    apply_btn.click(fn=_apply, inputs=[new_key], outputs=[result_md])

            _key_row("Anthropic API Key",  "ANTHROPIC_API_KEY")
            _key_row("OpenAI API Key",     "OPENAI_API_KEY")
            _key_row("Azure OpenAI Key",   "AZURE_OPENAI_API_KEY")
            _key_row("Together.ai Key",    "TOGETHER_API_KEY")
            _key_row("GitHub Token",       "GITHUB_TOKEN")
            _key_row("Tavily API Key",     "TAVILY_API_KEY")
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

            file_selector.change(
                fn=_read_config_yaml,
                inputs=[file_selector],
                outputs=[yaml_editor],
            )
            reload_btn.click(
                fn=_read_config_yaml,
                inputs=[file_selector],
                outputs=[yaml_editor],
            )
            save_btn.click(
                fn=_write_config_yaml,
                inputs=[file_selector, yaml_editor],
                outputs=[save_result],
            )

        # ── システム情報 ─────────────────────────────────────────
        with gr.TabItem("システム情報"):
            gr.Markdown("### 現在の設定サマリー")

            def _get_system_info() -> str:
                default_cfg = _load_yaml(_CONFIGS_DIR / "default.yaml")
                llm_cfg     = _load_yaml(_CONFIGS_DIR / "llm_config.yaml")

                anthropic_key = _get_env("ANTHROPIC_API_KEY")
                openai_key    = _get_env("OPENAI_API_KEY")
                azure_key     = _get_env("AZURE_OPENAI_API_KEY")
                together_key  = _get_env("TOGETHER_API_KEY")

                primary   = llm_cfg.get("primary_provider", "—")
                providers = llm_cfg.get("providers", {})

                lines = [
                    "| 項目 | 値 |",
                    "|------|-----|",
                    f"| アプリ名 | `{default_cfg.get('app', {}).get('name', '—')}` |",
                    f"| バージョン | `{default_cfg.get('app', {}).get('version', '—')}` |",
                    f"| ホスト | `{default_cfg.get('app', {}).get('host', '—')}:{default_cfg.get('app', {}).get('port', '—')}` |",
                    f"| 埋め込みモデル | `{default_cfg.get('embedding', {}).get('model', '—')}` |",
                    f"| 埋め込み次元 | `{default_cfg.get('embedding', {}).get('dim', '—')}` |",
                    f"| 主要プロバイダー | `{primary}` |",
                    f"| Anthropicモデル | `{providers.get('anthropic', {}).get('default_model', '—')}` |",
                    f"| OpenAIモデル | `{providers.get('openai', {}).get('default_model', '—')}` |",
                    f"| OllamaURL | `{providers.get('ollama', {}).get('base_url', '—')}` |",
                    f"| ANTHROPIC_API_KEY | `{'設定済 ' + _mask(anthropic_key) if anthropic_key else '未設定'}` |",
                    f"| OPENAI_API_KEY | `{'設定済 ' + _mask(openai_key) if openai_key else '未設定'}` |",
                    f"| AZURE_OPENAI_API_KEY | `{'設定済 ' + _mask(azure_key) if azure_key else '未設定'}` |",
                    f"| TOGETHER_API_KEY | `{'設定済 ' + _mask(together_key) if together_key else '未設定'}` |",
                ]

                custom = {k: v for k, v in providers.items()
                          if k not in _KNOWN_PROVIDERS}
                if custom:
                    lines.append(f"| カスタムプロバイダー数 | `{len(custom)}` |")

                return "\n".join(lines)

            info_md = gr.Markdown(_get_system_info())
            gr.Button("更新", variant="secondary").click(
                fn=_get_system_info, outputs=[info_md]
            )

"""src/gui/tabs/training.py — 学習ダッシュボードタブ。

GRPO + TinyLoRA 学習の設定・制御・モニタリングを提供する。
training.pipeline が未実装またはAPIオフライン時はモックUIを表示。
"""

from __future__ import annotations

import random
import time
from typing import Optional

import gradio as gr
import httpx

_ORCHESTRATOR_URL = "http://localhost:8000"


# ────────────────────────────────────────────────────────────────
# ヘルパー
# ────────────────────────────────────────────────────────────────

def _is_api_alive() -> bool:
    try:
        r = httpx.get(f"{_ORCHESTRATOR_URL}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def _get_training_status() -> dict:
    if _is_api_alive():
        try:
            r = httpx.get(f"{_ORCHESTRATOR_URL}/training/status", timeout=5.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return {
        "status": "idle",
        "step": 0,
        "total_steps": 0,
        "reward": None,
        "loss": None,
        "algorithm": "—",
        "adapter": "—",
        "api_connected": False,
    }


def _start_training(algorithm: str, adapter: str, reward: str, steps: int) -> dict:
    if _is_api_alive():
        try:
            r = httpx.post(
                f"{_ORCHESTRATOR_URL}/training/start",
                json={
                    "algorithm": algorithm,
                    "adapter": adapter,
                    "reward": reward,
                    "total_steps": steps,
                },
                timeout=10.0,
            )
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {
        "success": False,
        "error": "APIオフライン — 学習を開始できません。オーケストレーターを起動してください。",
    }


def _stop_training() -> dict:
    if _is_api_alive():
        try:
            r = httpx.post(f"{_ORCHESTRATOR_URL}/training/stop", timeout=5.0)
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "APIオフライン"}


# ────────────────────────────────────────────────────────────────
# タブ UI 構築
# ────────────────────────────────────────────────────────────────

def build_tab() -> None:
    """Gradio Blocks コンテキスト内で学習タブを描画する。"""

    with gr.Tabs():

        # ── 設定・制御 ──────────────────────────────────────────
        with gr.TabItem("学習制御"):
            gr.Markdown("### 学習パラメーター設定")

            with gr.Row():
                algorithm_dd = gr.Dropdown(
                    choices=["grpo", "ppo", "dpo", "sft", "reinforce"],
                    value="grpo",
                    label="学習アルゴリズム",
                    info="デフォルト: GRPO (Group Relative Policy Optimization)",
                )
                adapter_dd = gr.Dropdown(
                    choices=["tinylora", "lora", "lora_xs", "full_ft"],
                    value="tinylora",
                    label="パラメータアダプタ",
                    info="デフォルト: TinyLoRA (極少パラメータ)",
                )
                reward_dd = gr.Dropdown(
                    choices=["composite", "code_exec", "teacher_eval", "hybrid"],
                    value="composite",
                    label="Reward関数",
                    info="デフォルト: 複合Reward (多信号加重)",
                )

            with gr.Row():
                steps_slider = gr.Slider(
                    100, 10000, value=1000, step=100, label="総ステップ数"
                )
                batch_slider = gr.Slider(
                    4, 64, value=8, step=4, label="バッチサイズ"
                )
                lr_slider = gr.Slider(
                    1e-6, 1e-3, value=1e-4, step=1e-6, label="学習率"
                )

            gr.Markdown("#### TinyLoRA 設定 (デフォルト: Morris et al., 2026)")
            with gr.Row():
                frozen_rank = gr.Number(value=2, label="frozen_rank", precision=0)
                proj_dim = gr.Number(value=4, label="projection_dim", precision=0)
                tie_factor = gr.Number(value=7, label="tie_factor", precision=0)

            gr.Markdown("#### Composite Reward 重み配分")
            with gr.Row():
                w_correctness = gr.Slider(0, 1, value=0.35, label="correctness")
                w_retrieval = gr.Slider(0, 1, value=0.20, label="retrieval_quality")
                w_exec = gr.Slider(0, 1, value=0.20, label="exec_success")
                w_efficiency = gr.Slider(0, 1, value=0.10, label="efficiency")
                w_memory = gr.Slider(0, 1, value=0.15, label="memory_utilization")

            with gr.Row():
                start_btn = gr.Button("▶  学習開始", variant="primary", scale=2)
                stop_btn = gr.Button("⏹  停止", variant="stop", scale=1)

            train_result = gr.Markdown()

            def _on_start(algo, adapter, reward, steps):
                result = _start_training(algo, adapter, reward, int(steps))
                if result.get("success"):
                    return f"✅ 学習を開始しました (job_id: `{result.get('job_id', '—')}`)"
                return f"❌ {result.get('error', '不明なエラー')}"

            def _on_stop():
                result = _stop_training()
                if result.get("success"):
                    return "⏹ 学習を停止しました"
                return f"❌ {result.get('error', '不明なエラー')}"

            start_btn.click(
                fn=_on_start,
                inputs=[algorithm_dd, adapter_dd, reward_dd, steps_slider],
                outputs=[train_result],
            )
            stop_btn.click(fn=_on_stop, outputs=[train_result])

        # ── 進捗モニター ────────────────────────────────────────
        with gr.TabItem("進捗モニター"):
            gr.Markdown("### 学習進捗")

            status_md = gr.Markdown("_更新ボタンで最新状態を取得_")

            with gr.Row():
                reward_plot = gr.LinePlot(
                    x="step",
                    y="reward",
                    title="Reward推移",
                    x_title="ステップ",
                    y_title="Reward",
                    height=280,
                )
                loss_plot = gr.LinePlot(
                    x="step",
                    y="loss",
                    title="Loss推移",
                    x_title="ステップ",
                    y_title="Loss",
                    height=280,
                )

            refresh_btn = gr.Button("更新", variant="secondary")

            def _refresh_status():
                info = _get_training_status()
                connected = info.get("api_connected", False)
                step = info.get("step", 0)
                total = info.get("total_steps", 0)
                progress_pct = f"{step / total * 100:.1f}%" if total > 0 else "—"
                status_text = (
                    f"**状態:** `{info.get('status', '—')}` | "
                    f"**進捗:** {step}/{total} ({progress_pct}) | "
                    f"**アルゴリズム:** `{info.get('algorithm', '—')}` | "
                    f"**アダプタ:** `{info.get('adapter', '—')}`\n\n"
                    + ("🟢 APIオンライン" if connected else "🔴 APIオフライン — モックデータ")
                )

                # モックのグラフデータ生成
                import pandas as pd
                n = max(step, 10) if connected else 50
                steps = list(range(0, n, max(1, n // 50)))
                reward_data = pd.DataFrame({
                    "step": steps,
                    "reward": [0.3 + 0.5 * (1 - 1 / (1 + s / n * 5)) + random.gauss(0, 0.02) for s in steps],
                })
                loss_data = pd.DataFrame({
                    "step": steps,
                    "loss": [2.0 * (1 / (1 + s / n * 3)) + random.gauss(0, 0.05) for s in steps],
                })
                return status_text, reward_data, loss_data

            refresh_btn.click(
                fn=_refresh_status,
                outputs=[status_md, reward_plot, loss_plot],
            )

        # ── アルゴリズム説明 ────────────────────────────────────
        with gr.TabItem("アルゴリズム説明"):
            gr.Markdown("""
### 利用可能な学習コンポーネント

#### アルゴリズム
| 名前 | 説明 | 推奨ユースケース |
|------|------|----------------|
| **GRPO** | Group Relative Policy Optimization (デフォルト) | メモリ検索スキル学習 |
| **PPO** | Proximal Policy Optimization | 安定した強化学習 |
| **DPO** | Direct Preference Optimization | ペアデータから好み学習 |
| **SFT** | Supervised Fine-Tuning | ウォームアップフェーズ |
| **REINFORCE** | モンテカルロ政策勾配 | シンプルなタスク |

#### パラメータアダプタ
| 名前 | パラメータ数 | 説明 |
|------|------------|------|
| **TinyLoRA** | ~13パラメータ | frozen_rank=2, proj=4, tie=7 (Morris et al., 2026) |
| **LoRA** | ~数MB | 標準的な低ランク適応 |
| **LoRA-XS** | TinyLoRAの中間 | より多くのパラメータ |
| **Full FT** | 全パラメータ | 精度最大、コスト大 |

#### Reward関数
| 名前 | 構成 |
|------|------|
| **Composite** | correctness(0.35) + retrieval(0.20) + exec(0.20) + efficiency(0.10) + memory(0.15) |
| **code_exec** | コード実行成功率のみ |
| **teacher_eval** | Teacher LLMによる評価 |
| **hybrid** | code_exec + teacher_eval |

#### 3段階学習パイプライン
1. **SFTウォームアップ** — 基本的な回答フォーマット習得
2. **GRPO + TinyLoRA** — メモリ検索・利用スキルのRL学習
3. **評価** — Benchmark Suite で性能測定
""")

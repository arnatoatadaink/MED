# RAG × FAISS × LLM × Memory Environment Distillation
# プロジェクト計画書 v4

## 版管理

| 版 | 主要変更 |
|----|---------|
| v1 | MemN2Nベースの初期設計 |
| v2 | MemN2N → FAISS置換、Iterative Retrieval導入 |
| v3 | LLM統合、学習可能検索（LTR/Cross-Encoder/Adapter）、有用性管理 |
| v4（本版） | Teacher-Student二層構造、GRPO+TinyLoRA学習、拡張可能な学習フレームワーク |

## v3 → v4 変更サマリ

| 項目 | v3 | v4 |
|------|-----|-----|
| モデル構成 | 大モデル単体 | Teacher（大）+ Student（小）二層構成 |
| 学習方法 | LTR/Cross-Encoderのみ | GRPO+TinyLoRA + 拡張可能な学習IF |
| メモリ運用 | リアルタイム蓄積 | Teacher主導の成熟プロセス追加 |
| 推論コスト | 全クエリ大モデル（~$0.07） | 70%小モデル処理（~$0.01平均） |
| オフライン | 不可 | 小モデルでローカル推論可能 |
| ドメイン特化 | プロンプト調整 | TinyLoRAアダプタ切替（~1KB） |
| v3の内容 | — | 全て包含（v4はv3の上位互換） |

---

## 1. システム概要

### 1.1 目的

Teacher Model（大モデル）が外部検索・裏どり・実行検証を通じてFAISSメモリを成熟させ、
Student Model（小モデル）にRL（GRPO+TinyLoRA等）で「メモリの使い方」を極少パラメータで教える。
Docker Sandbox上でのコード実行結果を検証可能な報酬として活用する統合システム。

### 1.2 着想（TinyLoRA論文からの知見）

```
TinyLoRA論文: 「知識は大モデルに既にある。RLは使い方だけを教える」
   → 13パラメータでGSM8K 91%（Qwen2.5-8B + GRPO）

本プロジェクトへの応用:
   「知識はFAISSメモリに蓄積する。小モデルにはRLで検索・利用方法だけ教える」
   → 外部メモリという"知識の義肢"により、小モデルの能力上限を引き上げる
```

### 1.3 アーキテクチャ全体図

```
┌────────────────────────────────────────────────────────────────────────┐
│                           Client / CLI                                  │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────┐
│                        Orchestrator (FastAPI)                           │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     Model Router                                  │  │
│  │  simple → Student    moderate → Student+RAG    complex → Teacher  │  │
│  └──────────┬────────────────────┬────────────────────┬─────────────┘  │
│             │                    │                    │                 │
│  ┌──────────▼──────────┐  ┌─────▼─────┐  ┌──────────▼──────────┐    │
│  │   Student Model      │  │           │  │   Teacher Model      │    │
│  │   (Qwen-7B/8B等)     │  │  Shared   │  │   (Claude/GPT-4o)    │    │
│  │   + TinyLoRA Adapter │  │           │  │                      │    │
│  │                      │  │  FAISS    │  │  役割:                │    │
│  │  役割:               │  │  Memory   │  │  • メモリ成熟         │    │
│  │  • メモリ検索+利用   │  │  Module   │  │  • 裏どり・品質審査   │    │
│  │  • 回答生成          │  │           │  │  • Reward生成         │    │
│  │  • コード生成        │  │           │  │  • 複雑クエリ処理     │    │
│  └──────────┬───────────┘  │           │  └──────────┬───────────┘    │
│             │              │           │             │                 │
│             └──────────────▶           ◀─────────────┘                 │
│                            │           │                               │
│                            │  ┌─────┐ │                               │
│                            │  │ LTR │ │                               │
│                            │  │ CE  │ │                               │
│                            │  │Score│ │                               │
│                            │  └─────┘ │                               │
│                            └─────┬─────┘                               │
│                                  │                                     │
│  ┌───────────────────────────────▼─────────────────────────────────┐  │
│  │                 Docker Sandbox Manager                           │  │
│  │  Teacher: 実行検証 → メモリ品質向上                              │  │
│  │  Student: コード生成 → 実行 → Reward信号                        │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              Training Framework（拡張可能）                      │  │
│  │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐           │  │
│  │  │ GRPO +  │ │ PPO +    │ │ DPO +  │ │ SFT +     │  ...      │  │
│  │  │ TinyLoRA│ │ LoRA     │ │ LoRA-XS│ │ Full FT   │           │  │
│  │  └─────────┘ └──────────┘ └────────┘ └───────────┘           │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
       │              │              │                    │
┌──────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐   ┌────────▼────────┐
│ Embedding    │ │ Teacher  │ │ Student    │   │ Search APIs      │
│ Model        │ │ LLM API  │ │ (vLLM)    │   │ (GitHub,SO,etc.) │
└─────────────┘ └──────────┘ └────────────┘   └─────────────────┘
```

---

## 2. 二層モデル設計

### 2.1 Teacher Model（大モデル）

```
┌─────────────────────────────────────────────────────────────┐
│  Teacher Model の責務                                        │
│                                                              │
│  ■ メモリ成熟（Phase A）                                    │
│    • 外部API検索 → 裏どり → チャンク化 → FAISS蓄積         │
│    • Sandbox実行 → 成功/失敗記録 → 有用性スコア更新        │
│    • メモリ全体レビュー → 品質スコア・難易度タグ付与        │
│    • 古い情報の再検証 → 更新 or アーカイブ                  │
│                                                              │
│  ■ Reward生成（Phase B）                                    │
│    • Student回答の品質評価 → スカラー報酬                   │
│    • 検索行動の評価 → 適切なメモリを選べたか判定            │
│    • コード実行結果の分析 → 部分報酬の設計                  │
│                                                              │
│  ■ 複雑クエリ処理（Phase C 運用時）                         │
│    • Model Routerが「complex」判定したクエリに直接対応       │
│    • Studentが失敗したクエリのフォールバック                 │
│    • 対応結果をFAISSに蓄積（メモリの継続的成長）            │
│                                                              │
│  プロバイダ切替:                                             │
│    Claude Sonnet / GPT-4o / Claude Haiku（タスク別）         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Student Model（小モデル）

```
┌─────────────────────────────────────────────────────────────┐
│  Student Model の責務と学習                                   │
│                                                              │
│  ■ 推論時の動作                                             │
│    1. クエリを受け取る                                       │
│    2. FAISSメモリを検索（Iterative Retrieval）               │
│    3. 検索結果を統合して回答を生成                           │
│    4. コード生成が必要ならSandboxで実行                     │
│                                                              │
│  ■ 学習対象（TinyLoRA的に極少パラメータ）                   │
│    • いつ検索するか（検索トリガー判定）                     │
│    • 何を検索するか（検索クエリ生成）                       │
│    • どれを使うか（検索結果選択）                           │
│    • どう統合するか（回答構成）                             │
│                                                              │
│  ■ ベースモデル候補                                         │
│    • Qwen2.5-7B-Instruct （TinyLoRA論文で最も効率的）       │
│    • Llama-3.1-8B-Instruct                                  │
│    • Mistral-7B-Instruct                                    │
│    • ローカルGPU1枚（A100/4090）で推論可能なサイズ          │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Model Router（クエリ振り分け）

```python
# src/orchestrator/model_router.py

class ModelRouter:
    """クエリの複雑度に応じてTeacher/Studentを選択"""

    async def route(self, query: str, parsed: ParsedQuery) -> ModelChoice:

        # Level 1: 単純なクエリ → Student直接回答
        #   例: 「Pythonでリストのソート方法」
        #   条件: FAISSに類似度0.85以上の結果が3件以上
        if parsed.complexity == "simple":
            memory_results = await self.memory.search(query, k=5)
            if memory_results[0].score > 0.85:
                return ModelChoice.STUDENT

        # Level 2: 中程度 → Student + 外部RAG補完
        #   例: 「FastAPIでWebSocket認証を実装」
        #   条件: FAISSに部分的な情報あり、外部補完が必要
        if parsed.complexity == "moderate":
            return ModelChoice.STUDENT_WITH_RAG

        # Level 3: 複雑 → Teacher直接処理
        #   例: 「分散システムのCAP定理を踏まえたDB選定」
        #   条件: マルチステップ推論、メモリに情報不足
        if parsed.complexity == "complex":
            return ModelChoice.TEACHER

        # Level 4: Student失敗時のフォールバック
        #   Studentの回答品質が閾値未満 → Teacherに再ルーティング
        return ModelChoice.STUDENT_WITH_TEACHER_FALLBACK
```

---

## 3. 拡張可能な学習フレームワーク（★ 核心設計）

### 3.1 設計思想

```
要件:
  1. GRPO + TinyLoRA をデフォルト実装として提供
  2. PPO, DPO, SFT, REINFORCE 等を同一IFで差し替え可能
  3. TinyLoRA, LoRA, LoRA-XS, Full FT をアダプタとして切替可能
  4. Reward関数をプラガブルに差し替え可能
  5. 学習の記録・比較・再現が容易

実現方法:
  Strategy Pattern + Registry Pattern で全コンポーネントを抽象化
```

### 3.2 抽象インターフェース

```python
# src/training/base.py

from abc import ABC, abstractmethod

# ─── 学習アルゴリズム抽象 ───

class TrainingAlgorithm(ABC):
    """学習アルゴリズムの基底クラス"""

    @abstractmethod
    async def train_step(
        self,
        batch: TrainingBatch,
        reward_fn: RewardFunction,
        adapter: ParameterAdapter,
    ) -> TrainStepResult:
        """1ステップの学習を実行"""
        ...

    @abstractmethod
    def get_config(self) -> dict:
        """再現用の設定を返す"""
        ...


# ─── パラメータアダプタ抽象 ───

class ParameterAdapter(ABC):
    """パラメータ効率的学習の基底クラス"""

    @abstractmethod
    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        ...

    @abstractmethod
    def get_param_count(self) -> int:
        ...

    @abstractmethod
    def merge_weights(self, base_model: nn.Module) -> nn.Module:
        """推論用にベースモデルへマージ"""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...


# ─── Reward関数抽象 ───

class RewardFunction(ABC):
    """報酬関数の基底クラス"""

    @abstractmethod
    async def compute(
        self,
        query: str,
        response: str,
        retrieved_docs: list[Document],
        execution_result: ExecutionResult | None,
    ) -> RewardSignal:
        ...


# ─── Reward信号 ───

@dataclass
class RewardSignal:
    total: float                          # 加重合計スカラー
    components: dict[str, float]          # 各要素の内訳
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3.3 アルゴリズム Registry

```python
# src/training/registry.py

class TrainingRegistry:
    """学習コンポーネントのプラグイン管理"""

    _algorithms: dict[str, type[TrainingAlgorithm]] = {}
    _adapters: dict[str, type[ParameterAdapter]] = {}
    _rewards: dict[str, type[RewardFunction]] = {}

    @classmethod
    def register_algorithm(cls, name: str):
        def decorator(klass):
            cls._algorithms[name] = klass
            return klass
        return decorator

    @classmethod
    def register_adapter(cls, name: str):
        def decorator(klass):
            cls._adapters[name] = klass
            return klass
        return decorator

    @classmethod
    def register_reward(cls, name: str):
        def decorator(klass):
            cls._rewards[name] = klass
            return klass
        return decorator

    @classmethod
    def create_trainer(cls, config: TrainingConfig) -> TrainingAlgorithm:
        return cls._algorithms[config.algorithm](**config.algorithm_kwargs)

    @classmethod
    def create_adapter(cls, config: TrainingConfig) -> ParameterAdapter:
        return cls._adapters[config.adapter](**config.adapter_kwargs)

    @classmethod
    def create_reward(cls, config: TrainingConfig) -> RewardFunction:
        return cls._rewards[config.reward](**config.reward_kwargs)
```

### 3.4 デフォルト実装: GRPO + TinyLoRA

```python
# src/training/algorithms/grpo.py

@TrainingRegistry.register_algorithm("grpo")
class GRPOTrainer(TrainingAlgorithm):
    """Group Relative Policy Optimization"""

    def __init__(
        self,
        group_size: int = 8,
        kl_coeff: float = 0.001,
        clip_ratio: float = 0.2,
        temperature: float = 1.0,
    ):
        self.group_size = group_size
        self.kl_coeff = kl_coeff
        self.clip_ratio = clip_ratio
        self.temperature = temperature

    async def train_step(self, batch, reward_fn, adapter):
        # 1. 各promptに対してgroup_size個の応答を生成
        responses = await self._generate_group(batch.queries, adapter)

        # 2. 各応答にRewardを計算
        rewards = []
        for query, response_group in zip(batch.queries, responses):
            group_rewards = []
            for resp in response_group:
                signal = await reward_fn.compute(
                    query=query,
                    response=resp.text,
                    retrieved_docs=resp.retrieved_docs,
                    execution_result=resp.exec_result,
                )
                group_rewards.append(signal.total)
            rewards.append(group_rewards)

        # 3. グループ内の相対的な優位性を計算
        advantages = self._compute_advantages(rewards)

        # 4. Policy gradient更新（adapter内のパラメータのみ）
        loss = self._policy_gradient_loss(responses, advantages, adapter)
        loss.backward()

        return TrainStepResult(loss=loss.item(), mean_reward=np.mean(rewards))
```

```python
# src/training/adapters/tinylora.py

@TrainingRegistry.register_adapter("tinylora")
class TinyLoRAAdapter(ParameterAdapter):
    """TinyLoRA: 極少パラメータアダプタ"""

    def __init__(
        self,
        model: nn.Module,
        frozen_rank: int = 2,
        projection_dim: int = 4,        # u: 各モジュールの学習次元
        tie_factor: int = 7,             # n_tie: 重み共有モジュール数
        target_modules: list[str] = None, # 適用対象レイヤー
    ):
        self.frozen_rank = frozen_rank
        self.projection_dim = projection_dim
        self.tie_factor = tie_factor

        # 各対象レイヤーのSVD分解
        self.svd_cache = self._compute_svd(model, target_modules)

        # 学習可能パラメータ: v ∈ R^u（共有）
        n_groups = self._count_groups(model, target_modules, tie_factor)
        self.trainable_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(projection_dim) * 0.01)
            for _ in range(n_groups)
        ])

        # 固定ランダム射影テンソル: P ∈ R^{u×r×r}
        self.projections = self._init_random_projections(
            projection_dim, frozen_rank
        )

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_vectors)

    def apply_update(self, layer_idx: int, module_idx: int) -> torch.Tensor:
        """W' = W + U Σ (Σ v_i P_i) V^T"""
        group_idx = (layer_idx * self.modules_per_layer + module_idx) // self.tie_factor
        v = self.trainable_vectors[group_idx]
        P = self.projections[group_idx]

        U, S, Vt = self.svd_cache[(layer_idx, module_idx)]
        R = torch.einsum('i,ijk->jk', v, P)
        delta_W = U @ (S.diag() @ R) @ Vt
        return delta_W

    def save(self, path: str):
        # 数百バイト〜数KBのファイル
        torch.save({
            'vectors': [v.data for v in self.trainable_vectors],
            'config': self.get_config(),
        }, path)
```

### 3.5 拡張実装の例

```python
# src/training/algorithms/ppo.py
@TrainingRegistry.register_algorithm("ppo")
class PPOTrainer(TrainingAlgorithm):
    """Proximal Policy Optimization — クリッピング方式"""
    ...

# src/training/algorithms/dpo.py
@TrainingRegistry.register_algorithm("dpo")
class DPOTrainer(TrainingAlgorithm):
    """Direct Preference Optimization — Rewardモデル不要"""
    ...

# src/training/algorithms/reinforce.py
@TrainingRegistry.register_algorithm("reinforce")
class REINFORCETrainer(TrainingAlgorithm):
    """REINFORCE with baseline — シンプルなpolicy gradient"""
    ...

# src/training/algorithms/sft.py
@TrainingRegistry.register_algorithm("sft")
class SFTTrainer(TrainingAlgorithm):
    """Supervised Fine-Tuning — 教師データ模倣"""
    ...


# src/training/adapters/lora.py
@TrainingRegistry.register_adapter("lora")
class LoRAAdapter(ParameterAdapter):
    """標準LoRA (rank 1〜256)"""
    ...

# src/training/adapters/lora_xs.py
@TrainingRegistry.register_adapter("lora_xs")
class LoRAXSAdapter(ParameterAdapter):
    """LoRA-XS (r^2 パラメータ/モジュール)"""
    ...

# src/training/adapters/full_ft.py
@TrainingRegistry.register_adapter("full_ft")
class FullFinetuneAdapter(ParameterAdapter):
    """全パラメータ更新（ベースライン比較用）"""
    ...
```

### 3.6 Reward関数の拡張

```python
# src/training/rewards/composite.py
@TrainingRegistry.register_reward("composite")
class CompositeReward(RewardFunction):
    """デフォルト: 複数信号の加重合計"""

    def __init__(self, teacher: LLMGateway, sandbox: SandboxManager):
        self.teacher = teacher
        self.sandbox = sandbox
        self.weights = {
            "correctness": 0.35,
            "retrieval_quality": 0.20,
            "exec_success": 0.20,
            "efficiency": 0.10,
            "memory_utilization": 0.15,
        }

    async def compute(self, query, response, retrieved_docs, execution_result):
        components = {}

        # R1: 正確性（検証可能なら自動、不可なら大モデル評価）
        components["correctness"] = await self._eval_correctness(query, response)

        # R2: 検索品質
        components["retrieval_quality"] = self._eval_retrieval(query, retrieved_docs)

        # R3: コード実行
        components["exec_success"] = self._eval_execution(execution_result)

        # R4: 効率性（少ない検索回数）
        components["efficiency"] = max(0, 1.0 - len(retrieved_docs) / 20)

        # R5: メモリ活用（高品質ドキュメントを選べたか）
        components["memory_utilization"] = self._eval_memory_use(retrieved_docs)

        total = sum(self.weights[k] * components[k] for k in self.weights)
        return RewardSignal(total=total, components=components)


# src/training/rewards/code_exec.py
@TrainingRegistry.register_reward("code_exec")
class CodeExecutionReward(RewardFunction):
    """コード特化: Sandbox実行結果のみで報酬（大モデル不要）"""
    ...

# src/training/rewards/teacher_eval.py
@TrainingRegistry.register_reward("teacher_eval")
class TeacherEvalReward(RewardFunction):
    """大モデル評価のみ（高コスト・高精度）"""
    ...

# src/training/rewards/hybrid.py
@TrainingRegistry.register_reward("hybrid")
class HybridReward(RewardFunction):
    """検証可能なものは自動、不可なものだけ大モデル（コスト最適化）"""
    ...
```

### 3.7 学習設定ファイル

```yaml
# configs/training.yaml

# --- デフォルト構成: GRPO + TinyLoRA ---
default:
  algorithm: "grpo"
  algorithm_kwargs:
    group_size: 8
    kl_coeff: 0.001
    clip_ratio: 0.2
    temperature: 1.0

  adapter: "tinylora"
  adapter_kwargs:
    frozen_rank: 2
    projection_dim: 4
    tie_factor: 7
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                     "up_proj", "down_proj", "gate_proj"]

  reward: "composite"
  reward_kwargs:
    teacher_model: "claude-haiku"
    weights:
      correctness: 0.35
      retrieval_quality: 0.20
      exec_success: 0.20
      efficiency: 0.10
      memory_utilization: 0.15

  training:
    epochs: 3
    batch_size: 64
    learning_rate: 1.0e-4
    lr_sweep: [1.0e-7, 5.0e-7, 1.0e-6, 5.0e-6, 1.0e-5, 1.0e-4]
    max_generation_length: 4096
    seed: [42, 123, 456]   # 3 seeds for variance estimation

  student_model:
    name: "Qwen/Qwen2.5-7B-Instruct"
    inference_engine: "vllm"

# --- 代替構成例 ---
alternatives:
  ppo_lora:
    algorithm: "ppo"
    adapter: "lora"
    adapter_kwargs:
      rank: 8
    reward: "composite"

  dpo_loraxs:
    algorithm: "dpo"
    adapter: "lora_xs"
    reward: "teacher_eval"

  sft_warmup:
    algorithm: "sft"
    adapter: "lora"
    adapter_kwargs:
      rank: 4
    reward: null   # SFTは報酬不要（教師データで学習）
```

---

## 4. メモリ成熟プロセス

### 4.1 Phase A: Teacherによるメモリ成熟

```
┌─────────────────────────────────────────────────────────────────────┐
│                 メモリ成熟パイプライン                                │
│                                                                     │
│  Step 1: シードデータ投入                                           │
│    • 頻出プログラミングパターン（GitHub/SO top questions）          │
│    • 主要ライブラリの公式ドキュメント                               │
│    • 既知のベストプラクティス集                                     │
│    → seed_memory.py で初期FAISS構築                                │
│                                                                     │
│  Step 2: Teacher主導の拡充                                          │
│    • 代表的なクエリセットを投入                                     │
│    • Teacher が 検索 → 裏どり → 蓄積 を繰り返す                   │
│    • 各ドキュメントに品質スコア・難易度タグを付与                   │
│                                                                     │
│  Step 3: 実行検証                                                   │
│    • コード片を全てSandboxで実行                                   │
│    • 成功/失敗/エラー種別を記録                                     │
│    • 実行失敗のコードはTeacherが修正→修正版も蓄積                  │
│                                                                     │
│  Step 4: 品質レビュー                                               │
│    • Teacherが全スロットをバッチ評価                                │
│    • 低品質（confidence < 0.3）を削除候補に                        │
│    • 重複排除、古い情報の更新                                      │
│                                                                     │
│  完了条件:                                                          │
│    • ドキュメント数 > 10,000                                       │
│    • 平均confidence > 0.7                                          │
│    • 実行成功率 > 80%（コード片）                                  │
│    • カバレッジ: 代表クエリセットの Hit@5 > 70%                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 メモリ品質管理モジュール

```python
# src/memory/maturation/reviewer.py

class MemoryReviewer:
    """Teacherによるメモリ品質審査"""

    async def batch_review(self, domain: str, batch_size: int = 100):
        docs = await self.metadata_store.get_unreviewed(domain, batch_size)

        for doc_batch in chunked(docs, 10):
            review = await self.teacher.complete(
                messages=[{
                    "role": "system",
                    "content": self.REVIEW_PROMPT
                }, {
                    "role": "user",
                    "content": self._format_batch(doc_batch)
                }],
                task_type=TaskType.FEEDBACK_ANALYSIS,
                response_format="json"
            )

            for doc, score in zip(doc_batch, review.scores):
                await self.metadata_store.update_quality(
                    doc.id,
                    teacher_quality_score=score.quality,
                    difficulty_tag=score.difficulty,
                    reviewed_at=datetime.now()
                )

# src/memory/maturation/difficulty_tagger.py

class DifficultyTagger:
    """難易度タグ付け — Student学習のカリキュラムに利用"""

    LEVELS = ["beginner", "intermediate", "advanced", "expert"]

    async def tag(self, doc: Document) -> str:
        # Teacherが文書の難易度を判定
        # → Student学習時に簡単→難しいの順で提示（カリキュラム学習）
        ...
```

---

## 5. Student学習フロー

### 5.1 三段階学習プロセス

```
┌─────────────────────────────────────────────────────────────────────┐
│            Student Model 学習フロー（三段階）                         │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Stage 1: 検索行動ウォームアップ（SFT）                      │    │
│  │                                                              │    │
│  │  目的: FAISSの基本的な使い方を教える                         │    │
│  │  データ: Teacherの検索行動ログ                               │    │
│  │    (query → search_query → selected_docs → response)        │    │
│  │  手法: SFT + LoRA rank=4                                    │    │
│  │  規模: 数千サンプル、1エポック                               │    │
│  │  期間: 数時間                                                │    │
│  │                                                              │    │
│  │  ※ TinyLoRA論文の知見:                                      │    │
│  │    SFTは多くのパラメータが必要 → ここではLoRA使用            │    │
│  │    この段階は「粗い知識注入」であり、精錬はRLで行う          │    │
│  └──────────────────┬───────────────────────────────────────┘    │
│                     │                                               │
│                     ▼                                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Stage 2: RL本学習（GRPO + TinyLoRA）   ★メイン学習         │    │
│  │                                                              │    │
│  │  目的: メモリの「使い方」を極少パラメータで最適化            │    │
│  │  手法: GRPO + TinyLoRA (数百パラメータ)                     │    │
│  │  環境: クエリ → FAISS検索 → 回答生成 → Reward               │    │
│  │                                                              │    │
│  │  カリキュラム:                                               │    │
│  │    Epoch 1: beginner難易度のクエリのみ                       │    │
│  │    Epoch 2: + intermediate                                   │    │
│  │    Epoch 3: + advanced                                       │    │
│  │                                                              │    │
│  │  Reward:                                                     │    │
│  │    • コード実行成功 (自動判定、低コスト)                     │    │
│  │    • 検索結果の関連度 (FAISS score、低コスト)                │    │
│  │    • Teacher評価 (サンプリング、高コスト)                    │    │
│  │                                                              │    │
│  │  手法切替（configで変更可能）:                                │    │
│  │    algorithm: grpo → ppo / dpo / reinforce                   │    │
│  │    adapter: tinylora → lora / lora_xs / full_ft              │    │
│  └──────────────────┬───────────────────────────────────────┘    │
│                     │                                               │
│                     ▼                                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Stage 3: 評価＋ドメイン特化（反復的）                       │    │
│  │                                                              │    │
│  │  評価:                                                       │    │
│  │    Teacher vs Student の品質比較                             │    │
│  │    → 品質比 > 80%: Student を推論パイプラインに投入          │    │
│  │    → 品質比 < 80%: Stage 2 を継続                           │    │
│  │                                                              │    │
│  │  ドメイン特化:                                               │    │
│  │    base + tinylora_code.bin  (500 params) → コード特化       │    │
│  │    base + tinylora_debug.bin (500 params) → デバッグ特化     │    │
│  │    base + tinylora_docs.bin  (500 params) → ドキュメント特化 │    │
│  │    → 同一ベースモデル + 異なるアダプタで複数ドメイン対応     │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 学習パイプライン実装

```python
# src/training/pipeline.py

class StudentTrainingPipeline:
    """Student学習の全体制御"""

    def __init__(self, config: TrainingConfig):
        self.algorithm = TrainingRegistry.create_trainer(config)
        self.adapter = TrainingRegistry.create_adapter(config)
        self.reward_fn = TrainingRegistry.create_reward(config)
        self.evaluator = StudentEvaluator()
        self.logger = TrainingLogger()

    async def run(self, stages: list[str] = ["warmup", "rl", "evaluate"]):

        if "warmup" in stages:
            await self._stage_warmup()

        if "rl" in stages:
            await self._stage_rl()

        if "evaluate" in stages:
            await self._stage_evaluate()

    async def _stage_rl(self):
        """Stage 2: GRPO + TinyLoRA 本学習"""

        for epoch in range(self.config.epochs):
            difficulty = self._get_curriculum_difficulty(epoch)
            batches = self._get_training_batches(difficulty)

            for batch in batches:
                result = await self.algorithm.train_step(
                    batch=batch,
                    reward_fn=self.reward_fn,
                    adapter=self.adapter,
                )
                self.logger.log_step(result)

            # エポック終了時の評価
            eval_result = await self.evaluator.evaluate(self.adapter)
            self.logger.log_epoch(epoch, eval_result)

            # Early stopping
            if eval_result.quality_ratio > 0.9:
                break

        # アダプタ保存
        self.adapter.save(self.config.output_path)

    async def _stage_evaluate(self):
        """Stage 3: Teacher vs Student 品質比較"""

        test_queries = await self._load_test_set()
        results = {"teacher": [], "student": []}

        for query in test_queries:
            teacher_resp = await self.teacher.generate(query)
            student_resp = await self.student.generate(query, self.adapter)

            teacher_score = await self.reward_fn.compute(
                query, teacher_resp, ...)
            student_score = await self.reward_fn.compute(
                query, student_resp, ...)

            results["teacher"].append(teacher_score.total)
            results["student"].append(student_score.total)

        quality_ratio = mean(results["student"]) / mean(results["teacher"])
        self.logger.log_evaluation(quality_ratio, results)
```

---

## 6. モジュール構成（統合版）

### 6.1 ディレクトリ構造

```
rag-faiss-llm/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.sandbox
├── Dockerfile.student              # ★ Student推論用
├── pyproject.toml
├── README.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── query_parser.py         # 🤖LLMベース意図分類
│   │   ├── model_router.py         # ★ Teacher/Student振り分け
│   │   └── pipeline.py
│   │
│   ├── llm/                        # v3から継続
│   │   ├── __init__.py
│   │   ├── gateway.py              # LLMプロバイダ抽象化
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── anthropic.py        # Teacher: Claude
│   │   │   ├── openai.py           # Teacher: GPT
│   │   │   ├── ollama.py           # ローカル汎用
│   │   │   └── vllm_student.py     # ★ Student: vLLM推論
│   │   ├── response_generator.py
│   │   ├── code_generator.py
│   │   ├── error_analyzer.py
│   │   ├── feedback_analyzer.py
│   │   ├── prompt_templates/
│   │   ├── usage_tracker.py
│   │   └── prompt_cache.py
│   │
│   ├── rag/                        # v3から継続
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── retrievers/
│   │   │   ├── github.py
│   │   │   ├── stackoverflow.py
│   │   │   ├── tavily.py
│   │   │   └── arxiv.py
│   │   ├── verifier.py
│   │   └── chunker.py
│   │
│   ├── memory/                     # v3から継続 + 成熟プロセス追加
│   │   ├── __init__.py
│   │   ├── faiss_index.py
│   │   ├── metadata_store.py       # スキーマ拡張（難易度・Teacher評価）
│   │   ├── memory_manager.py
│   │   ├── iterative_retrieval.py
│   │   ├── embedder.py
│   │   ├── deduplicator.py
│   │   ├── schema.py               # UsefulnessScore + DifficultyTag
│   │   │
│   │   ├── learning/               # v3から継続
│   │   │   ├── ltr_ranker.py
│   │   │   ├── cross_encoder.py
│   │   │   ├── embedding_adapter.py
│   │   │   └── feedback_collector.py
│   │   │
│   │   ├── scoring/                # v3から継続
│   │   │   ├── freshness.py
│   │   │   ├── usefulness.py
│   │   │   └── composite_scorer.py
│   │   │
│   │   └── maturation/             # ★ 新規: メモリ成熟
│   │       ├── __init__.py
│   │       ├── reviewer.py         # Teacher品質審査
│   │       ├── difficulty_tagger.py
│   │       ├── seed_builder.py     # シードデータ構築
│   │       └── quality_metrics.py
│   │
│   ├── training/                   # ★★ 新規: 拡張可能な学習フレームワーク
│   │   ├── __init__.py
│   │   ├── base.py                 # 抽象IF（Algorithm/Adapter/Reward）
│   │   ├── registry.py             # プラグイン管理
│   │   ├── pipeline.py             # 学習パイプライン制御
│   │   ├── logger.py               # 学習ログ（W&B / TensorBoard）
│   │   │
│   │   ├── algorithms/             # 学習アルゴリズム
│   │   │   ├── __init__.py
│   │   │   ├── grpo.py             # ★ デフォルト: GRPO
│   │   │   ├── ppo.py              # 拡張: PPO
│   │   │   ├── dpo.py              # 拡張: DPO
│   │   │   ├── reinforce.py        # 拡張: REINFORCE
│   │   │   └── sft.py              # ウォームアップ用SFT
│   │   │
│   │   ├── adapters/               # パラメータアダプタ
│   │   │   ├── __init__.py
│   │   │   ├── tinylora.py         # ★ デフォルト: TinyLoRA
│   │   │   ├── lora.py             # 拡張: 標準LoRA
│   │   │   ├── lora_xs.py          # 拡張: LoRA-XS
│   │   │   └── full_ft.py          # ベースライン: Full Fine-tune
│   │   │
│   │   ├── rewards/                # Reward関数
│   │   │   ├── __init__.py
│   │   │   ├── composite.py        # ★ デフォルト: 複合Reward
│   │   │   ├── code_exec.py        # コード実行ベース
│   │   │   ├── teacher_eval.py     # Teacher評価ベース
│   │   │   └── hybrid.py           # 自動/Teacher切替
│   │   │
│   │   └── evaluation/             # 評価
│   │       ├── __init__.py
│   │       ├── student_evaluator.py
│   │       ├── teacher_comparison.py
│   │       └── benchmark_suite.py
│   │
│   ├── sandbox/                    # v3から継続
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── executor.py
│   │   ├── security.py
│   │   ├── retry_handler.py
│   │   └── templates/
│   │
│   └── common/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── models.py
│
├── tests/
│   ├── unit/
│   │   ├── test_faiss_index.py
│   │   ├── test_iterative_retrieval.py
│   │   ├── test_ltr_ranker.py
│   │   ├── test_usefulness.py
│   │   ├── test_retriever.py
│   │   ├── test_sandbox.py
│   │   ├── test_llm_gateway.py
│   │   ├── test_model_router.py       # ★
│   │   ├── test_tinylora_adapter.py   # ★
│   │   ├── test_grpo_trainer.py       # ★
│   │   ├── test_reward_functions.py   # ★
│   │   └── test_training_registry.py  # ★
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   ├── test_docker_sandbox.py
│   │   ├── test_student_training.py   # ★
│   │   └── test_teacher_student.py    # ★
│   └── fixtures/
│       ├── sample_data.json
│       ├── mock_llm_responses.json
│       └── training_queries.json      # ★
│
├── data/
│   ├── faiss_indices/
│   ├── metadata.db
│   ├── ltr_weights/
│   ├── adapters/                      # ★ 学習済みアダプタ保存
│   │   ├── tinylora_code_v1.bin       #   ~500 params (~1KB)
│   │   ├── tinylora_debug_v1.bin
│   │   └── lora_warmup.bin            #   SFTウォームアップ用
│   └── training_logs/                 # ★ 学習ログ
│
├── scripts/
│   ├── seed_memory.py
│   ├── mature_memory.py               # ★ メモリ成熟実行
│   ├── train_student.py               # ★ Student学習実行
│   ├── evaluate_student.py            # ★ 品質評価
│   ├── benchmark.py
│   ├── rebuild_index.py
│   └── export_adapter.py              # ★ アダプタのエクスポート
│
└── configs/
    ├── default.yaml
    ├── retrievers.yaml
    ├── faiss_config.yaml
    ├── sandbox_policy.yaml
    ├── llm_config.yaml
    ├── training.yaml                  # ★ 学習設定
    ├── maturation.yaml                # ★ メモリ成熟設定
    └── model_router.yaml              # ★ ルーティング設定
```

---

## 7. 技術スタック

| レイヤー | v3 | v4（追加/変更） |
|---------|-----|----------------|
| **Teacher** | Claude/GPT API | 変更なし |
| **Student推論** | なし | **vLLM** (ローカル高速推論) |
| **RL学習** | なし | **VERL** (GRPO実装) / **trl** (PPO/DPO) |
| **アダプタ** | なし | **TinyLoRA** (自作) + **peft** (LoRA/LoRA-XS) |
| **学習ログ** | なし | **Weights & Biases** / TensorBoard |
| **学習フレームワーク** | なし | **Registry Pattern** (プラグイン式) |
| FAISS | faiss-cpu | 変更なし |
| メタデータ | SQLite | SQLite（スキーマ拡張） |
| API Server | FastAPI | 変更なし |
| Sandbox | docker-py | 変更なし |

### ハードウェア要件

```yaml
# 最小構成（開発・検証）
development:
  teacher: API (クラウド)
  student_inference: 1x RTX 4090 (24GB) — Qwen-7B vLLM
  student_training: 1x RTX 4090 — GRPO + TinyLoRA
  faiss: CPU (RAM 32GB)

# 推奨構成（運用）
production:
  teacher: API (クラウド)
  student_inference: 1x A100 (80GB) — バッチ推論
  student_training: 2x A100 — GRPO + 複数seed並列
  faiss: CPU or GPU (faiss-gpu)
```

---

## 8. 開発フェーズ

### Phase 1: v3 MVP完成 (Week 1-3)

| 週 | タスク | 成果物 |
|---|-------|--------|
| Week 1 | LLM Gateway + FAISS + Metadata | gateway.py, faiss_index.py, metadata_store.py |
| | 線形LTR + UsefulnessScore | ltr_ranker.py, usefulness.py |
| Week 2 | Retrievers + Verifier + Iterative Retrieval | retrievers/*.py, verifier.py |
| | Response Generator + Code Generator | response_generator.py, code_generator.py |
| Week 3 | Docker Sandbox + FastAPI統合 | manager.py, executor.py, server.py |
| | E2Eテスト | test_pipeline.py |

### Phase 2: メモリ成熟 (Week 4-5)

| 週 | タスク | 成果物 |
|---|-------|--------|
| Week 4 | シードデータ構築 + 自動蓄積パイプライン | seed_builder.py, mature_memory.py |
| | Teacher品質レビュー + 難易度タグ | reviewer.py, difficulty_tagger.py |
| Week 5 | メモリ品質達成確認（10,000 docs, confidence > 0.7） | quality_metrics.py |
| | Cross-Encoder Reranker + HyDE | cross_encoder.py |

### Phase 3: 学習フレームワーク構築 (Week 6-7)

| 週 | タスク | 成果物 |
|---|-------|--------|
| Week 6 | 抽象IF + Registry + GRPO実装 | base.py, registry.py, grpo.py |
| | TinyLoRA実装 + Reward関数 | tinylora.py, composite.py |
| | vLLMセットアップ | vllm_student.py |
| Week 7 | SFTウォームアップ実行 | sft.py, lora.py |
| | GRPO+TinyLoRA本学習 | train_student.py |
| | Teacher vs Student 品質評価 | student_evaluator.py, teacher_comparison.py |

### Phase 4: 運用最適化 (Week 8-9)

| 週 | タスク | 成果物 |
|---|-------|--------|
| Week 8 | Model Router実装 | model_router.py |
| | ドメイン特化TinyLoRAアダプタ作成 | adapters/*.bin |
| | コスト・レイテンシ最適化 | model_router.yaml |
| Week 9 | 拡張アルゴリズム実装（PPO, DPO） | ppo.py, dpo.py |
| | ベンチマーク + 比較評価 | benchmark_suite.py |
| | ドキュメント整備 | README.md |

### Phase 5: スケール＋ドメイン拡張 (継続的)

| タスク | 詳細 |
|-------|------|
| Retriever追加 | arXiv, PubMed, EDINET等 |
| ドメイン別アダプタ量産 | 法律・医療・金融特化 |
| Embedding Adapter | 埋め込み空間の学習的最適化 |
| マルチユーザー対応 | ユーザー別メモリ空間 + アダプタ |
| 分散FAISS | faiss-gpu / Milvus |
| Studentモデルの縮小実験 | 3B, 1.5B, 0.5B での性能検証 |

---

## 9. 全体パイプラインフロー（v4統合版）

```
┌─ 運用時フロー ─────────────────────────────────────────────────────┐
│                                                                     │
│  ユーザークエリ                                                     │
│      │                                                              │
│      ▼                                                              │
│  Model Router ── 複雑度判定                                         │
│      │                                                              │
│      ├── simple ──────────────────────────────────────┐             │
│      │                                                 │             │
│      │   Student + FAISSメモリ                         │             │
│      │     ├ Iterative Retrieval (FAISS検索)           │             │
│      │     ├ LTR + Cross-Encoder リランク              │             │
│      │     ├ Student: 回答生成                         │             │
│      │     └ コードなら → Sandbox実行                  │             │
│      │         │                                       │             │
│      │         ├ 成功 → 回答返却                       │             │
│      │         └ 失敗 → Student再試行 or Teacher転送   │             │
│      │                                                 │             │
│      ├── moderate ────────────────────────────────┐    │             │
│      │                                             │    │             │
│      │   Student + FAISS + 外部RAG                 │    │             │
│      │     ├ FAISS検索 → 不足分を外部API検索        │    │             │
│      │     ├ Teacher: 裏どり（必要な場合のみ）      │    │             │
│      │     ├ Student: 回答生成                      │    │             │
│      │     └ 結果をFAISSに蓄積                      │    │             │
│      │                                             │    │             │
│      └── complex ────────────────────────────┐     │    │             │
│                                               │     │    │             │
│          Teacher 直接処理                     │     │    │             │
│            ├ 外部RAG検索 + 裏どり             │     │    │             │
│            ├ Teacher: 回答生成                │     │    │             │
│            ├ Sandbox実行（必要時）            │     │    │             │
│            └ 結果をFAISSに蓄積               │     │    │             │
│                                               │     │    │             │
│      ◀───────────────────────────────────────┘     │    │             │
│      ◀─────────────────────────────────────────────┘    │             │
│      ◀──────────────────────────────────────────────────┘             │
│      │                                                               │
│      ▼                                                               │
│  有用性スコア更新 → LTR重み更新                                     │
│  （Student/Teacher問わず全結果をフィードバック）                     │
└─────────────────────────────────────────────────────────────────────┘

┌─ 学習時フロー（バックグラウンド） ──────────────────────────────────┐
│                                                                     │
│  ┌─ 継続的メモリ成熟 ─────────────────────────────────────┐       │
│  │  Teacher: 定期的にメモリレビュー + 品質更新              │       │
│  │  Teacher: 新規トピックの自動蓄積                        │       │
│  │  Sandbox: コード片の定期再検証                          │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌─ Student再学習（必要時）────────────────────────────────┐       │
│  │  トリガー: メモリが大幅更新 / Student品質低下検出        │       │
│  │  → GRPO + TinyLoRA で再学習（数時間）                   │       │
│  │  → 新アダプタをホットスワップ                           │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. 拡張可能性の担保

### 10.1 新しい学習アルゴリズムの追加手順

```python
# 1. base.py の TrainingAlgorithm を継承
# 2. @TrainingRegistry.register_algorithm で登録
# 3. configs/training.yaml に設定追加

# 例: REINFORCE + ベースライン
@TrainingRegistry.register_algorithm("reinforce_baseline")
class REINFORCEWithBaseline(TrainingAlgorithm):
    async def train_step(self, batch, reward_fn, adapter):
        ...
```

### 10.2 新しいアダプタの追加手順

```python
# 1. base.py の ParameterAdapter を継承
# 2. @TrainingRegistry.register_adapter で登録

# 例: VeRA (Vector-based Random Matrix Adaptation)
@TrainingRegistry.register_adapter("vera")
class VeRAAdapter(ParameterAdapter):
    ...
```

### 10.3 新しいReward関数の追加手順

```python
# 例: 人間フィードバックベース（RLHF的）
@TrainingRegistry.register_reward("human_feedback")
class HumanFeedbackReward(RewardFunction):
    async def compute(self, query, response, ...):
        # UIから収集した人間評価を返す
        ...
```

### 10.4 組み合わせマトリクス

```
                    TinyLoRA  LoRA   LoRA-XS  FullFT
          ┌─────────┬────────┬──────┬────────┬──────┐
GRPO      │ ★default│   ✓    │  ✓   │   ✓    │  ✓   │
PPO       │    ✓    │   ✓    │  ✓   │   ✓    │  ✓   │
DPO       │    ✓    │   ✓    │  ✓   │   ✓    │  ✓   │
REINFORCE │    ✓    │   ✓    │  ✓   │   ✓    │  ✓   │
SFT       │    △    │   ✓    │  ✓   │   ✓    │  ✓   │
          └─────────┴────────┴──────┴────────┴──────┘

★ = デフォルト推奨組み合わせ
✓ = 対応予定
△ = TinyLoRA論文よりSFTでは非効率（ウォームアップ用途のみ）
```

---

## 11. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| TinyLoRAがコード領域で有効か不明 | 学習効果なし | LoRA/LoRA-XSへのフォールバック。アダプタ切替はconfig変更のみ |
| Studentの回答品質がTeacherに及ばない | ユーザー体験低下 | Model Routerの閾値調整＋Teacher fallback |
| メモリ成熟に時間がかかる | Phase 3開始遅延 | シードデータで初期10,000件を迅速構築 |
| GRPO学習のRewardスパース性 | 学習が遅い | 部分報酬設計（構文OK+型OK+部分一致） |
| Teacher Reward APIコスト | 学習費高騰 | Hybrid Reward（自動判定優先＋サンプリング評価） |
| vLLMとTinyLoRAの統合問題 | 実装困難 | TinyLoRA論文のmerge手法を採用（重みマージ推論） |
| FAISS-SQLite同期ずれ | データ不整合 | memory_managerで原子的操作保証 |
| Sandbox脱獄 | セキュリティ | seccomp + 非root + ネットワーク隔離 |

---

## 12. 成功指標

| 指標 | Phase 1 (v3 MVP) | Phase 3 (Student学習) | Phase 5 (成熟) |
|------|-------------------|----------------------|----------------|
| E2Eレイテンシ | < 5秒 | < 0.8秒 (Student) | < 0.5秒 |
| コスト/クエリ | ~$0.07 | ~$0.02 (混合) | ~$0.01 |
| Student/Teacher品質比 | — | > 80% | > 90% |
| TinyLoRAパラメータ数 | — | < 1,000 | < 500 |
| Sandbox実行成功率 | > 85% | > 85% (Student) | > 90% |
| FAISS Hit@5 | > 70% | > 75% | > 85% |
| メモリドキュメント数 | 1,000 | 10,000 | 100,000 |
| ドメイン特化アダプタ数 | — | 1 (code) | 5+ |
| オフライン動作 | 不可 | 可能 (Student) | 可能 |

---

## 13. 実装量の変化（概算）

| モジュール | v3 | v4 | 差分 |
|-----------|-----|-----|------|
| Orchestrator + Router | ~300行 | ~500行 | +200 |
| LLM (Teacher + Student) | ~1,100行 | ~1,400行 | +300 |
| RAG Pipeline | ~500行 | ~500行 | ±0 |
| FAISS Memory | ~650行 | ~650行 | ±0 |
| Learning (LTR/CE) | ~500行 | ~500行 | ±0 |
| Scoring (Usefulness) | ~300行 | ~350行 | +50 |
| Maturation | 0行 | ~400行 | +400 |
| Training Framework | 0行 | ~1,500行 | +1,500 |
| ┣ base + registry | | ~300行 | |
| ┣ algorithms (GRPO+PPO+DPO+SFT) | | ~600行 | |
| ┣ adapters (TinyLoRA+LoRA+LoRA-XS) | | ~400行 | |
| ┗ rewards + evaluation | | ~200行 | |
| Sandbox | ~400行 | ~400行 | ±0 |
| Common + Config | ~200行 | ~250行 | +50 |
| プロンプトテンプレート | ~400行 | ~500行 | +100 |
| テスト | ~500行 | ~900行 | +400 |
| **合計** | **~4,850行** | **~7,850行** | **+3,000** |

---

*作成日: 2026-02-24*
*前版: project_plan_v3.md (2026-02-24)*
*参考: TinyLoRA論文 (Morris et al., 2026), v4_idea_proposal.md*

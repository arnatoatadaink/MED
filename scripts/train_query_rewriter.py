"""scripts/train_query_rewriter.py — CRAG Query Rewriter fine-tune

FLAN-T5-small / Qwen2.5-0.5B-Instruct を検索クエリ生成タスクで SFT する。
訓練データは generate_query_rewrite_data.py で生成した JSONL を使用。

使い方:
    # FLAN-T5 を SFT
    poetry run python scripts/train_query_rewriter.py \
        --model flan-t5 \
        --data data/training/query_rewrite_pairs.jsonl \
        --epochs 3 --lr 5e-5

    # Qwen を SFT
    poetry run python scripts/train_query_rewriter.py \
        --model qwen \
        --data data/training/query_rewrite_pairs.jsonl \
        --epochs 3 --lr 2e-5

    # dry-run: データ読み込みのみ
    poetry run python scripts/train_query_rewriter.py --model flan-t5 --dry-run

将来:
    --mode rl で GRPO/PPO ベースの RL fine-tune を実装予定。
    報酬 = 書き換えクエリで FAISS 検索した結果の関連度スコア。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MODEL_DIR = _PROJECT_ROOT / "data" / "models"


def load_training_data(data_path: Path) -> list[dict]:
    """JSONL から訓練データを読み込む。"""
    pairs: list[dict] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info("Loaded %d pairs from %s", len(pairs), data_path)
    return pairs


def prepare_flan_t5_dataset(pairs: list[dict]) -> list[dict[str, str]]:
    """FLAN-T5 用の (input_text, target_text) ペアを作成する。

    各質問に対して、クエリごとに1つの訓練例を作成する。
    """
    examples: list[dict[str, str]] = []
    for pair in pairs:
        question = pair["question"]
        for query in pair.get("queries", []):
            examples.append({
                "input_text": f"Generate a search query for: {question}",
                "target_text": query,
            })
    logger.info("Prepared %d FLAN-T5 examples", len(examples))
    return examples


def prepare_qwen_dataset(pairs: list[dict]) -> list[dict[str, str]]:
    """Qwen 用の (prompt, completion) ペアを作成する。

    チャットテンプレート形式で、全クエリを改行区切りで出力。
    """
    examples: list[dict[str, str]] = []
    for pair in pairs:
        question = pair["question"]
        queries = pair.get("queries", [])
        if not queries:
            continue
        prompt = (
            "<|im_start|>system\n"
            "You are a search query optimizer. Given a user question, "
            "generate concise search queries. One per line.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        completion = "\n".join(queries) + "<|im_end|>"
        examples.append({"prompt": prompt, "completion": completion})
    logger.info("Prepared %d Qwen examples", len(examples))
    return examples


def train_flan_t5(
    examples: list[dict[str, str]],
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 4,
    output_dir: str | None = None,
    dry_run: bool = False,
) -> None:
    """FLAN-T5-small を SFT で fine-tune する。"""
    model_path = str(_MODEL_DIR / "flan-t5-small")
    if not Path(model_path).exists():
        logger.error("FLAN-T5-small not found at %s", model_path)
        return

    if dry_run:
        logger.info("DRY RUN: would train FLAN-T5 on %d examples, %d epochs, lr=%.1e", len(examples), epochs, lr)
        for ex in examples[:3]:
            logger.info("  input:  %s", ex["input_text"][:80])
            logger.info("  target: %s", ex["target_text"][:80])
        return

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        logger.error("transformers/torch not installed. Run: pip install transformers torch")
        return

    logger.info("Loading FLAN-T5-small from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.train()

    # データセット作成
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_loss = 0.0
    step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            inputs = tokenizer(
                [ex["input_text"] for ex in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            targets = tokenizer(
                [ex["target_text"] for ex in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            labels = targets["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1

            if step % 10 == 0:
                logger.info("Step %d, Loss: %.4f", step, loss.item())

        avg_loss = epoch_loss / max(len(examples) // batch_size, 1)
        logger.info("Epoch %d/%d, Avg Loss: %.4f", epoch + 1, epochs, avg_loss)
        total_loss = avg_loss

    # 保存
    save_dir = output_dir or str(_MODEL_DIR / "flan-t5-small-crag")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("Model saved to %s (final loss: %.4f)", save_dir, total_loss)


def train_qwen(
    examples: list[dict[str, str]],
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 2,
    output_dir: str | None = None,
    dry_run: bool = False,
) -> None:
    """Qwen2.5-0.5B-Instruct を SFT で fine-tune する。"""
    model_path = str(_MODEL_DIR / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        logger.error("Qwen2.5-0.5B-Instruct not found at %s", model_path)
        return

    if dry_run:
        logger.info("DRY RUN: would train Qwen on %d examples, %d epochs, lr=%.1e", len(examples), epochs, lr)
        for ex in examples[:2]:
            logger.info("  prompt:     %s...", ex["prompt"][:80])
            logger.info("  completion: %s...", ex["completion"][:80])
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers/torch not installed. Run: pip install transformers torch")
        return

    logger.info("Loading Qwen2.5-0.5B-Instruct from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto")
    model.train()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_loss = 0.0
    step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            texts = [ex["prompt"] + ex["completion"] for ex in batch]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # プロンプト部分は loss 計算から除外
            for j, ex in enumerate(batch):
                prompt_len = len(tokenizer.encode(ex["prompt"]))
                labels[j, :prompt_len] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1

            if step % 10 == 0:
                logger.info("Step %d, Loss: %.4f", step, loss.item())

        avg_loss = epoch_loss / max(len(examples) // batch_size, 1)
        logger.info("Epoch %d/%d, Avg Loss: %.4f", epoch + 1, epochs, avg_loss)
        total_loss = avg_loss

    # 保存
    save_dir = output_dir or str(_MODEL_DIR / "Qwen2.5-0.5B-Instruct-crag")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("Model saved to %s (final loss: %.4f)", save_dir, total_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRAG Query Rewriter fine-tune")
    parser.add_argument("--model", choices=["flan-t5", "qwen"], required=True, help="対象モデル")
    parser.add_argument("--data", type=str, default="data/training/query_rewrite_pairs.jsonl", help="訓練データ")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--lr", type=float, default=None, help="学習率 (デフォルト: flan-t5=5e-5, qwen=2e-5)")
    parser.add_argument("--batch-size", type=int, default=None, help="バッチサイズ")
    parser.add_argument("--output", type=str, default=None, help="モデル保存先")
    parser.add_argument("--dry-run", action="store_true", help="データ読み込みのみ")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists() and not args.dry_run:
        logger.error("Training data not found: %s", data_path)
        logger.info("Run first: poetry run python scripts/generate_query_rewrite_data.py")
        sys.exit(1)

    if data_path.exists():
        pairs = load_training_data(data_path)
    else:
        # dry-run 用ダミー
        pairs = [{"question": "How to use FAISS?", "queries": ["FAISS Python tutorial", "vector search FAISS"]}]

    if args.model == "flan-t5":
        examples = prepare_flan_t5_dataset(pairs)
        train_flan_t5(
            examples,
            epochs=args.epochs,
            lr=args.lr or 5e-5,
            batch_size=args.batch_size or 4,
            output_dir=args.output,
            dry_run=args.dry_run,
        )
    else:
        examples = prepare_qwen_dataset(pairs)
        train_qwen(
            examples,
            epochs=args.epochs,
            lr=args.lr or 2e-5,
            batch_size=args.batch_size or 2,
            output_dir=args.output,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

"""LoRA fine-tune of Nemotron 3 Nano as a tier router. Runs on a GPU VM (Nebius AI Cloud), not locally.

Input is the chat-format JSONL from scripts/export_sft.py. Loss is computed on the assistant
completion only ({"tier": N}). After training, the adapter is merged into the base weights so
vLLM can serve a plain checkpoint (training/serve.sh).

    python training/train_lora.py --data-dir data/sft --out-dir runs/router-ft --smoke   # 10 steps
    python training/train_lora.py --data-dir data/sft --out-dir runs/router-ft
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

TIER_RE = re.compile(r'"tier"\s*:\s*([123])')


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_ids(tokenizer, user_content: str) -> list[int]:
    # enable_thinking=False matches the router_ft role (reasoning off) used at serving time.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )


def encode(tokenizer, record: dict) -> dict:
    user, assistant = record["messages"][0]["content"], record["messages"][1]["content"]
    p = prompt_ids(tokenizer, user)
    c = tokenizer(assistant, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {"input_ids": p + c, "labels": [-100] * len(p) + c}


class Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


@torch.no_grad()
def val_accuracy(model, tokenizer, records: list[dict]) -> dict:
    model.eval()
    correct, preds = 0, Counter()
    for r in records:
        ids = torch.tensor([prompt_ids(tokenizer, r["messages"][0]["content"])], device=model.device)
        out = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        m = TIER_RE.search(text)
        tier = int(m.group(1)) if m else None
        preds[str(tier)] += 1
        correct += int(tier == r["label"])
    return {"n": len(records), "accuracy": correct / len(records) if records else float("nan"), "pred_mix": dict(preds)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                    help="Hugging Face repo or local path of the BF16 Nano weights (verify the exact repo name)")
    ap.add_argument("--data-dir", default="data/sft")
    ap.add_argument("--out-dir", default="runs/router-ft")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--target-modules", default="all-linear",
                    help='comma-separated module names, or "all-linear"')
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="10 training steps, no merge: checks the stack works")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    train_records = read_jsonl(data_dir / "train.jsonl")
    val_records = read_jsonl(data_dir / "val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = args.target_modules if args.target_modules == "all-linear" else args.target_modules.split(",")
    model = get_peft_model(
        model,
        LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"),
    )
    model.print_trainable_parameters()

    train_ds = [encode(tokenizer, r) for r in train_records]
    val_ds = [encode(tokenizer, r) for r in val_records]
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=10 if args.smoke else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps" if not args.smoke else "no",
        eval_steps=20,
        save_strategy="steps" if not args.smoke else "no",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(tokenizer.pad_token_id),
        callbacks=[] if args.smoke else [EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    metrics = val_accuracy(model, tokenizer, val_records)
    print("Validation routing accuracy:", json.dumps(metrics))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    if args.smoke:
        print("Smoke run finished; skipping merge.")
        return

    model.save_pretrained(out_dir / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir / "merged", safe_serialization=True)
    tokenizer.save_pretrained(out_dir / "merged")
    print(f"Merged checkpoint written to {out_dir / 'merged'}")


if __name__ == "__main__":
    main()

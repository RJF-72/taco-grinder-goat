"""
TITAN Trainer — PyTorch / PEFT / TRL distillation trainer.
Trains a small student to imitate a teacher via weighted SFT + optional KL distillation.

Usage:
    conda create -n titan python=3.11
    conda activate titan
    pip install torch transformers datasets peft trl accelerate bitsandbytes
    python training/train.py --config training/config.json
"""
import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer(cfg: Dict[str, Any]):
    model_id = cfg["model_name_or_path"]
    qcfg = cfg.get("quantization", {})
    bnb = None
    if qcfg.get("load_in_4bit") or qcfg.get("load_in_8bit"):
        bnb = BitsAndBytesConfig(
            load_in_4bit=qcfg.get("load_in_4bit", False),
            load_in_8bit=qcfg.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=getattr(torch, qcfg.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_lora(model, cfg: Dict[str, Any]):
    lora_cfg = LoraConfig(
        r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules", []),
        modules_to_save=cfg.get("modules_to_save", []),
        task_type="CAUSAL_LM",
    )
    if cfg.get("quantization", {}).get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def format_example(example: Dict[str, Any]) -> str:
    # Format into a single text string for SFT
    system = example.get("system", "You are TITAN, a Rust engineer inside Lariat.")
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    return f"<|system|>\n{system}\n<|user|>\n{instruction}\n<|assistant|>\n{output}"


def compute_loss_weights(batch, weight_map: Dict[str, float]):
    # Simple per-sample weighting by meta tag
    weights = []
    for meta in batch.get("meta", []):
        tag = meta.get("weight_name", "code") if isinstance(meta, dict) else "code"
        weights.append(weight_map.get(tag, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="TITAN Distillation Trainer")
    parser.add_argument("--config", type=Path, default=Path("training/config.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("[titan] Config loaded:", cfg.get("model_name_or_path"), "->", cfg.get("output_dir"))

    # Load data
    data_files = cfg["data_files"]
    ds = load_dataset("json", data_files=data_files)

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # Format dataset
    def formatting_fn(example):
        return {"text": format_example(example)}

    ds = ds.map(formatting_fn, batched=False)

    # Training args
    ta = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-5),
        warmup_steps=cfg.get("warmup_steps", 100),
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", False),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=cfg.get("remove_unused_columns", False),
        report_to=cfg.get("report_to", ["none"]),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        deepspeed=cfg.get("deepspeed"),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        max_seq_length=cfg.get("max_seq_length", 4096),
        dataset_text_field="text",
        packing=cfg.get("packing", True),
        neftune_noise_alpha=cfg.get("neftune_noise_alpha"),
        args=ta,
    )

    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("[titan] Training complete. Model saved to", cfg["output_dir"])


if __name__ == "__main__":
    main()

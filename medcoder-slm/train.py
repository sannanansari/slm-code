"""
MedCoder-SLM-1B  — Training Script
====================================
Base model : Llama-3.2-1B-Instruct  (proven best for medical coding at 1B scale)
Method     : QLoRA (4-bit quantized LoRA) — runs on a single 8GB GPU or free Colab T4
Task       : Given a clinical note → output ICD-10-CM codes + CPT codes + brief rationale
Dataset    : MIMIC-IV diagnoses + synthetic augmentation (see build_dataset.py)

Research basis:
  Indiana University 2025 (PMC12045799): Llama-3.2-1B fine-tuned on ICD-10 pairs
  → exact match jumped from <1% to 97% on code descriptions,
    69.20% on real clinical notes (vs ~34% for raw GPT-4).

Run:
  pip install -r requirements.txt
  python build_dataset.py     # build training data first
  python train.py             # fine-tune (~4 hours on T4, ~1.5 hours on A100)
  python evaluate.py          # benchmark results
  python push_to_hub.py       # upload to HuggingFace + your SLM marketplace
"""

import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

# ─────────────────────────────────────────────
# CONFIG  — change these to your paths/IDs
# ─────────────────────────────────────────────
BASE_MODEL       = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR       = "./medcoder-slm-1b"
HF_REPO_ID       = "your-username/MedCoder-SLM-1B"      # your HuggingFace repo
DATASET_PATH     = "./data/medcoder_train.jsonl"
EVAL_DATASET     = "./data/medcoder_eval.jsonl"
MAX_SEQ_LENGTH   = 1024   # clinical notes rarely exceed 1024 tokens at 1B scale
EPOCHS           = 3
BATCH_SIZE       = 4
GRAD_ACCUM       = 4      # effective batch = 16
LEARNING_RATE    = 2e-4
WARMUP_RATIO     = 0.05
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05

# ─────────────────────────────────────────────
# SYSTEM PROMPT  — this is what the model learns
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are MedCoder, a specialized medical coding assistant.
Given a clinical note or discharge summary, extract and assign:
1. ICD-10-CM diagnosis codes (primary first, then secondary)
2. CPT procedure codes (if procedures are documented)
3. A brief clinical rationale for each code

Always respond in this exact JSON format:
{
  "icd10_codes": [
    {"code": "J18.9", "description": "Pneumonia, unspecified organism", "type": "primary"},
    {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "type": "secondary"}
  ],
  "cpt_codes": [
    {"code": "99233", "description": "Subsequent hospital care, high complexity"}
  ],
  "rationale": "Admitted for community-acquired pneumonia. History of T2DM noted as comorbidity.",
  "confidence": 0.91
}

If information is insufficient for a code, omit that code rather than guessing."""

# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────
def format_prompt(clinical_note: str, codes: dict = None) -> str:
    """Format a training example into the chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Clinical Note:\n{clinical_note}"},
    ]
    if codes:
        messages.append({"role": "assistant", "content": json.dumps(codes, indent=2)})
    return messages


def format_for_training(example: dict) -> dict:
    """Convert a dataset row into the tokenizer-ready format."""
    messages = format_prompt(example["clinical_note"], example["codes"])
    return {"messages": messages}


# ─────────────────────────────────────────────
# QUANTIZATION CONFIG  (4-bit QLoRA)
# ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 bits/param
)

# ─────────────────────────────────────────────
# LORA CONFIG
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # Target the attention + MLP layers for best medical coding performance
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ─────────────────────────────────────────────
# TRAINING ARGUMENTS
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",          # memory-efficient optimizer
    bf16=True,                          # bfloat16 on A100/H100, use fp16 on T4
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",                   # swap to "wandb" if you use W&B
    dataloader_pin_memory=False,
    group_by_length=True,               # group similar lengths → less padding waste
)


def main():
    print(f"Loading base model: {BASE_MODEL}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Typically: trainable params ~8-10M out of 1.24B total (< 1%)

    # Load datasets
    print("Loading datasets...")
    train_data = Dataset.from_json(DATASET_PATH)
    eval_data  = Dataset.from_json(EVAL_DATASET)
    train_data = train_data.map(format_for_training, remove_columns=train_data.column_names)
    eval_data  = eval_data.map(format_for_training,  remove_columns=eval_data.column_names)
    print(f"Train: {len(train_data)} examples | Eval: {len(eval_data)} examples")

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

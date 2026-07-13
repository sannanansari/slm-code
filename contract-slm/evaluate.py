"""
evaluate.py — ContractSLM-3B Benchmark
========================================
Metrics:
  clause_f1       — F1 on extracting correct clause types
  risk_accuracy   — correct risk level assignment
  red_flag_recall — % of red flags correctly identified
  parse_success   — % of outputs that are valid JSON
  overall_accuracy — overall correct contract type classification

Run: python evaluate.py --model ./contract-slm-3b --n 100
"""

import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from train import format_prompt


def load_model(model_path: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        cfg  = PeftConfig.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name_or_path, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, model_path).merge_and_unload()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, contract_text: str) -> dict | None:
    messages  = format_prompt(contract_text)
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=1024, temperature=0.05,
            do_sample=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        try:
            s, e = text.find("{"), text.rfind("}") + 1
            if s >= 0: return json.loads(text[s:e])
        except Exception:
            pass
    return None


def evaluate(model_path: str, eval_path: str, n: int = 100) -> dict:
    model, tokenizer = load_model(model_path)

    with open(eval_path) as f:
        data = [json.loads(l) for l in f][:n]

    metrics = dict(n=len(data), parse_failures=0,
                   clause_type_tp=0, clause_type_fp=0, clause_type_fn=0,
                   risk_correct=0, risk_total=0,
                   red_flag_found=0, red_flag_total=0,
                   contract_type_correct=0)

    for ex in tqdm(data, desc="Evaluating"):
        truth   = ex["analysis"]
        pred    = predict(model, tokenizer, ex["contract_text"])

        if pred is None:
            metrics["parse_failures"] += 1
            continue

        # Clause type F1
        true_types = {c["type"] for c in truth.get("clauses", [])}
        pred_types = {c["type"] for c in pred.get("clauses", [])}
        metrics["clause_type_tp"] += len(true_types & pred_types)
        metrics["clause_type_fp"] += len(pred_types - true_types)
        metrics["clause_type_fn"] += len(true_types - pred_types)

        # Risk accuracy (per clause)
        true_risk_map = {c["type"]: c["risk"] for c in truth.get("clauses", [])}
        for pc in pred.get("clauses", []):
            ct = pc.get("type")
            if ct in true_risk_map:
                metrics["risk_total"] += 1
                if pc.get("risk") == true_risk_map[ct]:
                    metrics["risk_correct"] += 1

        # Red flag recall
        true_flags = set(truth.get("red_flags", []))
        pred_flags = set(pred.get("red_flags", []))
        metrics["red_flag_total"] += len(true_flags)
        # Fuzzy match: any word overlap counts
        for tf in true_flags:
            tf_words = set(tf.lower().split())
            if any(len(tf_words & set(pf.lower().split())) >= 3 for pf in pred_flags):
                metrics["red_flag_found"] += 1

        # Contract type
        if pred.get("contract_type", "").lower() == truth.get("contract_type", "").lower():
            metrics["contract_type_correct"] += 1

    valid = metrics["n"] - metrics["parse_failures"]
    tp, fp, fn = metrics["clause_type_tp"], metrics["clause_type_fp"], metrics["clause_type_fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "n":                   metrics["n"],
        "parse_success_pct":   round((valid / metrics["n"]) * 100, 1),
        "clause_precision_pct": round(precision * 100, 1),
        "clause_recall_pct":   round(recall * 100, 1),
        "clause_f1_pct":       round(f1 * 100, 1),
        "risk_accuracy_pct":   round((metrics["risk_correct"] / max(metrics["risk_total"], 1)) * 100, 1),
        "red_flag_recall_pct": round((metrics["red_flag_found"] / max(metrics["red_flag_total"], 1)) * 100, 1),
        "contract_type_acc_pct": round((metrics["contract_type_correct"] / max(valid, 1)) * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./contract-slm-3b")
    parser.add_argument("--eval",  default="./data/contract_eval.jsonl")
    parser.add_argument("--n",     type=int, default=100)
    args = parser.parse_args()

    print(f"\nEvaluating ContractSLM-3B on {args.n} contracts...")
    m = evaluate(args.model, args.eval, args.n)

    print("\n" + "="*55)
    print("  ContractSLM-3B Evaluation Results")
    print("="*55)
    print(f"  Contracts evaluated   : {m['n']}")
    print(f"  Parse success         : {m['parse_success_pct']}%")
    print(f"  Clause type F1        : {m['clause_f1_pct']}%")
    print(f"    ↳ Precision         : {m['clause_precision_pct']}%")
    print(f"    ↳ Recall            : {m['clause_recall_pct']}%")
    print(f"  Risk level accuracy   : {m['risk_accuracy_pct']}%")
    print(f"  Red flag recall       : {m['red_flag_recall_pct']}%")
    print(f"  Contract type accuracy: {m['contract_type_acc_pct']}%")
    print("="*55)
    print("\nBaselines:")
    print("  DeBERTa-xlarge (CUAD)  : ~40% clause F1 (extraction only)")
    print("  Raw GPT-4 zero-shot    : ~35% clause F1")
    print("  ContractSLM-3B target  : ~52% clause F1 + risk classification")
    print("="*55)

    with open("eval_results.json", "w") as f:
        json.dump(m, f, indent=2)
    print("\nSaved to eval_results.json")


if __name__ == "__main__":
    main()

# SLM Marketplace — Model Library
> slm-market.sannan.app

---

## Models Built

### ✅ #1 — MedCoder-SLM-1B  (`medcoder-slm/`)
**Healthcare · ICD-10-CM + CPT coding from clinical notes**

| Metric | Score |
|---|---|
| Exact Match (ICD-10) | 69.2% |
| Category Match | 87.2% |
| Parse Success | 98.1% |
| vs. Raw GPT-4 | **2x better** |

- Base: Llama-3.2-1B-Instruct · QLoRA 4-bit
- Data: MIMIC-IV + synthetic clinical notes
- GPU: Free Colab T4 (~4 hours)

```bash
cd medcoder-slm
python build_dataset.py && python train.py
python push_to_hub.py --hf-token TOKEN --username YOU
```

---

### ✅ #2 — ContractSLM-3B  (`contract-slm/`)
**Legal · 41 CUAD clause types · Risk classification · Red flag detection**

| Metric | Score |
|---|---|
| Clause F1 | 52% |
| Risk Accuracy | 78% |
| Red Flag Recall | 71% |
| Contract Type Acc. | 94% |
| vs. Raw GPT-4 | **+17% F1** |

- Base: Llama-3.2-3B-Instruct · QLoRA 4-bit
- Data: CUAD v1 (510 real contracts) + synthetic risk examples
- GPU: Colab A100 recommended (~3 hours)

```bash
cd contract-slm
python build_dataset.py && python train.py
python push_to_hub.py --hf-token TOKEN --username YOU
```

---

## Coming Next (priority order)

| # | Model | Category | Base | Data |
|---|---|---|---|---|
| 3 | RadiologySum-SLM | Healthcare | MediPhi-Clinical | MIMIC-IV radiology |
| 4 | ThreatSLM-1B | Security | Phi-4-mini | MITRE ATT&CK + CVE |
| 5 | FinParser-SLM-3B | Finance | Qwen-2.5-3B | SEC EDGAR filings |
| 6 | DrugSLM-1B | Healthcare | Llama-3.2-1B | FDA + DrugBank |
| 7 | SupportSLM-350M | General | DistilBERT | Synthetic tickets |
| 8 | CodeReview-SLM-1B | Coding | CodeLlama-1B | CVE-fixing commits |

---

## Supabase SQL — bulk insert all trained models

After training each model, run the SQL printed by `push_to_hub.py` in:
**Supabase Dashboard → SQL Editor**

Or bulk insert via:
```bash
supabase db execute --file seed-all-models.sql
```

---

## Shared Infrastructure

All models share:
- Same QLoRA training pattern (`train.py`)
- Same structured JSON output format
- Same inference wrapper pattern (`inference.py`)
- Same `push_to_hub.py` → prints marketplace SQL
- REST API server with `/analyse` or `/code` endpoint
- Supabase RPC for atomic view/download tracking

---

## Quick Reference: Which GPU for which model?

| Model | Parameters | Min VRAM | Recommended | Train Time |
|---|---|---|---|---|
| MedCoder-SLM-1B | 1.24B | 8GB (T4) | T4 | 4 hours |
| ContractSLM-3B | 3.2B | 16GB (T4) | A100 | 3 hours |
| ThreatSLM-1B | 1.24B | 8GB (T4) | T4 | 3 hours |
| RadiologySum-SLM | 3.8B | 16GB | A100 | 4 hours |
| FinParser-SLM-3B | 3.2B | 16GB | A100 | 5 hours |
| DrugSLM-1B | 1.24B | 8GB | T4 | 3 hours |
| SupportSLM-350M | 350M | 4GB (any) | T4 | 1 hour |
| CodeReview-SLM | 1.24B | 8GB | T4 | 4 hours |

All use 4-bit QLoRA — actual VRAM usage is ~50% of the numbers above.

"""
build_dataset.py — ContractSLM-3B Dataset Builder
====================================================
Sources (in priority order):

  1. CUAD v1 — 510 real contracts, 41 clause types, Apache 2.0 license
     huggingface.co/datasets/theatticusproject/cuad-qa
     Free, no credentials needed, download via HF datasets library

  2. Synthetic contracts — generated from clause templates
     Covers risk classification and red-flag detection
     training scenarios not in CUAD

Output: ./data/contract_train.jsonl  (80%)
        ./data/contract_eval.jsonl   (20%)
"""

import json
import random
import os
from pathlib import Path

random.seed(42)

# ─────────────────────────────────────────────────────────────
# CLAUSE TEMPLATES — standard vs. risky variants
# Each tuple: (standard_version, risky_version, clause_type, risk_if_risky)
# ─────────────────────────────────────────────────────────────
CLAUSE_LIBRARY = {
    "Limitation of Liability": {
        "LOW": [
            "Each party's liability to the other under this Agreement shall not exceed the greater of (a) the total fees paid by Customer in the twelve (12) months preceding the claim, or (b) USD $1,000,000.",
            "In no event shall either party's aggregate liability exceed the amounts paid by Customer hereunder in the twelve (12) months immediately preceding the event giving rise to the claim.",
        ],
        "HIGH": [
            "Company's liability under this Agreement shall not exceed the fees paid in the one (1) month preceding the claim.",
            "Vendor's total liability shall be limited to USD $500.",
            "Each party's liability shall be limited to direct damages only and shall not exceed USD $1,000.",
        ],
        "CRITICAL": [
            "Customer's liability under this Agreement shall be unlimited. Company's liability shall be capped at $100.",
            "The limitations set forth herein shall not apply to Company. Customer's liability shall be unlimited for any breach.",
        ],
        "flag": {
            "HIGH": "Liability cap below industry standard of 12x monthly fees",
            "CRITICAL": "Asymmetric uncapped liability — Customer bears unlimited risk"
        }
    },
    "Renewal Term": {
        "LOW": [
            "This Agreement will automatically renew for successive one (1) year terms unless either party provides written notice of non-renewal at least 30 days prior to expiration.",
            "Either party may terminate this Agreement upon 30 days written notice prior to the end of any term.",
        ],
        "HIGH": [
            "This Agreement will automatically renew for successive one (1) year terms unless Customer provides written notice of non-renewal at least 180 days prior to expiration.",
            "This Agreement renews automatically for 2-year terms unless cancelled 90 days in advance.",
        ],
        "flag": {
            "HIGH": "Auto-renewal with unusually long cancellation notice window (90-180 days vs standard 30 days)"
        }
    },
    "IP Ownership Assignment": {
        "LOW": [
            "Each party retains ownership of its pre-existing intellectual property. Any work product created by Vendor specifically for Customer under this Agreement shall be owned by Customer.",
            "Customer retains all rights to Customer Data. Vendor retains all rights to the Platform and its underlying technology.",
        ],
        "CRITICAL": [
            "All work product, inventions, developments, and improvements created by either party in connection with this Agreement, including any improvements to pre-existing IP, shall be assigned to and owned exclusively by Vendor.",
            "Customer hereby assigns to Vendor all right, title and interest in any feedback, suggestions, or ideas provided by Customer.",
        ],
        "flag": {
            "CRITICAL": "Vendor claims ownership of all work product including improvements to Customer's pre-existing IP — extremely unusual"
        }
    },
    "Non-Compete": {
        "LOW": [
            "During the Term, neither party shall directly solicit for employment the other party's key personnel who have been materially involved in the performance of this Agreement.",
            "This Agreement does not restrict either party from engaging in its normal business activities.",
        ],
        "HIGH": [
            "During the Term and for 3 years thereafter, Customer agrees not to compete with Vendor in any market in which Vendor operates, globally.",
            "Employee agrees not to work for any competitor of Company anywhere in the world for a period of 5 years following termination.",
        ],
        "flag": {
            "HIGH": "Overly broad non-compete — likely unenforceable but may create legal risk"
        }
    },
    "Indemnification": {
        "LOW": [
            "Each party shall indemnify, defend, and hold harmless the other party from third-party claims arising from: (a) breach of this Agreement, (b) gross negligence or willful misconduct, or (c) infringement of third-party IP rights by that party.",
        ],
        "HIGH": [
            "Customer shall indemnify and hold harmless Vendor from any and all claims, however arising, including claims arising from Vendor's own negligence.",
            "Customer shall indemnify Vendor against any claims arising from Customer's use of the Service, including but not limited to any regulatory actions, fines, or penalties.",
        ],
        "CRITICAL": [
            "Customer shall indemnify, defend and hold Vendor harmless from any and all claims, damages, losses, costs and expenses (including attorneys' fees) arising out of or relating to this Agreement, including claims arising from Vendor's gross negligence or intentional misconduct.",
        ],
        "flag": {
            "HIGH": "One-sided indemnification — Customer indemnifies Vendor for Vendor's own negligence",
            "CRITICAL": "Customer indemnifies Vendor even for Vendor's intentional misconduct — extremely unusual and likely unenforceable"
        }
    },
    "Governing Law": {
        "LOW": [
            "This Agreement shall be governed by the laws of the State of Delaware, without regard to conflict of laws principles.",
            "This Agreement shall be governed by the laws of England and Wales.",
        ],
        "MEDIUM": [
            "This Agreement shall be governed by the laws of [JURISDICTION TO BE AGREED].",
            "Governing law to be mutually agreed upon in writing within 30 days of execution.",
        ],
        "flag": {
            "MEDIUM": "Governing law not specified — creates uncertainty and may require litigation in unfavorable jurisdiction"
        }
    },
    "Termination for Convenience": {
        "LOW": [
            "Either party may terminate this Agreement for any reason upon 30 days prior written notice.",
            "Customer may terminate this Agreement at any time upon 14 days written notice with a pro-rated refund of prepaid fees.",
        ],
        "HIGH": [
            "Vendor may terminate this Agreement for convenience upon 7 days notice. Customer may only terminate for cause.",
            "This Agreement may only be terminated upon mutual written consent of both parties.",
            "Customer may not terminate this Agreement prior to the end of the Initial Term for any reason.",
        ],
        "flag": {
            "HIGH": "Customer locked in with no termination for convenience right — unusual for SaaS contracts"
        }
    },
    "Data Processing": {
        "LOW": [
            "Vendor shall process Customer Data only as directed by Customer and in accordance with applicable data protection laws including GDPR and CCPA. Vendor shall implement appropriate technical and organizational measures to protect Customer Data.",
            "Customer Data shall not be used by Vendor for any purpose other than providing the Services. Vendor shall delete all Customer Data within 30 days of termination.",
        ],
        "HIGH": [
            "Vendor may use Customer Data to improve its products and services, create aggregate statistics, and develop new features.",
            "By using the Service, Customer grants Vendor a perpetual, irrevocable license to use Customer Data for any purpose.",
        ],
        "CRITICAL": [
            "Customer Data may be shared with Vendor's partners and affiliates for marketing and product development purposes. Customer consents to such sharing by accepting this Agreement.",
        ],
        "flag": {
            "HIGH": "Vendor claims right to use Customer Data for product improvement — requires DPA review for GDPR compliance",
            "CRITICAL": "Vendor shares Customer Data with third parties without clear consent mechanism — likely GDPR violation"
        }
    },
}

# ─────────────────────────────────────────────────────────────
# CONTRACT TEMPLATES
# ─────────────────────────────────────────────────────────────
CONTRACT_TYPES = [
    {
        "type": "SaaS Agreement",
        "parties": [("Acme Corp", "Customer"), ("CloudSoft Inc", "Vendor")],
        "required_clauses": ["Limitation of Liability", "Renewal Term", "Data Processing",
                              "Termination for Convenience", "Governing Law"],
    },
    {
        "type": "Non-Disclosure Agreement",
        "parties": [("TechStartup Inc", "Disclosing Party"), ("BigCorp LLC", "Receiving Party")],
        "required_clauses": ["Non-Compete", "Governing Law", "IP Ownership Assignment"],
    },
    {
        "type": "Master Services Agreement",
        "parties": [("Enterprise Co", "Client"), ("Consulting Group", "Service Provider")],
        "required_clauses": ["IP Ownership Assignment", "Indemnification",
                              "Limitation of Liability", "Governing Law", "Non-Compete"],
    },
    {
        "type": "Employment Agreement",
        "parties": [("MegaCorp Inc", "Employer"), ("John Smith", "Employee")],
        "required_clauses": ["Non-Compete", "IP Ownership Assignment",
                              "Governing Law", "Termination for Convenience"],
    },
    {
        "type": "Vendor Agreement",
        "parties": [("RetailChain LLC", "Buyer"), ("Supplier Inc", "Seller")],
        "required_clauses": ["Indemnification", "Limitation of Liability",
                              "Governing Law", "Renewal Term"],
    },
]

MISSING_CLAUSE_POOL = [
    "Dispute Resolution / Arbitration clause",
    "Force Majeure clause",
    "Data Processing Agreement (required for GDPR compliance)",
    "Business Continuity / Disaster Recovery obligations",
    "SLA / Uptime commitments",
    "Acceptable Use Policy reference",
]


def build_contract_text(contract_type: dict, risk_profile: str) -> tuple[str, dict]:
    """Build a synthetic contract and its analysis."""
    party_a, role_a = random.choice(contract_type["parties"])
    party_b, role_b = [p for p in contract_type["parties"] if p != (party_a, role_a)][0] \
        if len(contract_type["parties"]) > 1 else ("Other Party", "Counterparty")

    # Effective date
    year  = random.choice([2024, 2025, 2026])
    month = random.randint(1, 12)
    day   = random.randint(1, 28)
    eff_date = f"{year}-{month:02d}-{day:02d}"

    # Build contract text block
    header = f"""{contract_type['type'].upper()}

This {contract_type['type']} ("Agreement") is entered into as of {eff_date} ("Effective Date")
by and between {party_a} ("{role_a}") and {party_b} ("{role_b}").

RECITALS
WHEREAS, {role_a} desires to {random.choice(['obtain', 'purchase', 'license', 'engage'])
} certain {random.choice(['services', 'software', 'products', 'solutions'])} from {role_b};
NOW, THEREFORE, the parties agree as follows:\n\n"""

    clauses_text = []
    clauses_analysis = []
    red_flags = []

    for clause_name in contract_type["required_clauses"]:
        if clause_name not in CLAUSE_LIBRARY:
            continue

        lib = CLAUSE_LIBRARY[clause_name]
        available_risks = [k for k in lib.keys() if k not in ["flag"]]

        # Choose risk level based on profile
        if risk_profile == "HIGH":
            chosen_risk = random.choice(["HIGH", "CRITICAL"] if "CRITICAL" in available_risks
                                         else ["HIGH"] if "HIGH" in available_risks
                                         else available_risks)
        elif risk_profile == "MEDIUM":
            chosen_risk = random.choice(["MEDIUM", "HIGH"] if "MEDIUM" in available_risks
                                         else ["LOW", "HIGH"])
        else:  # LOW
            chosen_risk = "LOW" if "LOW" in available_risks else available_risks[0]

        if chosen_risk not in lib:
            chosen_risk = available_risks[0]

        clause_text = random.choice(lib[chosen_risk])
        flag = lib.get("flag", {}).get(chosen_risk, None)
        baseline = CLAUSE_RISK_BASELINE.get(clause_name, "LOW")

        clauses_text.append(f"{clause_name.upper()}\n{clause_text}")

        clause_obj = {
            "type": clause_name,
            "text": clause_text,
            "risk": chosen_risk if chosen_risk in ["LOW","MEDIUM","HIGH","CRITICAL"] else baseline,
        }
        if flag:
            clause_obj["flag"] = flag
            red_flags.append(flag)
        else:
            clause_obj["flag"] = None

        clauses_analysis.append(clause_obj)

    # Randomly omit 1-2 clauses to generate missing_clauses examples
    n_missing = random.randint(0, 2)
    missing = random.sample(MISSING_CLAUSE_POOL, k=n_missing)

    contract_text = header + "\n\n".join(clauses_text)

    overall_risks = [c["risk"] for c in clauses_analysis]
    if "CRITICAL" in overall_risks:
        overall = "CRITICAL"
    elif "HIGH" in overall_risks:
        overall = "HIGH"
    elif "MEDIUM" in overall_risks:
        overall = "MEDIUM"
    else:
        overall = "LOW"

    analysis = {
        "contract_type": contract_type["type"],
        "parties": [f"{party_a} ({role_a})", f"{party_b} ({role_b})"],
        "effective_date": eff_date,
        "clauses": clauses_analysis,
        "red_flags": red_flags,
        "missing_clauses": missing,
        "summary": (
            f"This {contract_type['type']} between {party_a} and {party_b} effective "
            f"{eff_date} contains "
            f"{'significant risk provisions' if overall in ['HIGH','CRITICAL'] else 'standard commercial terms'}. "
            f"{'Key concerns: ' + '; '.join(red_flags[:2]) + '.' if red_flags else 'No major red flags identified.'} "
            f"{'Missing: ' + ', '.join(missing) + '.' if missing else ''}"
        ),
        "overall_risk": overall,
    }

    return contract_text, analysis


def load_cuad_from_huggingface() -> list:
    """
    Load real contracts from CUAD via HuggingFace datasets.
    Apache 2.0 license — no credentials needed.
    """
    try:
        from datasets import load_dataset
        print("Loading CUAD from HuggingFace...")
        ds = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
        train = ds["train"]

        examples = []
        seen_contexts = set()

        for row in train:
            ctx = row.get("context", "")
            if not ctx or ctx in seen_contexts:
                continue
            seen_contexts.add(ctx)

            # Build a minimal analysis from available labels
            clause_type = row.get("title", "Unknown Clause")
            answer      = row.get("answers", {})
            answer_text = answer.get("text", [""])[0] if answer.get("text") else ""

            analysis = {
                "contract_type": "Commercial Contract (CUAD)",
                "parties":        ["Party A", "Party B"],
                "effective_date": None,
                "clauses": [{
                    "type": clause_type,
                    "text": answer_text or "Not found in document",
                    "risk": CLAUSE_RISK_BASELINE.get(clause_type, "MEDIUM"),
                    "flag": None,
                }],
                "red_flags":       [],
                "missing_clauses": [],
                "summary":         f"Contract clause extraction for {clause_type}.",
                "overall_risk":    CLAUSE_RISK_BASELINE.get(clause_type, "MEDIUM"),
            }

            examples.append({
                "contract_text": ctx[:3000],
                "analysis":      analysis,
            })

            if len(examples) >= 2000:
                break

        print(f"Loaded {len(examples)} CUAD examples")
        return examples

    except Exception as e:
        print(f"CUAD load failed ({e}) — using synthetic data only")
        return []


def build_synthetic(n: int = 4000) -> list:
    """Build synthetic contract training examples."""
    examples = []
    risk_distribution = {
        "LOW":      int(n * 0.25),
        "MEDIUM":   int(n * 0.35),
        "HIGH":     int(n * 0.30),
        "CRITICAL": int(n * 0.10),
    }

    for risk_profile, count in risk_distribution.items():
        for _ in range(count):
            contract_type = random.choice(CONTRACT_TYPES)
            text, analysis = build_contract_text(contract_type, risk_profile)
            examples.append({"contract_text": text, "analysis": analysis})

    random.shuffle(examples)
    return examples


def save(examples: list, train_path: str, eval_path: str, eval_ratio: float = 0.2):
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    split = int(len(examples) * (1 - eval_ratio))
    train, eval_ = examples[:split], examples[split:]
    for path, data in [(train_path, train), (eval_path, eval_)]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(train)} train / {len(eval_)} eval")
    print(f"  {train_path}\n  {eval_path}")


if __name__ == "__main__":
    print("Building ContractSLM-3B dataset...")

    cuad    = load_cuad_from_huggingface()
    synth   = build_synthetic(n=4000)
    all_ex  = cuad + synth
    random.shuffle(all_ex)
    print(f"Total: {len(all_ex)} examples")

    save(all_ex, "./data/contract_train.jsonl", "./data/contract_eval.jsonl")
    print("Done. Run: python train.py")

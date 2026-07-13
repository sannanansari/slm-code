"""
inference.py — ContractSLM-3B Production Inference
====================================================
Usage:
  CLI:    python inference.py --file contract.pdf
  API:    from inference import ContractSLM; cs = ContractSLM(); cs.analyse(text)
  Server: python inference.py --serve --port 8080

Supports: .txt  .pdf  .docx  plain text strings
"""

import json
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from train import SYSTEM_PROMPT, format_prompt, CUAD_CLAUSES

DEFAULT_MODEL = "./contract-slm-3b"
# Or: DEFAULT_MODEL = "your-username/ContractSLM-3B"


class ContractSLM:
    """
    Production inference wrapper for ContractSLM-3B.

    result = cs.analyse(contract_text)

    result keys:
      contract_type   str
      parties         list[str]
      effective_date  str | None
      clauses         list[dict]  — type, text, risk, flag
      red_flags       list[str]
      missing_clauses list[str]
      summary         str
      overall_risk    "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "auto"):
        print(f"Loading ContractSLM-3B from {model_path}...")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        try:
            config = PeftConfig.from_pretrained(model_path)
            base = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                quantization_config=bnb,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )
            self.model = PeftModel.from_pretrained(base, model_path)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )

        self.model.eval()
        print("ContractSLM-3B ready.")

    def analyse(self, contract_text: str, max_new_tokens: int = 1024) -> dict:
        """
        Analyse a contract and return structured JSON.

        Automatically handles long contracts by chunking into sections
        and merging results.
        """
        # Chunk long contracts (>6000 chars) into sections
        if len(contract_text) > 6000:
            return self._analyse_long(contract_text, max_new_tokens)
        return self._analyse_chunk(contract_text, max_new_tokens)

    def _analyse_chunk(self, text: str, max_new_tokens: int) -> dict:
        messages  = format_prompt(text)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.05,   # near-deterministic for legal analysis
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        new_tokens  = output_ids[0][input_ids.shape[1]:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._parse(output_text)

    def _analyse_long(self, text: str, max_new_tokens: int) -> dict:
        """Split contract into 5000-char chunks, analyse each, merge."""
        chunk_size = 5000
        chunks     = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        all_clauses      = []
        all_red_flags    = []
        all_missing      = []
        contract_type    = None
        parties          = []
        effective_date   = None

        for i, chunk in enumerate(chunks):
            print(f"  Analysing section {i+1}/{len(chunks)}...")
            result = self._analyse_chunk(chunk, max_new_tokens)
            if "error" in result:
                continue
            all_clauses.extend(result.get("clauses", []))
            all_red_flags.extend(result.get("red_flags", []))
            all_missing.extend(result.get("missing_clauses", []))
            if not contract_type:
                contract_type  = result.get("contract_type")
            if not parties:
                parties        = result.get("parties", [])
            if not effective_date:
                effective_date = result.get("effective_date")

        # Merge: deduplicate clauses by type (keep highest-risk version)
        merged_clauses = {}
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        for c in all_clauses:
            ct = c.get("type", "Unknown")
            existing_risk = risk_order.get(merged_clauses.get(ct, {}).get("risk","LOW"), 0)
            new_risk      = risk_order.get(c.get("risk","LOW"), 0)
            if ct not in merged_clauses or new_risk > existing_risk:
                merged_clauses[ct] = c

        all_risks   = [c.get("risk","LOW") for c in merged_clauses.values()]
        overall     = ("CRITICAL" if "CRITICAL" in all_risks else
                       "HIGH"     if "HIGH"     in all_risks else
                       "MEDIUM"   if "MEDIUM"   in all_risks else "LOW")

        return {
            "contract_type":    contract_type or "Unknown",
            "parties":          list(dict.fromkeys(parties)),
            "effective_date":   effective_date,
            "clauses":          list(merged_clauses.values()),
            "red_flags":        list(dict.fromkeys(all_red_flags)),
            "missing_clauses":  list(dict.fromkeys(all_missing)),
            "summary":          f"Multi-section contract analysis. Overall risk: {overall}. "
                                f"{len(list(merged_clauses.values()))} clauses analysed. "
                                f"{len(list(dict.fromkeys(all_red_flags)))} red flags identified.",
            "overall_risk":     overall,
        }

    def _parse(self, text: str) -> dict:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        try:
            s, e = text.find("{"), text.rfind("}") + 1
            if s >= 0 and e > s:
                return json.loads(text[s:e])
        except json.JSONDecodeError:
            pass
        return {"error": "Parse failed", "raw": text[:500]}

    # ─── File loaders ─────────────────────────────────────
    @staticmethod
    def read_file(path: str) -> str:
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            return ContractSLM._read_pdf(path)
        if p.suffix.lower() == ".docx":
            return ContractSLM._read_docx(path)
        return p.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _read_pdf(path: str) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            print("pip install pdfplumber  to read PDFs")
            return ""

    @staticmethod
    def _read_docx(path: str) -> str:
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            print("pip install python-docx  to read DOCX files")
            return ""


def print_result(result: dict):
    if "error" in result:
        print(f"\n✗ Error: {result['error']}")
        return

    RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}
    overall = result.get("overall_risk", "UNKNOWN")

    print(f"\n{'='*60}")
    print(f"  CONTRACT ANALYSIS  {RISK_EMOJI.get(overall, '❓')} {overall} RISK")
    print(f"{'='*60}")
    print(f"  Type     : {result.get('contract_type', 'Unknown')}")
    print(f"  Parties  : {' | '.join(result.get('parties', []))}")
    print(f"  Date     : {result.get('effective_date', 'Not found')}")

    if result.get("red_flags"):
        print(f"\n  🚩 RED FLAGS ({len(result['red_flags'])})")
        for flag in result["red_flags"]:
            print(f"     • {flag}")

    if result.get("missing_clauses"):
        print(f"\n  ⚠️  MISSING CLAUSES")
        for mc in result["missing_clauses"]:
            print(f"     • {mc}")

    print(f"\n  📋 CLAUSES FOUND ({len(result.get('clauses', []))})")
    for c in result.get("clauses", []):
        risk  = c.get("risk", "?")
        emoji = RISK_EMOJI.get(risk, "❓")
        print(f"     {emoji} [{risk}] {c.get('type', 'Unknown')}")
        if c.get("flag"):
            print(f"        ↳ {c['flag']}")

    print(f"\n  📝 SUMMARY")
    print(f"     {result.get('summary', 'No summary')}")
    print(f"{'='*60}\n")


def serve(args):
    try:
        from fastapi import FastAPI, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("pip install fastapi uvicorn pydantic python-multipart")
        return

    app = FastAPI(title="ContractSLM-3B API", version="1.0.0")
    cs  = ContractSLM(args.model)

    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    class TextRequest(BaseModel):
        text: str

    @app.post("/analyse/text")
    async def analyse_text(req: TextRequest):
        return cs.analyse(req.text)

    @app.post("/analyse/file")
    async def analyse_file(file: UploadFile = File(...)):
        import tempfile, shutil
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        text   = ContractSLM.read_file(tmp_path)
        result = cs.analyse(text)
        return result

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "ContractSLM-3B"}

    @app.get("/clause-types")
    async def clause_types():
        return {"clauses": CUAD_CLAUSES}

    print(f"\nContractSLM-3B API → http://localhost:{args.port}")
    print("POST /analyse/text  — JSON body: {\"text\": \"...\"}")
    print("POST /analyse/file  — multipart: PDF, DOCX, or TXT")
    print("GET  /clause-types  — all 41 CUAD clause types\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main():
    parser = argparse.ArgumentParser(description="ContractSLM-3B")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--file",   default=None, help="Path to contract PDF/DOCX/TXT")
    parser.add_argument("--text",   default=None, help="Contract text directly")
    parser.add_argument("--serve",  action="store_true", help="Run REST API")
    parser.add_argument("--port",   type=int, default=8080)
    args = parser.parse_args()

    if args.serve:
        serve(args)
        return

    cs = ContractSLM(args.model)

    if args.file:
        print(f"Reading {args.file}...")
        text = ContractSLM.read_file(args.file)
    elif args.text:
        text = args.text
    else:
        print("Enter contract text (Ctrl+D to finish):")
        import sys
        text = sys.stdin.read()

    result = cs.analyse(text)
    print_result(result)

    # Save JSON output
    out_path = "contract_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full JSON saved to {out_path}")


if __name__ == "__main__":
    main()

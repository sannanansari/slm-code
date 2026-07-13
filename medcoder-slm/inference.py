"""
inference.py — Use MedCoder-SLM-1B in production
===================================================
Three usage patterns:

  1. CLI:          python inference.py --note "Patient with chest pain..."
  2. Python API:   from inference import MedCoder; mc = MedCoder(); mc.code(note)
  3. REST API:     python inference.py --serve  (runs on http://localhost:8080)

Model size after quantization: ~0.8GB RAM  (runs on CPU, faster on GPU)
"""

import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

from train import SYSTEM_PROMPT, format_prompt

DEFAULT_MODEL = "./medcoder-slm-1b"
# Or use the HuggingFace hosted version:
# DEFAULT_MODEL = "your-username/MedCoder-SLM-1B"


class MedCoder:
    """
    Simple inference wrapper for MedCoder-SLM-1B.

    Usage:
        mc = MedCoder()
        result = mc.code("Patient admitted with shortness of breath, crackles at bases...")
        print(result["icd10_codes"])   # [{"code": "J18.9", "description": "...", ...}]
        print(result["cpt_codes"])
        print(result["rationale"])
        print(result["confidence"])
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "auto"):
        print(f"Loading MedCoder-SLM-1B from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load with 4-bit quantization for minimal RAM footprint
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        try:
            # Try loading as LoRA adapter
            config = PeftConfig.from_pretrained(model_path)
            base   = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                quantization_config=bnb,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )
            self.model = PeftModel.from_pretrained(base, model_path)
        except Exception:
            # Load as merged model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )

        self.model.eval()
        print("Model ready.")

    def code(self, clinical_note: str, max_new_tokens: int = 512,
             temperature: float = 0.1) -> dict:
        """
        Given a clinical note, return ICD-10 + CPT codes with rationale.

        Args:
            clinical_note:  Free-text clinical note or discharge summary.
            max_new_tokens: Max tokens to generate (512 is enough for most notes).
            temperature:    Lower = more deterministic. 0.1 recommended for coding.

        Returns:
            dict with keys: icd10_codes, cpt_codes, rationale, confidence
            Returns {"error": "..."} if parsing fails.
        """
        messages = format_prompt(clinical_note)

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
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        new_tokens  = output_ids[0][input_ids.shape[1]:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._parse_output(output_text)

    def _parse_output(self, text: str) -> dict:
        """Parse model output to structured dict."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        try:
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return {"error": "Could not parse model output", "raw": text[:500]}


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def cli_mode(args):
    mc = MedCoder(args.model)
    note = args.note or input("Enter clinical note: ")
    result = mc.code(note)
    print("\n" + "="*60)
    print("CODING RESULT")
    print("="*60)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\nICD-10-CM Codes:")
    for c in result.get("icd10_codes", []):
        tag = "[PRIMARY]" if c.get("type") == "primary" else "[SECONDARY]"
        print(f"  {tag} {c['code']}  —  {c['description']}")

    if result.get("cpt_codes"):
        print("\nCPT Codes:")
        for c in result["cpt_codes"]:
            print(f"  {c['code']}  —  {c['description']}")

    print(f"\nRationale: {result.get('rationale', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print("="*60)


# ─────────────────────────────────────────────
# REST API SERVER (FastAPI)
# ─────────────────────────────────────────────
def serve_mode(args):
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("Run: pip install fastapi uvicorn pydantic")
        return

    app   = FastAPI(title="MedCoder-SLM-1B API", version="1.0.0")
    coder = MedCoder(args.model)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST"],
        allow_headers=["*"],
    )

    class CodeRequest(BaseModel):
        clinical_note: str
        temperature: float = 0.1
        max_tokens:  int   = 512

    class CodeResponse(BaseModel):
        icd10_codes: list
        cpt_codes:   list
        rationale:   str
        confidence:  float

    @app.post("/code", response_model=None)
    async def code_note(req: CodeRequest):
        result = coder.code(req.clinical_note, req.max_tokens, req.temperature)
        return result

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "MedCoder-SLM-1B"}

    print(f"\nMedCoder API running at http://localhost:{args.port}")
    print("POST /code  — submit a clinical note, get ICD-10 + CPT codes")
    print("GET  /health\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main():
    parser = argparse.ArgumentParser(description="MedCoder-SLM-1B Inference")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--note",  default=None, help="Clinical note text")
    parser.add_argument("--serve", action="store_true", help="Run as REST API")
    parser.add_argument("--port",  type=int, default=8080)
    args = parser.parse_args()

    if args.serve:
        serve_mode(args)
    else:
        cli_mode(args)


if __name__ == "__main__":
    main()

# backend/app.py
from dotenv import load_dotenv
load_dotenv()

import os, tempfile, subprocess, json, time, sys, logging, hashlib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from transformers import pipeline
from huggingface_hub import InferenceClient

# -------------------------
# Logging & simple cache
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-code-scanner")
CACHE = {}

def cache_key(code, filename, language):
    h = hashlib.sha256((language + filename + code).encode('utf-8')).hexdigest()
    return h

# -------------------------
# Gemini client (optional)
# -------------------------
try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI()

# -------------------------
# Hugging Face model
# -------------------------
HF_MODEL = "mrm8488/codebert-base-finetuned-detect-insecure-code"
classifier = None
use_hf_inference_api = False
hf_inference_client = None

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if HF_API_TOKEN:
    try:
        hf_inference_client = InferenceClient(token=HF_API_TOKEN)
        use_hf_inference_api = True
        logger.info("Using Hugging Face Inference API")
    except Exception as e:
        logger.warning(f"HF API init failed: {e}")
        use_hf_inference_api = False

if not use_hf_inference_api:
    try:
        classifier = pipeline("text-classification", model=HF_MODEL, tokenizer=HF_MODEL)
        logger.info("Loaded local HF classifier")
    except Exception as e:
        logger.warning(f"Local HF load failed: {e}")
        classifier = None
        if HF_API_TOKEN:
            try:
                hf_inference_client = InferenceClient(token=HF_API_TOKEN)
                use_hf_inference_api = True
            except Exception:
                use_hf_inference_api = False

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

# -------------------------
# Request schema
# -------------------------
class AnalyzeReq(BaseModel):
    language: str
    filename: str
    code: str

def _normalize_semgrep_item(item: Dict) -> Dict:
    start = item.get("start") or item.get("extra", {}).get("start") or {}
    end = item.get("end") or item.get("extra", {}).get("end") or {}
    extra = item.get("extra", {})
    return {
        "start": {"line": start.get("line", 1), "col": start.get("col", 0)},
        "end": {"line": end.get("line", start.get("line", 1)), "col": end.get("col", 0)},
        "extra": extra,
        "msg": item.get("msg") or extra.get("message") or item.get("message")
    }

# -------------------------
# Analyze endpoint
# -------------------------
@app.post("/analyze")
async def analyze(req: AnalyzeReq) -> Dict[str, Any]:
    start_time = time.time()
    findings: Dict[str, Any] = {
        "semgrep": [], "bandit": [], "hf": None,
        "hf_findings": [], "gemini": None, "gemini_findings": []
    }

    # Cache check
    try:
        key = cache_key(req.code, req.filename, req.language)
        cached = CACHE.get(key)
        if cached and time.time() - cached["ts"] < 3600:
            res = cached["val"]
            res["cached"] = True
            res["timing_s"] = time.time() - start_time
            return res
    except:
        pass

    # Write temp file
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix="."+req.language, delete=False, mode="w", encoding="utf-8") as tf:
            path = tf.name
            tf.write(req.code)
    except Exception as e:
        findings["error"] = f"tempfile_error: {e}"
        findings["timing_s"] = time.time() - start_time
        return findings

    # -------------------------
    # Semgrep
    # -------------------------
    try:
        semgrep_cmd = ["semgrep", "--json", path]
        proc = subprocess.run(semgrep_cmd, capture_output=True, text=True, timeout=60)
        if proc.stdout:
            data = json.loads(proc.stdout)
            findings["semgrep"] = [_normalize_semgrep_item(i) for i in data.get("results", [])]
    except:
        pass

    # -------------------------
    # Bandit (Python only)
    # -------------------------
    if req.language.lower() in ["py", "python"]:
        try:
            proc = subprocess.run(["bandit", "-f", "json", "-r", path], capture_output=True, text=True, timeout=10)
            if proc.stdout:
                bdata = json.loads(proc.stdout)
                findings["bandit"] = bdata.get("results", [])
        except:
            pass

    # -------------------------
    # Hugging Face
    # -------------------------
    try:
        if use_hf_inference_api and hf_inference_client:
            res = hf_inference_client.text_generation(HF_MODEL, req.code)
            findings["hf"] = res
        elif classifier:
            hf_result = classifier(req.code[:4096])
            findings["hf"] = hf_result
    except:
        pass

    # HF synthetic findings
    try:
        if isinstance(findings.get("hf"), list) and findings["hf"]:
            top = findings["hf"][0]
            label = top.get("label","")
            score = float(top.get("score",0))
            if label.upper().startswith("LABEL_1") or score>0.7:
                for idx, line in enumerate(req.code.splitlines()):
                    if line.strip():
                        findings["hf_findings"].append({
                            "message": f"HF flagged vulnerability (label={label}, score={score:.2f})",
                            "line": idx+1,
                            "col":0
                        })
                        break
    except:
        pass

    # -------------------------
    # Gemini
    # -------------------------
    try:
        escalate = bool(findings.get("semgrep") or findings.get("bandit") or findings.get("hf_findings"))
        if escalate and gemini_client:
            prompt = (
                f"Analyze this code for vulnerabilities and return JSON: line(int), type, severity(low/medium/high), explanation.\n\n"
                f"Code:\n```\n{req.code}\n```"
            )
            resp = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            findings["gemini"] = resp.text
            try:
                gjson = json.loads(resp.text)
                line = max(0, (gjson.get("line") or 1)-1)
                findings["gemini_findings"].append({
                    "message": f"{gjson.get('type','Vulnerability')} - {gjson.get('explanation','')}",
                    "line": line,
                    "col":0,
                    "severity": gjson.get("severity","medium")
                })
            except:
                pass
    except:
        pass

    # Timing & cache
    findings["timing_s"] = time.time() - start_time
    try:
        CACHE[key] = {"ts": time.time(), "val": findings}
    except:
        pass

    # Cleanup
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except:
        pass

    return findings

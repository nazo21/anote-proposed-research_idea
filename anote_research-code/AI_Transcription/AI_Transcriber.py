from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
from transformers import pipeline
import tempfile, os, requests, json
import re

# ── Optional Qwen import — only loaded when USE_QWEN=True ────────────────────
# If you don't have the qwen_asr package installed, leave USE_QWEN = False.
# USE_QWEN = True will attempt: from qwen_asr import Qwen3ASRModel

app = FastAPI(title="piVoT AI Transcriber", version="1.0.0")

# ── Device selection (Apple Silicon MPS → CPU fallback) ──────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device set to use {DEVICE}")

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Toggle models here ────────────────────────────────────────────────────────
USE_QWEN = False  # Switch to True when Qwen3 is stable on your platform

# ── ASR model setup ──────────────────────────────────────────────────────────
if USE_QWEN:
    from qwen_asr import Qwen3ASRModel
    asr_model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-0.6B",
        max_new_tokens=256,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=torch.float16, device_map=DEVICE),
    )
    def transcribe(path: str) -> str:
        return asr_model.transcribe(audio=[path], language=None)[0].text
else:
    _pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=DEVICE,
        chunk_length_s=30,      # handles audio > 30 s
        return_timestamps=True,
    )
    def transcribe(path: str) -> str:
        result = _pipe(path, return_timestamps=True)
        return result["text"]


# ── Few-shot training data ────────────────────────────────────────────────────
# info.json lives next to this file. Adjust the path if needed.
_json_path = os.path.join(os.path.dirname(__file__), "info.json")
with open(_json_path) as f:
    training_data = json.load(f)


def build_few_shot(data: list, n: int = 3) -> str:
    """Return up to *n* completed training examples as a formatted string."""
    examples = ""
    try:
        completed = [
            d for d in data
            if not all(
                d["output"]["SOAP"].get(k, "") == "" and
                d["output"]["SBAR"].get(k2, "") == ""
                for k, k2 in zip(
                    ["subject", "objective", "assessment", "plan"],
                    ["situation", "background", "assessment", "recommendation"]
                )
            )
        ]
        for ex in completed[:n]:
            examples += f"Input: {ex['input']}\nOutput: {json.dumps(ex['output'])}\n\n"
    except (KeyError, TypeError) as e:
        print("build_few_shot error:", e)
    return examples


# ── Extraction prompt ─────────────────────────────────────────────────────────
PROMPT = """
You are a medical SOAP & SBAR note extraction assistant.

Your ONLY job is to extract information EXPLICITLY stated in the transcript.
You may make a clinical assumption ONLY when you are 100% certain it is medically correct.
Medical errors—even minor ones—are taken seriously. Do NOT take this lightly.

STRICT RULES:
- If a value is NOT stated, output "Action Required" for that field. Never invent data.
- Do not infer, summarize, or add clinical context beyond what is stated.
- Copy exact words from the transcript; do not rephrase unless needed for grammar.
- Return ONLY the JSON object. No explanation, no markdown fences.
- If you add any clinical context not directly stated, append "(THIS WAS ADDED BY AI)" to that field.
- Always convert Fahrenheit to Celsius: C = (F - 32) * 5/9, rounded to nearest hundredth.
- For any missing VITALS, output "Action Required".
- DO NOT copy the example data below into the output for a real transcript.
  The examples are reference only; treat every input as a brand new case.

Here are some reference examples (do NOT copy into a real output):
{few_shot}

Return ONLY a JSON object with this exact structure:

{{
  "Patient": {{
    "Room #": "",
    "Medical_Background": "",
    "Reason for Visit": "",
    "Current Status of Patient": "",
    "Special Notes for Caregiver": ""
  }},
  "SOAP": {{
    "subject": "",
    "objective": "",
    "assessment": "",
    "plan": ""
  }},
  "SBAR": {{
    "situation": "",
    "background": "",
    "assessment": "",
    "recommendation": ""
  }},
  "VITALS": {{
    "Heart Rate": "",
    "Oxygen Level": "",
    "Blood Pressure": "",
    "Temperature": "",
    "Respirations": ""
  }},
  "Action": {{
    "Current Goal of Patient": "",
    "Last Activity Caregiver took": "",
    "Last seen by Nurse": "",
    "Last seen by Physician": ""
  }}
}}

Now extract from this:
Input: {transcript}
Output:"""


# ── Empty fallback structure ──────────────────────────────────────────────────
def _empty_result(raw: str = "") -> dict:
    return {
        "Patient": {
            "Room #": None,
            "Medical_Background": None,
            "Reason for Visit": None,
            "Current Status of Patient": None,
            "Special Notes for Caregiver": None,
        },
        "SOAP": {
            "subject": None,
            "objective": None,
            "assessment": None,
            "plan": None,
        },
        "SBAR": {
            "situation": None,
            "background": None,
            "assessment": None,
            "recommendation": None,
        },
        "VITALS": {
            "Heart Rate": None,
            "Oxygen Level": None,
            "Blood Pressure": None,
            "Temperature": None,
            "Respirations": None,
        },
        "Action": {
            "Current Goal of Patient": None,
            "Last Activity Caregiver took": None,
            "Last seen by Nurse": None,
            "Last seen by Physician": None,
        },
        "_raw": raw,
    }


# ── Llama3 extraction via Ollama ──────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def extract(transcript: str) -> dict:
    few_shot = build_few_shot(training_data)
    prompt = PROMPT.format(few_shot=few_shot, transcript=transcript)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repeat_penalty": 1.0,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print("Ollama request error:", e)
        return _empty_result(str(e))

    raw = response.json().get("response", "")
    clean = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        print("JSON parse error. Raw response:\n", clean[:500])
        result = _empty_result(clean)
        return result


# ── Routes ────────────────────────────────────────────────────────────────────

# Serve proto.html at root
@app.get("/")
async def serve_frontend():
    index_path = os.path.join(os.path.dirname(__file__), "..", "proto.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="proto.html not found")
    return FileResponse(index_path)


@app.get("/transcribe-and-extract")
async def transcribe_page():
    index_path = os.path.join(os.path.dirname(__file__), "..", "proto.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="proto.html not found")
    return FileResponse(index_path)


@app.post("/transcribe-and-extract")
async def transcribe_and_extract(
    file: UploadFile = File(None),
    transcript: str = Form(None),
):
    """
    Accepts either:
      - an audio file (multipart `file` field) → run Whisper/Qwen → Llama3
      - a pre-transcribed string (`transcript` field) → skip ASR, run Llama3 only
    """
    try:
        if transcript:
            # Resend / edited-text path — skip ASR
            text = transcript

        elif file:
            # Determine file suffix from content-type or filename
            filename = file.filename or "audio"
            suffix = os.path.splitext(filename)[-1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            try:
                text = transcribe(tmp_path)
            finally:
                os.remove(tmp_path)

        else:
            return {
                "transcript": None,
                "extracted": None,
                "error": "No input provided. Send an audio file or transcript text.",
            }

        extracted = extract(text)

        print("TRANSCRIPT:", text)
        print("EXTRACTED:", json.dumps(extracted, indent=2))

        return {"transcript": text, "extracted": extracted}

    except Exception as e:
        print("ERROR:", str(e))
        return {"transcript": None, "extracted": None, "error": str(e)}


@app.post("/submit")
async def submit_notes(payload: dict):
    """
    Receives the final (possibly user-edited) note for a given room.
    Extend this endpoint to write to a database or EHR system.
    """
    room = payload.get("room")
    if not room:
        raise HTTPException(status_code=400, detail="Room number is required.")
    # TODO: persist payload to your database / EHR integration here
    print(f"Notes submitted for room {room}:", json.dumps(payload, indent=2))
    return {"status": "ok", "room": room}


# ── Static files — MUST be mounted after all route definitions ────────────────
# FIX: removed the duplicate app.mount that was defined at the top of the file
_static_root = os.path.join(os.path.dirname(__file__), "..")
if os.path.isdir(_static_root):
    app.mount("/static", StaticFiles(directory=_static_root), name="static")
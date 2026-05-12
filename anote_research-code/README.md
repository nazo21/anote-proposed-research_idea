> ⚠️ **DEMO for Anote Research Paper Idea.** Current build not meant for commerical use. Purely research purposes in exploration of AI agents in Medical Field

---

## Current Features

- **Audio transcription** — Whisper (default) or Qwen3-ASR
- **Structured extraction** — SOAP, SBAR, Vitals, Patient info via Llama 3
- **Editable output** — nurses can correct fields inline before submitting
- **Fully local** — no cloud APIs, no PHI leaves the device

---

## Requirements

| Dependency                   | Version              |
| ---------------------------- | -------------------- |
| Python                       | 3.10 +               |
| [Ollama](https://ollama.com) | Latest               |
| CUDA / Apple MPS             | Optional (CPU works) |

---

## Project Structure

```
root/
├── AI_Transcription/
│   └── AI_Transcriber.py   ← FastAPI server (main backend)
├── proto.html              ← Frontend UI  (rename main.html → index.html)
├── info.json               ← Few-shot training examples
├── requirements.txt
└── README.md
```

> The backend serves `../proto.html` relative to its own location.  
> Keep `AI_Transcriber.py` in a `AI_Transcription/` subfolder, or adjust the paths in the file.

---

## Installation

### 1 — Clone the repo to your own branch

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
fastapi
uvicorn[standard]
torch
transformers
requests
python-multipart
```

> On Apple Silicon you may want `torch` installed via the MPS wheel.  
> On CUDA machines, install the appropriate `torch` for your CUDA version from [pytorch.org](https://pytorch.org).

### 4 — Install and start Ollama

```bash
# Install Ollama (macOS / Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3.2 (≈ 2 GB)
ollama pull llama3.2

# Start the Ollama server (runs on http://localhost:11434)
ollama serve
```

> Windows: download the installer from [ollama.com](https://ollama.com) and run `ollama serve` in a terminal.

### 5 — Rename the frontend file

```bash
mv main.html index.html
```

The backend serves `proto.html` from the project root.

You can always change the source code to what your preferences as well

---

## Running the App

```bash
# From the project root
uvicorn backend.AI_Transcriber:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## Environment Variables

| Variable       | Default                               | Description              |
| -------------- | ------------------------------------- | ------------------------ |
| `OLLAMA_URL`   | `http://localhost:11434/api/generate` | Ollama API endpoint      |
| `OLLAMA_MODEL` | `llama3.2`                            | LLM model name in Ollama |

Example:

```bash
OLLAMA_MODEL=llama3.1 uvicorn backend.AI_Transcriber:app --reload
```

---

## Switching to Qwen3-ASR

If you want to use Qwen3 instead of Whisper, set `USE_QWEN = True` in `AI_Transcriber.py` and install:

```bash
pip install qwen-asr   # package name may vary — check the Qwen repo
```

Qwen3-ASR requires additional VRAM and is best on an M2/M3 Mac or CUDA GPU.

---

## API Endpoints

| Method | Path                      | Description                                         |
| ------ | ------------------------- | --------------------------------------------------- |
| `GET`  | `/`                       | Serves the frontend                                 |
| `POST` | `/transcribe-and-extract` | Upload audio file **or** send plain transcript text |
| `POST` | `/submit`                 | Save final reviewed note (extend for EHR/DB)        |

### POST `/transcribe-and-extract`

Send as `multipart/form-data`:

| Field        | Type   | Description                                |
| ------------ | ------ | ------------------------------------------ |
| `file`       | File   | Audio file (`.wav`, `.mp3`, `.m4a`, etc.)  |
| `transcript` | String | Pre-typed or edited transcript (skips ASR) |

### Example with `curl`

```bash
# Audio file
curl -X POST http://localhost:8000/transcribe-and-extract \
     -F "file=@recording.wav"

# Plain text
curl -X POST http://localhost:8000/transcribe-and-extract \
     -F "transcript=Patient reports chest pain and shortness of breath."
```

---

## Output Format

```json
{
  "transcript": "Patient reports...",
  "extracted": {
    "Patient": {
      "Room #": "",
      "Medical_Background": "",
      "Reason for Visit": "",
      "Current Status of Patient": "",
      "Special Notes for Caregiver": ""
    },
    "SOAP": {
      "subject": "",
      "objective": "",
      "assessment": "",
      "plan": ""
    },
    "SBAR": {
      "situation": "",
      "background": "",
      "assessment": "",
      "recommendation": ""
    },
    "VITALS": {
      "Heart Rate": "",
      "Oxygen Level": "",
      "Blood Pressure": "",
      "Temperature": "",
      "Respirations": ""
    },
    "Action": {
      "Current Goal of Patient": "",
      "Last Activity Caregiver took": "",
      "Last seen by Nurse": "",
      "Last seen by Physician": ""
    }
  }
}
```

Missing values are returned as `"Action Required"`.  
Any AI-inferred content (not directly stated) is flagged with `(THIS WAS ADDED BY AI)`.

---

## Adding Training Examples

Edit `info.json` to add more few-shot examples in the existing format:

```json
{
  "instruction": "Extract structured medical SOAP and SBAR note data as JSON",
  "input": "Your raw nurse note here...",
  "output": {
    "SOAP": {
      "subject": "...",
      "objective": "...",
      "assessment": "...",
      "plan": "..."
    },
    "SBAR": {
      "situation": "...",
      "background": "...",
      "assessment": "...",
      "recommendation": "..."
    }
  }
}
```

The backend uses up to 3 examples per request for few-shot prompting.

---

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push and open a PR

---

## License

MIT License — free to use, modify, and distribute. See `LICENSE` for details.

---

## Disclaimer

The current demo is a research and educational tool. It is **not** a substitute for professional clinical judgment. Always have a licensed clinician review AI-generated notes before use in patient care.

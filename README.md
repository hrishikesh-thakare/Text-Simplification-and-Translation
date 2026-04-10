# Text Simplification + Indic Translation Pipeline

This project takes **complex English text**, uses a local language model to break it down into simpler sentences to improve readability, and then translates the simplified output into a variety of **Indic languages**.

## What This Project Is

The app runs a sophisticated 2-step NLP pipeline:

1. **Anti-Hallucination English Simplification:** Uses a local LoRA adapter at `model/simplifier-4090`. It actively splits inputs sentence-by-sentence, verifies faithfulness against the source (to prevent hallucinated facts or dropped entities), and reconstructs the text.
2. **English -> Indic Translation:** Uses the state-of-the-art `ai4bharat/indictrans2-en-indic-dist-200M` model to seamlessly translate the output into your choice of supported languages.

Final output includes:
- Clean, vertical progression view
- Simplified English text (with latency metrics)
- Translated Indic text (with latency metrics)

## Tech Stack

- **Language:** Python 3
- **Core Libraries:**
  - `gradio` (Modern web UI framework)
  - `transformers`, `torch`, `peft`
  - `IndicTransToolkit`
- **Models:**
  - Base Simplifier: `unsloth/llama-3-8b-bnb-4bit` + adapter: `model/simplifier-4090`
  - Translator: `ai4bharat/indictrans2-en-indic-dist-200M`
- **Runtime:** CUDA-first when available, automatic fallback for low VRAM setups (e.g., RTX 3050 6GB)

## Project Files

- `web_app.py` - The primary Gradio interactive web application interface (Entry Point)
- `simplify.py` - Core English simplification logic, LLM wrapping, and faithfulness verification rules
- `translate.py` - English -> Indic translation model component wrapper
- `requirements.txt` - Python dependencies

## First-Time Setup

### 1. Open terminal in this folder

```powershell
cd text-simplification
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Hugging Face Access (Important)

The translation model and simplifier base model are pulled from Hugging Face on the first run.
- Login from terminal:

```powershell
huggingface-cli login
```
When prompted in CLI, paste your Hugging Face token and press Enter. Ensure your Hugging Face account has requested access to the base models if prompted.

## How To Run

The entire pipeline is wrapped in a beautiful, pipeline-oriented web interface.

```powershell
python web_app.py
```

Then open **`http://127.0.0.1:7860`** in your browser.

**Web App Features:**
- **Dynamic Translation:** Process text into Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, or Urdu.
- **Pipeline Progress Tracking:** Step-by-step visual indication (Input → Simplify → Translate).
- **Runtime Metrics:** Each output card tells you exactly how many seconds the model took to process step.
- **Model Telemetry:** Click "Runtime / Model Info" to verify your active device (CUDA/CPU), adapter paths, and fallback execution status.
- **Anti-Hallucination:** Prevents the simplification model from adding explanatory loops or skipping important nouns.

## What Happens on First Run

- Base models are downloaded from Hugging Face (can take time).
- Local LoRA adapter is loaded from `model/simplifier-4090`.
- Internet connection is required for the initial fetch.
- Startup is slower on the first run due to model download. Later runs are extremely fast because models load natively from your local Hugging Face cache.

## RTX 3050 6GB Notes

- The Simplifier loads with a highly optimized memory strategy:
  - Tries full CUDA runtime first.
  - Falls back to automatic offload if VRAM runs out.
  - Final fallback to CPU full precision if 4-bit runtime fails on your setup.
- This ensures your pipeline runs reliably without crashing your PC, even when rendering heavy transformer sequences.

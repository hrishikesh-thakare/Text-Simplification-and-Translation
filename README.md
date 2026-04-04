# Text Simplification + Hindi Translation Pipeline

This project takes **complex English text**, simplifies it, and then translates the simplified output to **Hindi**.

## What This Project Is

The app runs a 2-step NLP pipeline:

1. **English Simplification** using local LoRA adapter at `model/simplifier-4090`
2. **English -> Hindi Translation** using `ai4bharat/indictrans2-en-indic-dist-200M`

Final output includes:

- Original complex English text
- Simplified English text
- Hindi translation

## Tech Stack

- **Language:** Python 3
- **Core libraries:**
  - `transformers`
  - `torch`
  - `sentencepiece`
  - `sacremoses`
  - `IndicTransToolkit`
  - `protobuf`
- **Models:**
  - Base: `unsloth/llama-3-8b-bnb-4bit` + adapter: `model/simplifier-4090`
  - `ai4bharat/indictrans2-en-indic-dist-200M`
- **Runtime:** CUDA-first when available, automatic fallback for low VRAM

## Project Files

- `main.py` - End-to-end interactive pipeline (recommended entry point)
- `simplify.py` - English simplification model wrapper
- `translate.py` - English -> Hindi translation model wrapper
- `simplifier.py` - Alternate API-based simplification experiment
- `requirements.txt` - Python dependencies

## First-Time Setup

### 1. Open terminal in this folder

```powershell
cd text-simplification
```

### 2. (Recommended) Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Hugging Face access (important for first run)

The translation model and simplifier base model are pulled from Hugging Face.

- Create/login to Hugging Face account
- Request access to: `ai4bharat/indictrans2-en-indic-dist-200M` (if prompted)
- Login from terminal:

```powershell
huggingface-cli login
```

When prompted in CLI, paste your Hugging Face token and press Enter.

## How To Run

### Option 1: Web App (Recommended)

```powershell
python web_app.py
```

Then open `http://127.0.0.1:7860` in your browser.

**Features:**
- Clean web interface (built with Gradio)
- Real-time processing with visual feedback
- Shows active runtime and model source
- Displays simplified text and Hindi translation side-by-side
- Uses the local 4090 simplifier from `model/simplifier-4090`

### Option 2: Interactive Terminal Pipeline

```powershell
python main.py
```

**Interactive commands:**
- Enter any complex English sentence to process it
- Type `examples` to run built-in sample texts
- Type `quit` (or `exit` / `q`) to stop
- Uses the local 4090 simplifier from `model/simplifier-4090`

### Option 3: Simplification Only

To use just the text simplifier (4090 LoRA adapter):

```powershell
python simplify.py
```

This runs the simplified model from `model/simplifier-4090` without translation.

## What Happens on First Run

- Base models are downloaded from Hugging Face (can take time)
- Local LoRA adapter is loaded from `model/simplifier-4090`
- Internet connection is required
- Startup is slower on first run due to model download and caching
- Later runs are faster because models are loaded from local cache

## RTX 3050 6GB Notes

- Simplifier loads with a speed-first strategy:
  - try CUDA runtime first
  - fallback to automatic offload if VRAM is not enough
  - final fallback to CPU full precision if 4-bit runtime fails on your setup
- This keeps the pipeline running even when CUDA memory is tight.

## Simplifier Source Guarantee

- Simplification is loaded from local adapter path: `model/simplifier-4090`.
- The old `keep_it_simple` model is not used by `simplify.py`.
- The web UI shows active runtime and model source on each run.

## Example Flow

Input:

```text
The administration of comprehensive pharmacological interventions necessitates meticulous monitoring.
```

Output sections:

- Original complex text
- Simplified English
- Hindi translation

## Common Issues

- **Model loading fails:**
  - Ensure internet is available
  - Ensure Hugging Face login is done
  - Ensure access to translation model is approved
- **Slow performance:**
  - CPU inference is expected to be slower
  - First run is always the slowest

## Notes

- The web UI runs locally on `http://127.0.0.1:7860`
- You can adapt the device selection if your system supports it

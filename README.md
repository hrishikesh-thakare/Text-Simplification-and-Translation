# Text Simplification + Hindi Translation Pipeline

This project takes **complex English text**, simplifies it, and then translates the simplified output to **Hindi**.

## What This Project Is

The app runs a 2-step NLP pipeline:

1. **English Simplification** using `philippelaban/keep_it_simple`
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
  - `philippelaban/keep_it_simple`
  - `ai4bharat/indictrans2-en-indic-dist-200M`
- **Runtime:** CPU by default (GPU can be used if you modify device settings)

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

The translation model will require approved access and authentication.

- Create/login to Hugging Face account
- Request access to: `ai4bharat/indictrans2-en-indic-dist-200M` (if prompted)
- Login from terminal:

```powershell
huggingface-cli login
```

When prompted in CLI, paste your Hugging Face token and press Enter.

## How To Run

### Run the full interactive pipeline

```powershell
python main.py
```

After starting:
- Enter any complex English sentence to process it
- Type `examples` to run built-in sample texts
- Type `quit` (or `exit` / `q`) to stop

## What Happens on First Run

- Models are downloaded from Hugging Face (can take time)
- Internet connection is required
- Startup is slower on first run due to model download and caching
- Later runs are faster because models are loaded from local cache

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

- Current default device is `cpu`
- You can adapt code to use CUDA if your system supports it

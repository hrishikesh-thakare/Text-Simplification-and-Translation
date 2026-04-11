# Text Simplification + Indic Translation Pipeline

This project takes **complex English text**, uses a local language model to break it down into simpler sentences to improve readability, and then translates the simplified output into a variety of **Indic languages**.

## What This Project Is

The app runs a sophisticated 2-step NLP pipeline:

1. **Anti-Hallucination English Simplification:** Uses a local LoRA adapter at `model/simplifier-4090`. It actively splits inputs sentence-by-sentence, verifies faithfulness against the source (to prevent hallucinated facts or dropped entities), and reconstructs the text.
2. **English → Indic Translation:** Uses the state-of-the-art `ai4bharat/indictrans2-en-indic-dist-200M` model to seamlessly translate the output into your choice of supported languages.

## Evaluation Results

All metrics evaluated on `wikilarge_test.csv` (191 sentences) using an anti-cheat evaluation pipeline with sequence-similarity filtering.

### SARI (Simplification Quality)

| Metric | Score |
|--------|-------|
| **SARI** | `23.46` |

Evaluated with strict anti-cheat filtering — outputs with sequence similarity > 0.95 vs. source are excluded to prevent SARI inflation from copying.

---

### FKGL (Readability — Flesch-Kincaid Grade Level)

| Text | Grade Level |
|------|-------------|
| Source (complex input) | `11.88` |
| Predicted (simplified output) | `10.38` |

Lower grade level = easier to read. The model successfully reduces reading complexity by **1.5 grade levels**.

---

### BERTScore (Semantic Meaning Preservation)

| Comparison | F1 Score |
|------------|----------|
| Prediction vs. Reference | `0.2599` |
| Prediction vs. Source | `0.3097` |

BERTScore measures semantic similarity using `roberta-large` embeddings. Values are rescaled with baseline. The model rewrites content in new phrasing rather than direct copying.

---

### Copy Rate & Length Ratio (Anti-Cheat Diagnostics)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Copy Rate | `0.00%` | < 40% | ✅ No copying |
| Length Ratio | `1.04` | 0.7–0.9 | ⚠️ Slightly longer |

- **Copy Rate 0%** confirms all SARI scores are genuine — no metric inflation from passthrough.
- **Length Ratio 1.04** means the model lightly expands outputs — it paraphrases rather than deletes, which suppresses SARI (SARI rewards deletion of complex words).

---

### Summary

| Metric | Score | Notes |
|--------|-------|-------|
| SARI | 23.46 | Genuine score, no inflation |
| FKGL Reduction | −1.50 grades | Complexity is reduced |
| BERTScore (vs Ref) | 0.2599 | Different phrasing from reference |
| BERTScore (vs Src) | 0.3097 | Meaning partially preserved |
| Copy Rate | 0.00% | ✅ Clean |
| Length Ratio | 1.04 | Model expands slightly |

## Tech Stack

- **Language:** Python 3
- **UI:** Streamlit (`streamlit_app.py`)
- **Core Libraries:**
  - `streamlit`, `transformers`, `torch`, `peft`
  - `IndicTransToolkit`
  - `textstat`, `bert-score`, `evaluate` (for metrics)
- **Models:**
  - Base Simplifier: `unsloth/llama-3-8b-bnb-4bit` + adapter: `model/simplifier-4090`
  - Translator: `ai4bharat/indictrans2-en-indic-dist-200M`
- **Runtime:** CUDA-first when available, automatic fallback for low VRAM setups

## Project Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main Streamlit web app (Entry Point) |
| `simplify.py` | Core simplification logic with LoRA adapter |
| `translate.py` | IndicTrans2 translation wrapper |
| `requirements.txt` | Python dependencies |

## First-Time Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Hugging Face Access

```powershell
huggingface-cli login
```

Paste your Hugging Face token when prompted. Ensure you have access to the base models.

## How To Run

```powershell
python -m streamlit run streamlit_app.py
```

Then open **`http://localhost:8501`** in your browser.

**Supported Output Languages:** Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu.

## What Happens on First Run

- Base models are downloaded from Hugging Face (can take time).
- Local LoRA adapter is loaded from `model/simplifier-4090`.
- Internet connection is required for the initial fetch.
- Subsequent runs load from local HuggingFace cache — much faster.

## RTX 3050 6GB Notes

The simplifier loads with a memory-efficient strategy:
- Tries full CUDA runtime first.
- Falls back to automatic VRAM offload if memory is tight.
- Final fallback to CPU if 4-bit runtime fails.

This ensures the pipeline runs reliably without crashing, even on 6GB VRAM cards.

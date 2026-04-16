# Text Simplification + Indic Translation Pipeline

This project takes **complex English text**, uses a local language model to break it down into simpler sentences to improve readability, and then translates the simplified output into a variety of **Indic languages**.

Goal: Simplify complex English text into easier, readable form while preserving meaning, and translate it into Indic languages.

## What This Project Is

The app runs a sophisticated 2-step NLP pipeline:

1. **Anti-Hallucination English Simplification:** Uses a local LoRA adapter at `model/simplifier-4090`. It actively splits inputs sentence-by-sentence, verifies faithfulness against the source (to prevent hallucinated facts or dropped entities), and reconstructs the text.
2. **English → Indic Translation:** Uses the state-of-the-art `ai4bharat/indictrans2-en-indic-dist-200M` model to seamlessly translate the output into your choice of supported languages.

## Evaluation Results

All metrics are evaluated on `wikilarge_test.csv` (191 sentences). The main production recommendation is still the original runtime path (base + LoRA at runtime), with GGUF used primarily for faster deployment on low-VRAM machines.

### Runtime Baseline (Original Path)

| Metric                       | Value            |
| ---------------------------- | ---------------- |
| SARI                         | `23.46`          |
| FKGL (Source -> Predicted)   | `11.88 -> 10.38` |
| FRE (Source -> Predicted)    | `44.71 -> 54.53` |
| BERTScore F1 (vs Reference)  | `0.2599`         |
| BERTScore F1 (vs Source)     | `0.3097`         |
| Copy Rate (>0.95 similarity) | `0.00%`          |
| Length Ratio                 | `1.04`           |

### Runtime vs Quantized GGUF

The following metrics are computed for the two practical paths used on this machine:

- Original runtime path (`streamlit_app.py` + `simplify.py`)
- Quantized GGUF path (`gguf/web_app_gguf.py`)

The full merged model path is intentionally excluded from this comparison due to high latency and poor practicality on 6GB-class GPUs.

| Metric                             | Original Runtime | Quantized GGUF |
| ---------------------------------- | ---------------- | -------------- |
| SARI                               | 23.46            | 46.09          |
| FKGL (predicted, lower is simpler) | 10.38            | 10.71          |
| FRE (predicted, higher is easier)  | 54.53            | 48.92          |
| BERTScore F1 vs Reference          | 0.2599           | 0.6520         |
| BERTScore F1 vs Source             | 0.3097           | 0.8915         |
| Copy Rate (>0.95 similarity)       | 0.00%            | 50.79%         |
| Length Ratio                       | 1.04             | 0.89           |

SARI is shown in this top-level comparison for quick scanning and is also repeated in the dedicated GGUF section below for run-specific context.

The high BERTScore for GGUF (vs source) is largely influenced by copy-like outputs, which artificially increase semantic similarity without improving simplification quality.

Interpretation:

- Runtime path remains cleaner in anti-copy behavior and is the recommended quality-first path.
- GGUF is faster, but its high copy-like rate reduces true simplification quality, so metric scores must be interpreted carefully.
- Always combine automatic metrics with manual inspection.

### Metric Guide: Meaning, Direction, and Practical Targets

Use this quick guide when reading evaluation numbers.

| Metric                   | What it measures                                                | Better direction         | Practical target for this project                                              |
| ------------------------ | --------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------ |
| SARI                     | Quality of edit operations (add/delete/keep) against references | Higher is usually better | Prefer higher, but validate with copy rate + manual checks                     |
| FKGL                     | Reading grade level (text complexity)                           | Lower is better          | Predicted FKGL should be lower than source                                     |
| FRE                      | Reading ease score                                              | Higher is better         | Predicted FRE should be higher than source                                     |
| BERTScore (vs Reference) | Semantic similarity to reference simplifications                | Higher is better         | Moderate-to-high is good; low can still be acceptable with diverse paraphrases |
| BERTScore (vs Source)    | Meaning preservation from original input                        | Higher is better         | Higher is preferred, but not at the cost of copying                            |
| Length Ratio             | Output length vs input length                                   | Mid-range is best        | Around 0.7-0.9 is often ideal for simplification                               |
| Copy Rate                | Near-copy percentage from source                                | Lower is better          | As low as possible; typically < 40%                                            |

Important notes:

- There is no single universal "optimal" score for simplification.
- Best results come from balancing readability improvement (FKGL/FRE), meaning retention (BERTScore), and anti-copy behavior (Copy Rate).
- A model can score high on one metric and still be poor in practice, so always combine metrics with manual inspection.

## Key Insight

Effective text simplification requires balancing three factors:

- Readability (FKGL, FRE)
- Meaning preservation (BERTScore)
- True simplification behavior (SARI + low copy rate)

No single metric is sufficient on its own.

## Tech Stack

- **Language:** Python 3
- **UI:** Streamlit (`runtime/streamlit_app.py`)
- **Core Libraries:**
  - `streamlit`, `transformers`, `torch`, `peft`
  - `IndicTransToolkit`
  - `textstat`, `bert-score`, `evaluate` (for metrics)
- **Models:**
  - Base Simplifier: `unsloth/llama-3-8b-bnb-4bit` + adapter: `model/simplifier-4090`
  - Translator: `ai4bharat/indictrans2-en-indic-dist-200M`
- **Runtime:** CUDA-first when available, automatic fallback for low VRAM setups

## Project Files

| File                       | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| `app.py`                  | Runtime-only desktop launcher              |
| `runtime/streamlit_app.py` | Main Streamlit web app (Entry Point)        |
| `runtime/simplify.py`      | Core simplification logic with LoRA adapter |
| `translate.py`             | IndicTrans2 translation wrapper             |
| `requirements.txt`         | Python dependencies                         |
| `model/simplifier-4090/`    | Exported trained LoRA adapter used by runtime |

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

## Training (What Was Done)

Training has already been completed for this project. The simplification adapter shipped in this repository is the result of LoRA fine-tuning on WikiLarge.

What was done:

- Used the WikiLarge dataset from `data/wikilarge_train.csv`, `data/wikilarge_validation.csv`, and `data/wikilarge_test.csv`.
- Fine-tuned a LoRA adapter for complex-English to simple-English rewriting.
- Base model used for fine-tuning: `unsloth/llama-3-8b-bnb-4bit`.
- LoRA configuration: rank `r=16`, `lora_alpha=16`, `lora_dropout=0`, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Training ran to global step `6250` with evaluation every `200` steps.
- Intermediate checkpoints were produced during training (including step 6200 and step 6250), and the exported runtime adapter is available at `model/simplifier-4090/`.

Notebook workflow summary (executed training pipeline):

- Imported `unsloth`, `transformers`, `datasets`, and PyTorch training utilities.
- Loaded CSV splits for train/validation/test and used a reduced train subset of 50,000 examples for the run.
- Converted each sample into an instruction-style prompt with strict simplification constraints (simple words, short output, no extra facts).
- Loaded `unsloth/llama-3-8b-bnb-4bit` in 4-bit mode with `max_seq_length=256` and applied LoRA adapters.
- Tokenized prompts with truncation and fixed-length padding (`max_length=256`).
- Trained with Hugging Face `Trainer` using: batch size `8`, gradient accumulation `1`, epochs `1`, learning rate `2e-4`, `fp16=True`, logging every `10` steps, eval/save every `200` steps, and `save_total_limit=2`.
- Ran post-training validation and test loss passes with DataLoader-based evaluation.
- Saved adapter + tokenizer using `save_pretrained("simplifier-4090")`.
- Loaded the saved adapter for inference and validated behavior with curated manual test sentences across domains (medical, legal, business, environment, engineering, and others).

Runtime now consumes these trained artifacts directly, so users typically do not need to retrain unless they want a new model variant or different training configuration.

If needed, runtime can still be pointed at a custom adapter path using `SIMPLIFIER_ADAPTER_PATH`.

Note: the original training workspace/checkpoint folders are intentionally not part of the committed repository; this section preserves the key training metadata in documentation.

## How To Run

For the runtime-only app launcher:

```powershell
python app.py
```

This starts the local Streamlit server for the runtime path and opens the browser automatically.

### Build a Windows exe

If you want a local Windows launcher, install PyInstaller in the active environment and run:

```powershell
.\build_runtime_launcher.ps1
```

That creates `dist\TextSimplifierRuntime\TextSimplifierRuntime.exe` for the runtime-only path.

### Run the packaged app

After building, start the packaged launcher:

```powershell
dist\TextSimplifierRuntime\TextSimplifierRuntime.exe
```

Expected behavior:

- A local Streamlit server is started on `http://127.0.0.1:8501`.
- Your browser opens automatically.
- A console window appears because the launcher hosts the local runtime process.

This is normal for the current Streamlit-based desktop launcher architecture.

Offline note for packaged app:

- The packaged app works offline once required model files are already available locally (inside the packaged folder and/or local Hugging Face cache).
- On a fresh machine without local model files, an initial online model fetch may be required before fully offline use.

For direct development mode:

```powershell
python -m streamlit run runtime/streamlit_app.py
```

Then open **`http://localhost:8501`** in your browser.

## Separate GGUF Website (Merged Quantized Model)

This project now includes a separate Gradio website that uses a merged quantized GGUF simplifier model, so it does not rely on loading base model + LoRA adapter at runtime.

### Default GGUF file discovery

The GGUF app checks these paths in order:

1. `SIMPLIFIER_GGUF_PATH` environment variable
2. `full/simplifier-8b-q4.gguf`
3. `simplifier-8b-q4.gguf` (project root)
4. `../simplifier-8b-q4.gguf` (workspace root)
5. `model/simplifier-8b-q4.gguf`

### Run GGUF website

```powershell
python gguf/web_app_gguf.py
```

Then open **`http://127.0.0.1:7861`**.

If running from inside the `gguf` folder, use:

```powershell
python web_app_gguf.py
```

Optional explicit path setup:

```powershell
$env:SIMPLIFIER_GGUF_PATH="C:\Users\hrish\Desktop\sb\simplifier-8b-q4.gguf"
python gguf/web_app_gguf.py
```

### Practical GPU Guidance (Important)

Most users do not have high-end GPUs. For generic GPUs (for example RTX 3060 6GB), use the GGUF website for normal usage.

- Recommended for everyday use on 6GB-class GPUs: `gguf/web_app_gguf.py`
- Keep `full/web_app_merged.py` for testing/experiments only (full merged model can be very slow on low VRAM)

Typical wait times on 6GB VRAM hardware:

- GGUF website (`gguf/web_app_gguf.py`): usually 10-90 seconds per request
- Original LoRA runtime (`runtime/streamlit_app.py`): usually 20-120 seconds per request

Suggested max wait before retrying:

- GGUF: 90 seconds
- Original LoRA runtime: 2 minutes

## Final Recommendation For This Project

Based on observed outputs and runtime behavior in this repository:

- The quantized GGUF path is faster, but simplification quality is not consistently as good as the runtime base+adapter pipeline.
- Therefore, the best overall approach for this project is the original runtime path (base model + `model/simplifier-4090` adapter loaded at runtime), balancing quality and usability on consumer hardware.

**Supported Output Languages:** Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu.

## What Happens on First Run

- If required model files are missing locally, base models are downloaded from Hugging Face (can take time).
- Local LoRA adapter is loaded from `model/simplifier-4090`.
- Internet is required only when local model files are not already present.
- After model files are available locally, the packaged app can run offline.

## RTX 3050 6GB Notes

The simplifier loads with a memory-efficient strategy:

- Tries full CUDA runtime first.
- Falls back to automatic VRAM offload if memory is tight.
- Final fallback to CPU if 4-bit runtime fails.

This ensures the pipeline runs reliably without crashing, even on 6GB VRAM cards.

Note: a separate full-merged checkpoint was tested experimentally, but is not recommended for typical 6GB GPUs due to high latency.

- `model/simplifier-merged` size on disk: **~14.97 GB**
- With short VRAM GPUs (for example 6GB), this model runs mostly with CPU/offload and can become very slow.

<!-- GGUF_SARI_START -->

## GGUF SARI Evaluation

- Dataset: WikiLarge test (191 samples)
- GGUF SARI: **46.09**
- Copy-like outputs (>0.95 similarity): 97

Generated by `eval/evaluate_sari_gguf.py`.

The significantly higher SARI score in the GGUF pipeline is misleading, as a large number of outputs remain copy-like. This highlights a known limitation of SARI, where superficial lexical edits can inflate scores without improving actual readability or usefulness.

<!-- GGUF_SARI_END -->

### Metric Caveat: Why Higher SARI Can Still Look Worse

A higher SARI score does not always mean better real-world simplification quality.

In our GGUF run, SARI improved, but many outputs were still copy-like or dropped important details. This happens because SARI rewards lexical edits against a reference, not full semantic faithfulness or practical usefulness for users.

Practical takeaway:

- Use SARI as one signal, not the only quality gate.
- Also track meaning preservation, copy rate, readability, and human judgment.
- Prefer outputs that are both simpler and faithful, even if SARI is slightly lower.

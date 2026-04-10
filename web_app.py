"""Web frontend for text simplification + Indic translation pipeline."""

from __future__ import annotations

import os
import time
import torch
import gradio as gr

from simplify import TextSimplifier
from translate import HindiTranslator


# Default to online-first so first-time setup works after cloning.
os.environ.setdefault("APP_OFFLINE", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ── Global ── */
.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Header ── */
#app-title {
    text-align: center;
    margin-bottom: 4px !important;
}
#app-title h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f97316, #fb923c, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#app-subtitle {
    text-align: center;
    margin-top: 0 !important;
    margin-bottom: 20px !important;
}
#app-subtitle p {
    color: #9ca3af !important;
    font-size: 0.9rem !important;
}

/* ── Pipeline Progress Bar ── */
#pipeline-steps {
    margin-bottom: 24px !important;
}
#pipeline-steps .pipeline-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 14px 20px;
    border-radius: 12px;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #6b7280;
    transition: color 0.3s ease;
}
.step-item.active {
    color: #f97316;
}
.step-item.done {
    color: #22c55e;
}
.step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #374151;
    transition: background 0.3s ease, box-shadow 0.3s ease;
}
.step-item.active .step-dot {
    background: #f97316;
    box-shadow: 0 0 8px rgba(249, 115, 22, 0.5);
}
.step-item.done .step-dot {
    background: #22c55e;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.4);
}
.step-arrow {
    color: #374151;
    font-size: 0.75rem;
    margin: 0 16px;
}

/* ── Cards ── */
.output-card {
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    background: #141414 !important;
    padding: 0 !important;
    overflow: hidden;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.output-card:hover {
    border-color: #3a3a3a !important;
}
.output-card.card-active {
    border-color: #f97316 !important;
    box-shadow: 0 0 20px rgba(249, 115, 22, 0.08) !important;
}
.output-card.card-done {
    border-color: #22c55e !important;
    box-shadow: 0 0 20px rgba(34, 197, 94, 0.06) !important;
}

/* Card header bar */
.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: #1a1a1a;
    border-bottom: 1px solid #2a2a2a;
}
.card-title {
    font-weight: 600;
    font-size: 0.85rem;
    color: #e5e7eb;
    letter-spacing: 0.01em;
}
.card-status {
    font-size: 0.75rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 9999px;
}
.status-waiting {
    color: #6b7280;
    background: #1f2937;
}
.status-active {
    color: #f97316;
    background: rgba(249, 115, 22, 0.12);
    animation: pulse-glow 1.5s ease-in-out infinite;
}
.status-done {
    color: #22c55e;
    background: rgba(34, 197, 94, 0.12);
}

@keyframes pulse-glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* ── Input section ── */
#input-section {
    margin-bottom: 0 !important;
}
#input-section textarea {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    color: #e5e7eb !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s ease;
}
#input-section textarea:focus {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.15) !important;
}

/* ── Controls row ── */
#controls-row {
    display: flex !important;
    gap: 12px !important;
    align-items: flex-end !important;
    margin-top: 12px !important;
}
#run-btn {
    min-height: 42px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    border: none !important;
    transition: transform 0.1s ease, box-shadow 0.2s ease !important;
}
#run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(249, 115, 22, 0.35) !important;
}
#run-btn:active {
    transform: translateY(0) !important;
}
#clear-btn {
    min-height: 42px !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #9ca3af !important;
}
#clear-btn:hover {
    border-color: #3a3a3a !important;
    color: #e5e7eb !important;
}

/* ── Output textboxes inside cards ── */
.card-body {
    padding: 0 !important;
}
.card-body textarea {
    background: transparent !important;
    border: none !important;
    color: #d1d5db !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    padding: 14px 16px !important;
}
.card-body .wrap {
    border: none !important;
    background: transparent !important;
}

/* ── Accordion (runtime info) ── */
.runtime-accordion {
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    background: #141414 !important;
    margin-top: 16px !important;
}
.runtime-accordion .label-wrap {
    padding: 12px 16px !important;
    font-size: 0.82rem !important;
    color: #6b7280 !important;
}

/* ── Example chips ── */
#examples-section {
    margin-top: 16px !important;
}
.examples-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px !important;
}
.example-chip {
    padding: 6px 14px !important;
    margin: 0 !important;
    border-radius: 9999px !important;
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #9ca3af !important;
    font-size: 0.8rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.example-chip:hover {
    border-color: #f97316 !important;
    color: #f97316 !important;
    background: rgba(249, 115, 22, 0.06) !important;
}

/* ── Spacing polish ── */
.gap-sm { margin-top: 12px !important; }
.gap-md { margin-top: 24px !important; }

/* ── Hide default gradio label on card outputs ── */
.card-body .label-wrap { display: none !important; }
.card-body .wrap { padding: 0 !important; }

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2a2a, transparent);
    margin: 24px 0;
    border: none;
}
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class WebPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.simplifier = TextSimplifier(device=self.device)
        self.translator = HindiTranslator(device=self.device)
        self.languages = self.translator.get_supported_languages()

    def get_runtime_info_str(self):
        info = self.simplifier.get_runtime_info()
        return (
            f"Simplifier source: {info['model_source']}\n"
            f"Base model: {info['base_model']}\n"
            f"Runtime: {info['runtime']}\n"
            f"Active device: {info['active_device']}\n"
            f"Offline mode: {info['offline_mode']}"
        )


pipeline = WebPipeline()


# ---------------------------------------------------------------------------
# Pipeline step HTML builders
# ---------------------------------------------------------------------------

def _step_html(input_state="waiting", simplify_state="waiting", translate_state="waiting"):
    """Build the 3-step pipeline progress bar HTML."""
    def _cls(state):
        return state  # 'waiting', 'active', or 'done'

    return f"""<div class="pipeline-bar">
        <div class="step-item {_cls(input_state)}">
            <span class="step-dot"></span> Input
        </div>
        <span class="step-arrow">&#x2192;</span>
        <div class="step-item {_cls(simplify_state)}">
            <span class="step-dot"></span> Simplify
        </div>
        <span class="step-arrow">&#x2192;</span>
        <div class="step-item {_cls(translate_state)}">
            <span class="step-dot"></span> Translate
        </div>
    </div>"""


def _card_header(title, status="waiting", time_str=None):
    labels = {"waiting": "Waiting", "active": "Processing...", "done": "Completed"}
    status_text = labels[status]
    if status == "done" and time_str:
        status_text = f"Completed in {time_str}"
        
    return f"""<div class="card-header">
        <span class="card-title">{title}</span>
        <span class="card-status status-{status}">{status_text}</span>
    </div>"""


# ---------------------------------------------------------------------------
# Generator-based pipeline (step-by-step output)
# ---------------------------------------------------------------------------

def process_text(text: str, target_language: str):
    """Generator that yields incremental UI updates for each pipeline step."""
    text = (text or "").strip()
    if not text:
        yield (
            _step_html("waiting", "waiting", "waiting"),
            _card_header("Simplified English", "waiting"),
            "",
            _card_header(f"{target_language} Translation", "waiting"),
            "",
        )
        return

    # ── Step 1: Input received, simplification starting ──
    yield (
        _step_html("done", "active", "waiting"),
        _card_header("Simplified English", "active"),
        "",
        _card_header(f"{target_language} Translation", "waiting"),
        "",
    )

    t0 = time.time()
    simplified = pipeline.simplifier.simplify_text(text)
    t1 = time.time()
    simp_time = f"{(t1 - t0):.1f}s"

    # ── Step 2: Simplification done, translation starting ──
    yield (
        _step_html("done", "done", "active"),
        _card_header("Simplified English", "done", time_str=simp_time),
        simplified,
        _card_header(f"{target_language} Translation", "active"),
        "",
    )

    t2 = time.time()
    translated = pipeline.translator.translate(simplified, target_language=target_language)
    t3 = time.time()
    trans_time = f"{(t3 - t2):.1f}s"

    # ── Step 3: All done ──
    yield (
        _step_html("done", "done", "done"),
        _card_header("Simplified English", "done", time_str=simp_time),
        simplified,
        _card_header(f"{target_language} Translation", "done", time_str=trans_time),
        translated,
    )


def clear_all():
    return (
        "",
        _step_html(),
        _card_header("Simplified English"),
        "",
        _card_header("Translation"),
        "",
    )


# ---------------------------------------------------------------------------
# Example chip data
# ---------------------------------------------------------------------------

EXAMPLES = {
    "Medical": (
        "The administration of comprehensive pharmacological interventions "
        "necessitates meticulous monitoring of patient vitals and laboratory parameters."
    ),
    "Legal": (
        "The promulgation of legislative provisions pertaining to fiscal accountability "
        "and governmental transparency endeavors to ameliorate bureaucratic efficacy."
    ),
    "Climate": (
        "Global temperatures are rising at an unprecedented rate due to the accumulation "
        "of greenhouse gases in the atmosphere. This phenomenon is leading to the rapid "
        "melting of polar ice caps and a significant rise in sea levels."
    ),
    "Technology": (
        "The integration of decentralized ledger technologies within contemporary "
        "financial ecosystems aims to facilitate immutable transaction transparency "
        "and diminish reliance on centralized intermediaries."
    ),
    "Education": (
        "Pedagogical methodologies that emphasize experiential learning are believed "
        "to foster greater student engagement and facilitate the long-term retention "
        "of complex theoretical concepts."
    ),
}


def load_example(example_name):
    return EXAMPLES.get(example_name, "")


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Text Simplification + Indic Translation",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0f0f0f",
        block_background_fill="#141414",
        block_border_width="0px",
        block_label_text_color="#9ca3af",
        block_title_text_color="#e5e7eb",
        input_background_fill="#141414",
        button_primary_background_fill="linear-gradient(135deg, #f97316, #ea580c)",
        button_primary_text_color="#ffffff",
        border_color_primary="#2a2a2a",
    ),
) as app:

    # ── Header ──
    gr.Markdown("# Text Simplification + Indic Translation", elem_id="app-title")
    gr.Markdown(
        "Enter complex English text, select a target Indic language, "
        "and get simplified + translated output.",
        elem_id="app-subtitle",
    )

    # ── Pipeline Progress ──
    pipeline_html = gr.HTML(
        value=_step_html(),
        elem_id="pipeline-steps",
    )

    # ── Input Section ──
    with gr.Column(elem_id="input-section"):
        input_text = gr.Textbox(
            label="Input Text",
            lines=5,
            max_lines=12,
            placeholder="Paste complex English text here...",
            show_copy_button=False,
        )

    # ── Controls Row ──
    with gr.Row(elem_id="controls-row"):
        language_dropdown = gr.Dropdown(
            choices=pipeline.languages,
            value="Hindi",
            label="Target Language",
            scale=2,
        )
        run_btn = gr.Button("Run Pipeline", variant="primary", scale=3, elem_id="run-btn")
        clear_btn = gr.Button("Clear", variant="secondary", scale=1, elem_id="clear-btn")

    # ── Divider ──
    gr.HTML('<div class="gap-md"></div><hr class="section-divider"><div class="gap-md"></div>')

    # ── Simplified Output Card ──
    with gr.Group(elem_classes=["output-card"]):
        simple_header = gr.HTML(
            value=_card_header("Simplified English"),
        )
        with gr.Column(elem_classes=["card-body"]):
            simple_out = gr.Textbox(
                label="Simplified",
                lines=4,
                show_label=False,
                show_copy_button=True,
                interactive=False,
            )

    # ── Spacer ──
    gr.HTML('<div class="gap-md"></div>')

    # ── Translation Output Card ──
    with gr.Group(elem_classes=["output-card"]):
        translated_header = gr.HTML(
            value=_card_header("Translation"),
        )
        with gr.Column(elem_classes=["card-body"]):
            translated_out = gr.Textbox(
                label="Translated",
                lines=4,
                show_label=False,
                show_copy_button=True,
                interactive=False,
            )

    # ── Divider ──
    gr.HTML('<div class="gap-md"></div><hr class="section-divider"><div class="gap-sm"></div>')

    # ── Runtime Info (collapsed) ──
    with gr.Accordion("Runtime / Model Info", open=False, elem_classes=["runtime-accordion"]):
        model_info_out = gr.Textbox(
            value=pipeline.get_runtime_info_str(),
            lines=5,
            show_label=False,
            interactive=False,
        )

    # ── Example Chips ──
    gr.HTML('<div class="gap-sm"></div>')
    gr.Markdown("**Try an example**", elem_id="examples-section")
    with gr.Row(elem_classes=["examples-row"]):
        for name in EXAMPLES:
            gr.Button(
                name,
                variant="secondary",
                elem_classes=["example-chip"],
            ).click(
                fn=lambda n=name: load_example(n),
                inputs=None,
                outputs=[input_text],
            )

    # ── Wiring ──
    run_btn.click(
        fn=process_text,
        inputs=[input_text, language_dropdown],
        outputs=[pipeline_html, simple_header, simple_out, translated_header, translated_out],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_text, pipeline_html, simple_header, simple_out, translated_header, translated_out],
    )


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)

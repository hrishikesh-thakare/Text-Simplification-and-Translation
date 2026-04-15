"""Separate Gradio website using the full merged simplifier model."""

from __future__ import annotations

import time
import torch
import gradio as gr
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from simplify_merged import MergedTextSimplifier
from translate import IndicTranslator


CSS = """
.gradio-container {
  max-width: 920px !important;
  margin: 0 auto !important;
}
#title h1 {
  text-align: center;
  letter-spacing: -0.02em;
  background: linear-gradient(130deg, #f97316, #0ea5e9);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
#subtitle p {
  text-align: center;
  color: #4b5563;
  margin-top: -6px;
}
"""


class MergedWebPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.simplifier = MergedTextSimplifier(device=self.device)
        self.translator = IndicTranslator(device=self.device)
        self.languages = self.translator.get_supported_languages()

    @staticmethod
    def _sync_cuda():
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def runtime_info(self) -> str:
        info = self.simplifier.get_runtime_info()
        return (
            f"Simplifier runtime: {info['runtime']}\n"
            f"Merged model path: {info['model_source']}\n"
            f"Requested device: {info['requested_device']}\n"
            f"Active device: {info['active_device']}\n"
            f"Translator model: {self.translator.model_name}"
        )


pipeline = MergedWebPipeline()


def run_pipeline(text: str, language: str):
    text = (text or "").strip()
    if not text:
        return "", "", "Please enter text to simplify."

    try:
        pipeline._sync_cuda()
        t0 = time.time()
        simplified = pipeline.simplifier.simplify_text(text)
        pipeline._sync_cuda()
        t1 = time.time()

        pipeline._sync_cuda()
        t2 = time.time()
        translated = pipeline.translator.translate(simplified, target_language=language)
        pipeline._sync_cuda()
        t3 = time.time()

        timing = (
            f"Simplification: {(t1 - t0):.2f}s | "
            f"Translation: {(t3 - t2):.2f}s | "
            f"Total: {(t3 - t0):.2f}s"
        )
        return simplified, translated, timing
    except Exception as exc:
        return "", "", f"Error: {exc}"


with gr.Blocks(
    title="Merged Model Simplification + Indic Translation",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
    ),
) as app:
    gr.Markdown("# Merged Model Simplification + Indic Translation", elem_id="title")
    gr.Markdown(
        "This separate site uses your full merged base+adapter model (non-GGUF).",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Complex English Input",
                placeholder="Paste complex English text here...",
                lines=8,
            )
            language = gr.Dropdown(
                choices=pipeline.languages,
                value="Hindi",
                label="Target Language",
            )
            run_button = gr.Button("Simplify and Translate", variant="primary")
            clear_button = gr.Button("Clear", variant="secondary")
        with gr.Column(scale=1):
            simplified_out = gr.Textbox(
                label="Simplified English (Merged)",
                lines=8,
                interactive=False,
                show_copy_button=True,
            )
            translated_out = gr.Textbox(
                label="Translated Output",
                lines=8,
                interactive=False,
                show_copy_button=True,
            )

    timing_out = gr.Textbox(label="Runtime", interactive=False)

    with gr.Accordion("Model Runtime Info", open=False):
        gr.Textbox(value=pipeline.runtime_info(), lines=8, interactive=False)

    run_button.click(
        fn=run_pipeline,
        inputs=[input_text, language],
        outputs=[simplified_out, translated_out, timing_out],
    )

    clear_button.click(
        fn=lambda: ("", "", "", ""),
        inputs=None,
        outputs=[input_text, simplified_out, translated_out, timing_out],
    )


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7862)

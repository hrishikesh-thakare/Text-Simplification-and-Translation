"""Web frontend for text simplification + Indic translation pipeline."""

from __future__ import annotations

import os
import torch
import gradio as gr

from simplify import TextSimplifier
from translate import HindiTranslator


# Default to online-first so first-time setup works after cloning.
# Users can override by setting APP_OFFLINE=1 before launch.
os.environ.setdefault("APP_OFFLINE", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")


class WebPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.simplifier = TextSimplifier(device=self.device)
        self.translator = HindiTranslator(device=self.device)
        self.languages = self.translator.get_supported_languages()

    def run(self, text: str, target_language: str):
        text = (text or "").strip()
        if not text:
            return "", "", "Please enter some text.", ""

        simplified = self.simplifier.simplify_text(text)
        translated = self.translator.translate(simplified, target_language=target_language)
        info = self.simplifier.get_runtime_info()
        model_info = (
            f"Simplifier source: {info['model_source']}\n"
            f"Base model: {info['base_model']}\n"
            f"Runtime: {info['runtime']}\n"
            f"Active device: {info['active_device']}\n"
            f"Offline mode: {info['offline_mode']}"
        )
        return text, simplified, translated, model_info


pipeline = WebPipeline()


def process_text(text: str, target_language: str):
    return pipeline.run(text, target_language)


EXAMPLES = [
    [
        "The administration of comprehensive pharmacological interventions necessitates meticulous monitoring of patient vitals and laboratory parameters."
    ],
    [
        "The promulgation of legislative provisions pertaining to fiscal accountability and governmental transparency endeavors to ameliorate bureaucratic efficacy."
    ],
]


with gr.Blocks(title="Text Simplification + Indic Translation") as app:
    gr.Markdown("# Text Simplification + Indic Translation")
    gr.Markdown(
        "Enter complex English text, then select the target Indic language for translation."
    )

    with gr.Row():
        input_text = gr.Textbox(
            label="Complex English Input",
            lines=6,
            placeholder="Type complex English text here...",
        )
    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=pipeline.languages,
            value="Hindi",
            label="Target Language",
        )

    with gr.Row():
        run_btn = gr.Button("Run Pipeline", variant="primary")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        original_out = gr.Textbox(label="Original Complex Text", lines=4)
        simple_out = gr.Textbox(label="Simplified English", lines=4)

    with gr.Row():
        translated_out = gr.Textbox(label="Translated Output", lines=4)

    model_info_out = gr.Textbox(label="Runtime / Model Info", lines=5)

    run_btn.click(
        fn=process_text,
        inputs=[input_text, language_dropdown],
        outputs=[original_out, simple_out, translated_out, model_info_out],
    )

    clear_btn.click(
        fn=lambda: ("", "", "", "", ""),
        inputs=None,
        outputs=[input_text, original_out, simple_out, translated_out, model_info_out],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text],
    )


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)

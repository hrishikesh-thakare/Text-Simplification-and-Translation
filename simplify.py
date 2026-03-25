"""
Text simplification module using philippelaban/keep_it_simple.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class TextSimplifier:
    """Simplifies complex English text with keep_it_simple."""

    def __init__(self, device="cpu"):
        self.device = device
        self.model_name = "philippelaban/keep_it_simple"

        print(f"Loading Text Simplification Model from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ Model loaded successfully on {self.device}")

    def simplify_text(self, complex_text: str, max_length: int = 256) -> str:
        prompt = f"Simplify: {complex_text}\nSimplified:"

        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=max_length,
                num_beams=4,
                no_repeat_ngram_size=4,
                length_penalty=1.0,
                repetition_penalty=1.8,
                do_sample=False,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if "Simplified:" in decoded:
            decoded = decoded.split("Simplified:", 1)[1].strip()
        return " ".join(decoded.split())


def simplify_text(complex_text: str) -> str:
    global simplifier
    if "simplifier" not in globals():
        simplifier = TextSimplifier(device="cpu")
    return simplifier.simplify_text(complex_text)
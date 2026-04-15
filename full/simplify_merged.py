"""Text simplification using a locally merged full model (base + adapter)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class MergedTextSimplifier:
    """Simplifies complex English text using a local merged model folder."""

    def __init__(self, device: str = "cpu"):
        self.requested_device = device
        self.model_path = self._resolve_model_path()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Merged model folder not found: {self.model_path}")

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model, self.tokenizer, self.runtime = self._load_model_with_fallback()
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def _resolve_model_path(self) -> Path:
        env_path = os.getenv("SIMPLIFIER_MERGED_PATH")
        if env_path:
            return Path(env_path).resolve()

        here = Path(__file__).resolve().parent
        project_root = here.parent
        workspace_root = project_root.parent
        candidates = [
            project_root / "model" / "simplifier-merged",
            workspace_root / "Text-Simplification-and-Translation" / "model" / "simplifier-merged",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_model_with_fallback(self):
        errors = []

        loaders = (
            (self._load_cuda_offload, "cuda-offload"),
            (self._load_cpu, "cpu"),
        ) if self.device == "cuda" else ((self._load_cpu, "cpu"),)

        for loader, runtime in loaders:
            try:
                model, tokenizer = loader()
                return model, tokenizer, runtime
            except Exception as exc:
                errors.append(f"{runtime}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        raise RuntimeError("Failed loading merged simplifier: " + " | ".join(errors))

    def _load_cuda_offload(self):
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "5GiB", "cpu": "24GiB"},
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        return model, tokenizer

    def _load_cpu(self):
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        return model, tokenizer

    def get_runtime_info(self) -> dict:
        return {
            "model_source": str(self.model_path),
            "runtime": f"transformers-{self.runtime}",
            "requested_device": self.requested_device,
            "active_device": self.device,
        }

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _first_sentence(text: str) -> str:
        m = re.match(r"(.*?[.!?])(?:\s|$)", text.strip())
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _clean_sentence(text: str) -> str:
        s = " ".join((text or "").split()).strip()
        if not s:
            return s
        s = s[0].upper() + s[1:]
        if not s.endswith((".", "!", "?")):
            s += "."
        return s

    def _build_prompt(self, sentence: str) -> str:
        return (
            "Below is an instruction with an input. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "Rewrite the sentence in simpler English with shorter words. "
            "Keep meaning unchanged. Output only one sentence.\n\n"
            f"### Input:\n{sentence}\n\n"
            "### Response:\n"
        )

    def _generate_once(self, prompt: str, max_new_tokens: int = 64) -> str:
        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )

        target_device = getattr(self.model, "device", None)
        if target_device is not None and str(target_device) != "meta":
            model_inputs = {k: v.to(target_device) for k, v in model_inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        if "### Response:" in decoded:
            decoded = decoded.split("### Response:")[-1].strip()
        return self._first_sentence(decoded)

    def simplify_text(self, complex_text: str) -> str:
        sentences = self._split_sentences(complex_text)
        if not sentences:
            return complex_text

        results = []
        for sentence in sentences:
            prompt = self._build_prompt(sentence)
            out = self._generate_once(prompt)
            cleaned = self._clean_sentence(out)
            if not cleaned:
                cleaned = self._clean_sentence(sentence)
            results.append(cleaned)
        return " ".join(results)

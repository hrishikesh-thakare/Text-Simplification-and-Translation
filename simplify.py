"""Text simplification module using local LoRA adapter from training."""

from __future__ import annotations

import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import torch

try:
    from unsloth import FastLanguageModel  # type: ignore
except Exception:
    FastLanguageModel = None

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)


class TextSimplifier:
    """Simplifies complex English text using a local LoRA adapter."""

    def __init__(self, device="cpu"):
        self.requested_device = device
        self.local_files_only = os.getenv("APP_OFFLINE", "0") == "1"
        self.adapter_path = self._resolve_adapter_path()
        self.model_source = str(self.adapter_path)
        if not self.adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter directory not found: {self.adapter_path}"
            )

        self.base_model_name = self._read_base_model_name()
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        if self.device == "cuda":
            # Safe throughput optimization on Ampere+ GPUs without changing decoding params.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(
            "Loading Text Simplification adapter from "
            f"{self.adapter_path} (base: {self.base_model_name})..."
        )

        self.model, self.tokenizer, runtime = self._load_model_with_fallback()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.runtime = runtime
        print(f"✓ Adapter loaded successfully using runtime: {self.runtime}")

    def get_runtime_info(self) -> dict:
        return {
            "model_source": self.model_source,
            "base_model": self.base_model_name,
            "runtime": self.runtime,
            "requested_device": self.requested_device,
            "active_device": self.device,
            "offline_mode": self.local_files_only,
        }

    def _read_base_model_name(self) -> str:
        config_path = self.adapter_path / "adapter_config.json"
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("base_model_name_or_path", "unsloth/llama-3-8b-bnb-4bit")

    def _resolve_adapter_path(self) -> Path:
        env_path = os.getenv("SIMPLIFIER_ADAPTER_PATH")
        if env_path:
            return Path(env_path).resolve()

        here = Path(__file__).resolve().parent
        candidates = [
            here / "model" / "simplifier-4090",
            here.parent / "training" / "simplifier-4090",
            here / "training" / "simplifier-4090",
        ]

        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            bundle = Path(bundle_root)
            candidates.extend([
                bundle / "training" / "simplifier-4090",
                bundle / "simplifier-4090",
            ])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def _load_model_with_fallback(self):
        errors = []

        if self.device == "cuda":
            loaders = (
                (self._load_unsloth_cuda, "cuda-unsloth"),
                (self._load_transformers_cuda, "cuda-transformers"),
                (self._load_transformers_auto_offload, "auto-offload"),
                (self._load_transformers_full_precision_cpu, "cpu-full-precision"),
            )
        else:
            loaders = (
                (self._load_transformers_auto_offload, "auto-offload"),
                (self._load_transformers_full_precision_cpu, "cpu-full-precision"),
            )

        for loader, runtime_name in loaders:
            try:
                model, tokenizer = loader()
                return model, tokenizer, runtime_name
            except Exception as exc:
                errors.append(f"{runtime_name}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        message = " | ".join(errors) if errors else "unknown loading failure"
        raise RuntimeError(f"Failed to load simplifier adapter. Details: {message}")

    def _load_unsloth_cuda(self):
        if FastLanguageModel is None:
            raise ImportError("unsloth is not available")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.adapter_path),
            max_seq_length=256,
            dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
            max_memory={0: "5GiB", "cpu": "24GiB"},
            local_files_only=self.local_files_only,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def _load_tokenizer(self):
        """Load tokenizer robustly for locally exported adapters.

        Some adapter exports include tokenizer_class values that are not importable
        in plain Transformers environments (for example TokenizersBackend).
        In that case, we safely fallback to the base model tokenizer.
        """
        try:
            return AutoTokenizer.from_pretrained(
                str(self.adapter_path),
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            if "Tokenizer class" not in str(exc):
                raise
            try:
                return AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    trust_remote_code=True,
                    local_files_only=self.local_files_only,
                )
            except Exception:
                tokenizer_file = self.adapter_path / "tokenizer.json"
                config_path = self.adapter_path / "tokenizer_config.json"
                with config_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)

                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
                if cfg.get("bos_token"):
                    tokenizer.bos_token = cfg["bos_token"]
                if cfg.get("eos_token"):
                    tokenizer.eos_token = cfg["eos_token"]
                if cfg.get("pad_token"):
                    tokenizer.pad_token = cfg["pad_token"]
                if cfg.get("padding_side"):
                    tokenizer.padding_side = cfg["padding_side"]
                return tokenizer

    def _load_transformers_cuda(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = self._load_tokenizer()
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "5GiB", "cpu": "24GiB"},
            local_files_only=self.local_files_only,
        )
        model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        return model, tokenizer

    def _load_transformers_auto_offload(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = self._load_tokenizer()
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={"cpu": "24GiB"},
            local_files_only=self.local_files_only,
        )
        model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        return model, tokenizer

    def _load_transformers_full_precision_cpu(self):
        tokenizer = self._load_tokenizer()
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
        )
        model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        return model, tokenizer

    def _build_prompt(self, complex_text: str, strict: bool = True) -> str:
        return (
            "Simplify the sentence into easy English. Keep the meaning the same. "
            "Output only the simplified sentence. Do not explain.\n"
            f"Simplify: {complex_text}\n"
            "Simplified:"
        )

    def _extract_numbers(self, text: str):
        return set(re.findall(r"\d+(?:\.\d+)?%?", text))

    def _key_terms(self, text: str):
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
        stop = {
            "this", "that", "with", "from", "into", "onto", "over", "under",
            "about", "after", "before", "while", "where", "which", "their",
            "there", "these", "those", "were", "have", "has", "had", "been",
            "being", "will", "would", "could", "should", "than", "then", "when",
            "what", "your", "they", "them", "also", "only", "just", "very",
            "more", "most", "some", "such", "each", "many", "much", "because",
            "through", "between", "other", "into", "upon", "administration",
        }
        return {w for w in words if w not in stop}

    def _is_faithful(self, source: str, simplified: str) -> bool:
        if simplified.strip().lower() == source.strip().lower():
            return False

        similarity = SequenceMatcher(None, source.lower(), simplified.lower()).ratio()
        # Reject outputs that are effectively copies of the input.
        if similarity >= 0.92:
            return False

        src_nums = self._extract_numbers(source)
        out_nums = self._extract_numbers(simplified)
        # Never allow newly invented numbers/percentages.
        if not out_nums.issubset(src_nums):
            return False

        src_terms = self._key_terms(source)
        if not src_terms:
            return True
        out_terms = self._key_terms(simplified)
        missing_ratio = len(src_terms - out_terms) / max(len(src_terms), 1)
        # Allow some compression, but block severe information drop.
        return missing_ratio <= 0.55

    def _build_retry_prompt(self, complex_text: str) -> str:
        return (
            "Rewrite the sentence in simpler English using different words and shorter phrasing. "
            "Do not copy the original sentence. Keep the meaning exactly the same and do not add new facts.\n"
            f"Sentence: {complex_text}\n"
            "Simple:"
        )

    def _decode_output(self, outputs):
        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if "Simplified:" in decoded:
            decoded = decoded.split("Simplified:", 1)[1].strip()
        elif "Simple:" in decoded:
            decoded = decoded.split("Simple:", 1)[1].strip()
        return " ".join(decoded.split())

    def simplify_text(self, complex_text: str, max_length: int = 256) -> str:
        prompt = self._build_prompt(complex_text, strict=True)

        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=False,
        )

        # If model is sharded with accelerate/device_map, don't force tensors onto one device.
        target_device = getattr(self.model, "device", None)
        if target_device is not None and str(target_device) != "meta":
            model_inputs = {k: v.to(target_device) for k, v in model_inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=min(max_length, 64),
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                use_cache=True,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self._decode_output(outputs)


def simplify_text(complex_text: str) -> str:
    global simplifier
    if "simplifier" not in globals():
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        simplifier = TextSimplifier(device=default_device)
    return simplifier.simplify_text(complex_text)
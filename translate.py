"""
Translation Module using IndicTrans2.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import os

# IndicTrans2 tokenizer toolkit (migrated from old IndicTransTokenizer package).
try:
    from IndicTransToolkit import IndicProcessor
except Exception:
    class IndicProcessor:
        """Fallback processor when IndicTransToolkit is unavailable or incompatible.

        This keeps the app runnable in packaged/offline environments. The model
        still performs translation, but without the toolkit's extra preprocessing.
        """

        def __init__(self, inference=True):
            self.inference = inference

        def preprocess_batch(self, texts, src_lang=None, tgt_lang=None):
            return texts

        def postprocess_batch(self, texts, lang=None):
            return texts


class HindiTranslator:
    """Translates English text to Indic languages using IndicTrans2."""

    LANGUAGE_MAP = {
        "Hindi": "hin_Deva",
        "Bengali": "ben_Beng",
        "Gujarati": "guj_Gujr",
        "Kannada": "kan_Knda",
        "Malayalam": "mal_Mlym",
        "Marathi": "mar_Deva",
        "Odia": "ory_Orya",
        "Punjabi": "pan_Guru",
        "Tamil": "tam_Taml",
        "Telugu": "tel_Telu",
        "Urdu": "urd_Arab",
    }
    
    def __init__(self, device=None):
        """
        Initialize the IndicTrans2 translation model
        
        Args:
            device: 'cpu' or 'cuda' (defaults to 'cuda' if available)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        self.hf_token = os.getenv("HF_TOKEN")
        self.local_files_only = os.getenv("APP_OFFLINE", "0") == "1"
        
        print(f"Loading Translation Model from {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.hf_token,
                local_files_only=self.local_files_only,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                attn_implementation="eager",
                token=self.hf_token,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load IndicTrans2 model. Ensure access is approved on Hugging Face and run 'huggingface-cli login'."
            ) from exc

        self.model.to(self.device)
        self.model.config.use_cache = False
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = False
        self.model.eval()
        print(f"✓ Model loaded successfully on {self.device}")

        self.ip = IndicProcessor(inference=True)
        print("✓ IndicProcessor initialized")

    def _clean_output(self, text: str) -> str:
        """Normalize common spacing artifacts in generated text."""
        cleaned = text
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([।,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([(" + '"' + r"])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s+([)" + '"' + r"])", r"\1", cleaned)
        return cleaned.strip()

    def get_supported_languages(self):
        return list(self.LANGUAGE_MAP.keys())

    def translate(self, simplified_text: str, target_language: str = "Hindi") -> str:
        """Translate simplified English text to selected Indic language."""
        tgt_lang = self.LANGUAGE_MAP.get(target_language, "hin_Deva")

        batch = self.ip.preprocess_batch(
            [simplified_text],
            src_lang="eng_Latn",
            tgt_lang=tgt_lang,
        )

        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=1,
                use_cache=False,
                early_stopping=True,
            )

        generated = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        postprocessed = self.ip.postprocess_batch(generated, lang=tgt_lang)
        return self._clean_output(postprocessed[0])
    



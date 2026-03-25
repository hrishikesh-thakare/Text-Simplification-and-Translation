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
except ImportError:
    try:
        from IndicTransTokenizer import IndicProcessor
    except ImportError as exc:
        raise ImportError(
            "IndicProcessor is required for IndicTrans2. Install with: pip install IndicTransToolkit"
        ) from exc


class HindiTranslator:
    """Translates English text to Hindi using IndicTrans2."""
    
    def __init__(self, device='cpu'):
        """
        Initialize the IndicTrans2 translation model
        
        Args:
            device: 'cpu' or 'cuda' (we use CPU for broad compatibility)
        """
        self.device = device
        self.model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        self.hf_token = os.getenv("HF_TOKEN")
        
        print(f"Loading Translation Model from {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.hf_token,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.hf_token,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load IndicTrans2 model. Ensure access is approved on Hugging Face and run 'huggingface-cli login'."
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded successfully on {self.device}")

        self.ip = IndicProcessor(inference=True)
        print("✓ IndicProcessor initialized")

    def _clean_hindi_output(self, text: str) -> str:
        """Normalize common spacing artifacts in generated Hindi text."""
        cleaned = text
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([।,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([(" + '"' + r"])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s+([)" + '"' + r"])", r"\1", cleaned)
        return cleaned.strip()
    
    def translate_to_hindi(self, simplified_text: str) -> str:
        """
        Translate simplified English text to Hindi
        
        Args:
            simplified_text: Simplified English text to translate
        
        Returns:
            Hindi translation
        """
        
        # Preprocess with language tags/scripts expected by IndicTrans2.
        batch = self.ip.preprocess_batch(
            [simplified_text],
            src_lang="eng_Latn",
            tgt_lang="hin_Deva",
        )

        # Tokenize
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = inputs.to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode and postprocess back to readable Hindi.
        generated = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        postprocessed = self.ip.postprocess_batch(generated, lang="hin_Deva")
        hindi_text = postprocessed[0]

        return self._clean_hindi_output(hindi_text)


def translate_to_hindi(simplified_text: str) -> str:
    """
    Standalone function to translate text to Hindi
    Creates a global translator instance for reuse
    
    Args:
        simplified_text: Simplified English text to translate
    
    Returns:
        Hindi translation
    """
    global translator
    
    if 'translator' not in globals():
        translator = HindiTranslator(device='cpu')
    
    return translator.translate_to_hindi(simplified_text)


if __name__ == '__main__':
    # Test the translation module
    translator = HindiTranslator(device='cpu')
    
    test_texts = [
        "The government launched a new health program for citizens.",
        "Modern technology has changed how we communicate and work."
    ]
    
    print("\n" + "="*70)
    print("ENGLISH TO HINDI TRANSLATION TEST")
    print("="*70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nExample {i}:")
        print(f"English: {text}")
        hindi = translator.translate_to_hindi(text)
        print(f"Hindi:   {hindi}")

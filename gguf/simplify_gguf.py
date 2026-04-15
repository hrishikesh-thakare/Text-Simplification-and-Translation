"""Text simplification module backed by a merged GGUF model via llama.cpp."""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import torch

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


class GGUFTextSimplifier:
    """Simplifies complex English text using a local merged GGUF model."""

    def __init__(self, device: str = "cpu"):
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it first to use GGUF simplifier."
            )

        self.requested_device = device
        self.gguf_path = self._resolve_gguf_path()
        if not self.gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.gguf_path}")

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.n_ctx = int(os.getenv("SIMPLIFIER_GGUF_N_CTX", "2048"))
        self.n_batch = int(os.getenv("SIMPLIFIER_GGUF_N_BATCH", "256"))
        self.n_threads = int(
            os.getenv("SIMPLIFIER_GGUF_N_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
        )
        default_gpu_layers = "35" if self.device == "cuda" else "0"
        self.n_gpu_layers = int(os.getenv("SIMPLIFIER_GGUF_N_GPU_LAYERS", default_gpu_layers))

        self.llm = self._load_llm_with_fallback()

    def _resolve_gguf_path(self) -> Path:
        env_path = os.getenv("SIMPLIFIER_GGUF_PATH")
        if env_path:
            return Path(env_path).resolve()

        here = Path(__file__).resolve().parent
        project_root = here.parent
        workspace_root = project_root.parent
        candidates = [
            project_root / "full" / "simplifier-8b-q4.gguf",
            project_root / "simplifier-8b-q4.gguf",
            workspace_root / "simplifier-8b-q4.gguf",
            project_root / "model" / "simplifier-8b-q4.gguf",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_llm_with_fallback(self):
        try:
            return Llama(
                model_path=str(self.gguf_path),
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
        except Exception:
            # Retry fully on CPU so the app still starts if GPU offload is unavailable.
            self.n_gpu_layers = 0
            return Llama(
                model_path=str(self.gguf_path),
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                n_gpu_layers=0,
                verbose=False,
            )

    def get_runtime_info(self) -> dict:
        return {
            "model_source": str(self.gguf_path),
            "runtime": "llama.cpp-gguf",
            "requested_device": self.requested_device,
            "active_device": self.device,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_threads": self.n_threads,
        }

    @staticmethod
    def _first_sentence(text: str) -> str:
        m = re.match(r"(.*?[.!?])(?:\s|$)", text.strip())
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _extract_numbers(text: str):
        return set(re.findall(r"\d+(?:\.\d+)?%?", text))

    @staticmethod
    def _key_terms(text: str):
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
        stop = {
            "this", "that", "with", "from", "into", "onto", "over", "under",
            "about", "after", "before", "while", "where", "which", "their",
            "there", "these", "those", "were", "have", "has", "had", "been",
            "being", "will", "would", "could", "should", "than", "then", "when",
            "what", "your", "they", "them", "also", "only", "just", "very",
            "more", "most", "some", "such", "each", "many", "much", "because",
            "through", "between", "other", "upon", "administration",
        }
        return {w for w in words if w not in stop}

    @staticmethod
    def _token_words(text: str) -> list[str]:
        return re.findall(r"[A-Za-z]+", text)

    def _avg_word_len(self, text: str) -> float:
        words = self._token_words(text)
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    def _long_word_count(self, text: str, threshold: int = 9) -> int:
        return sum(1 for w in self._token_words(text) if len(w) >= threshold)

    def _is_mostly_copy(self, source: str, candidate: str) -> bool:
        if not candidate.strip():
            return True
        if candidate.strip().lower() == source.strip().lower():
            return True
        sim = SequenceMatcher(None, source.lower(), candidate.lower()).ratio()
        return sim >= 0.985

    def _is_simpler_surface(self, source: str, candidate: str) -> bool:
        source_words = max(1, len(self._token_words(source)))
        cand_words = max(1, len(self._token_words(candidate)))
        source_avg = self._avg_word_len(source)
        cand_avg = self._avg_word_len(candidate)
        source_long = self._long_word_count(source)
        cand_long = self._long_word_count(candidate)

        shorter_or_similar = cand_words <= int(source_words * 1.05)
        simpler_words = cand_avg <= source_avg - 0.25
        fewer_long_words = cand_long <= source_long - 1
        return shorter_or_similar and (simpler_words or fewer_long_words)

    def _heuristic_simplify(self, sentence: str) -> str:
        """Rule-based backup to avoid returning unchanged complex sentence."""
        replacements = {
            "rapid": "fast",
            "proliferation": "growth",
            "artificial intelligence": "AI",
            "technologies": "tools",
            "multifarious": "many",
            "precipitated": "caused",
            "paradigm shift": "big change",
            "manner": "way",
            "processed": "handled",
            "analyzed": "studied",
            "leveraged": "used",
            "decision-making": "making decisions",
        }

        s = sentence
        for old, new in sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True):
            s = re.sub(rf"\b{re.escape(old)}\b", new, s, flags=re.IGNORECASE)

        # Keep first clause to avoid long, dense run-ons in fallback mode.
        parts = re.split(r"[,;:]", s)
        if parts and len(parts[0].split()) >= 6:
            s = parts[0].strip()

        return self._clean_sentence(s)

    def _is_faithful(self, source: str, simplified: str) -> bool:
        if not simplified.strip():
            return False

        if simplified.strip().lower() == source.strip().lower():
            return False

        similarity = SequenceMatcher(None, source.lower(), simplified.lower()).ratio()
        if similarity >= 0.97:
            return False

        src_nums = self._extract_numbers(source)
        out_nums = self._extract_numbers(simplified)
        if not out_nums.issubset(src_nums):
            return False

        src_terms = self._key_terms(source)
        if not src_terms:
            return True

        out_terms = self._key_terms(simplified)
        missing_ratio = len(src_terms - out_terms) / max(len(src_terms), 1)
        if missing_ratio > 0.60:
            return False

        added_terms = out_terms - src_terms
        if len(added_terms) > max(4, int(len(src_terms) * 0.35)):
            return False

        return True

    @staticmethod
    def _clean_sentence(sentence: str) -> str:
        s = " ".join(sentence.split()).strip()
        if not s:
            return s
        s = s[0].upper() + s[1:]
        if not s.endswith((".", "!", "?")):
            s += "."
        return s

    def _build_prompt(self, text: str) -> str:
        return f"""
Simplify the sentence for a 10-year-old child.

Rules:
- Use very simple words
- Break long sentences
- Remove difficult words
- Keep the same meaning
- DO NOT copy the original sentence

Example:
Input: The rapid proliferation of technology has transformed communication.
Output: Technology is growing fast and has changed how people talk.

Now simplify:

Input: {text}
Output:
"""

    def _generate_once(self, prompt: str, max_tokens: int = 64) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.15,
            stop=["###", "\n\n"],
        )
        text = (output["choices"][0].get("text") or "").strip()
        return self._first_sentence(text)

    def _simplify_sentence(self, sentence: str) -> str:
        prompt = self._build_prompt(sentence)
        first_try = self._generate_once(prompt)
        if self._is_faithful(sentence, first_try):
            return self._clean_sentence(first_try)

        retry_prompt = f"""
Simplify aggressively.

- Use basic English
- Max 12-15 words
- No complex words
- No copying

Sentence: {sentence}
Simple:
"""
        second_try = self._generate_once(retry_prompt)
        if self._is_faithful(sentence, second_try):
            return self._clean_sentence(second_try)

        third_prompt = f"""
Rewrite this sentence in very plain English for a child.

Rules:
- Keep meaning
- Use common words only
- Keep it short
- Do not copy original wording

Sentence: {sentence}
Plain English:
"""
        third_try = self._generate_once(third_prompt, max_tokens=48)
        if self._is_faithful(sentence, third_try):
            return self._clean_sentence(third_try)

        candidates = [first_try, second_try, third_try]
        candidates = [self._clean_sentence(c) for c in candidates if isinstance(c, str) and c.strip()]
        non_copy_candidates = [
            c for c in candidates
            if not self._is_mostly_copy(sentence, c) and self._is_simpler_surface(sentence, c)
        ]
        if non_copy_candidates:
            return min(non_copy_candidates, key=len)

        # Last-resort deterministic simplification instead of returning unchanged text.
        return self._heuristic_simplify(sentence)

    def simplify_text(self, complex_text: str) -> str:
        sentences = self._split_sentences(complex_text)
        if not sentences:
            return complex_text
        return " ".join(self._simplify_sentence(s) for s in sentences)

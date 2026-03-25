"""
Vernacular Language Text Simplifier using IndicTrans Hosted API
"""

import requests
import json
from typing import List, Dict

class IndicTransAPI:
    """Interface to IndicTrans hosted API"""
    
    BASE_URL = "https://api.ai4bharat.org/v1/translate"
    
    LANGUAGE_CODES = {
        'assamese': 'as',
        'bengali': 'bn',
        'gujarati': 'gu',
        'hindi': 'hi',
        'kannada': 'kn',
        'malayalam': 'ml',
        'marathi': 'mr',
        'odia': 'or',
        'punjabi': 'pa',
        'tamil': 'ta',
        'telugu': 'te',
    }
    
    def __init__(self):
        self.session = requests.Session()
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate vernacular text to English"""
        
        lang_code = self.LANGUAGE_CODES.get(source_lang.lower())
        if not lang_code:
            raise ValueError(f"Unsupported language: {source_lang}")
        
        try:
            payload = {
                'text': text,
                'source': lang_code,
                'target': 'en'
            }
            
            response = self.session.post(self.BASE_URL, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get('translatedText', text)
            
        except requests.RequestException as e:
            print(f"Translation API error: {e}")
            return text
    
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate English text to vernacular"""
        
        lang_code = self.LANGUAGE_CODES.get(target_lang.lower())
        if not lang_code:
            raise ValueError(f"Unsupported language: {target_lang}")
        
        try:
            payload = {
                'text': text,
                'source': 'en',
                'target': lang_code
            }
            
            response = self.session.post(self.BASE_URL, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get('translatedText', text)
            
        except requests.RequestException as e:
            print(f"Translation API error: {e}")
            return text


class TextSimplifier:
    """Simplifies text using translation and basic NLP techniques"""
    
    def __init__(self):
        self.api = IndicTransAPI()
    
    def simplify(self, text: str, source_lang: str) -> Dict[str, str]:
        """
        Simplify vernacular text by translating to English and back
        
        Args:
            text: Input text in vernacular language
            source_lang: Source language name (e.g., 'hindi', 'tamil')
        
        Returns:
            Dictionary with original and simplified text
        """
        
        # Step 1: Translate to English
        english_text = self.api.translate_to_english(text, source_lang)
        
        # Step 2: Apply simplification rules
        simplified_english = self._simplify_english(english_text)
        
        # Step 3: Translate back to original language
        simplified_vernacular = self.api.translate_from_english(
            simplified_english, 
            source_lang
        )
        
        return {
            'original': text,
            'original_language': source_lang,
            'english': english_text,
            'simplified_english': simplified_english,
            'simplified_vernacular': simplified_vernacular
        }
    
    def _simplify_english(self, text: str) -> str:
        """Apply basic English simplification rules"""
        
        # Replace complex words with simpler alternatives
        replacements = {
            'utilize': 'use',
            'endeavor': 'try',
            'facilitate': 'help',
            'ameliorate': 'improve',
            'terminate': 'end',
            'commence': 'start',
            'consequently': 'so',
            'notwithstanding': 'however',
            'furthermore': 'also',
            'nevertheless': 'but',
        }
        
        simplified = text
        for complex_word, simple_word in replacements.items():
            simplified = simplified.replace(complex_word, simple_word)
            simplified = simplified.replace(complex_word.capitalize(), simple_word.capitalize())
        
        return simplified


def main():
    """Example usage"""
    
    simplifier = TextSimplifier()
    
    # Example: Hindi text
    hindi_text = "सरकार ने नई नीति को लागू करने का निर्णय लिया है।"
    
    print("=" * 60)
    print("TEXT SIMPLIFICATION EXAMPLE")
    print("=" * 60)
    print(f"\nOriginal (Hindi):\n{hindi_text}\n")
    
    result = simplifier.simplify(hindi_text, 'hindi')
    
    print(f"Translated to English:\n{result['english']}\n")
    print(f"Simplified English:\n{result['simplified_english']}\n")
    print(f"Simplified Vernacular:\n{result['simplified_vernacular']}\n")


if __name__ == '__main__':
    main()

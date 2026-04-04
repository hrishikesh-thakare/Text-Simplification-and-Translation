"""
Main Pipeline: Complex English → Simplified English → Hindi Translation
Full end-to-end multilingual text simplification and translation system
"""

from simplify import TextSimplifier
from translate import HindiTranslator
import torch


class TextSimplificationPipeline:
    """Complete pipeline for text simplification and translation"""
    
    def __init__(self, device='cpu'):
        """
        Initialize both models
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        print("\n" + "="*70)
        print("INITIALIZING TEXT SIMPLIFICATION PIPELINE")
        print("="*70 + "\n")
        
        self.simplifier = TextSimplifier(device=device)
        print()
        self.translator = HindiTranslator(device=device)
    
    def run_pipeline(self, complex_text: str) -> dict:
        """
        Run the complete pipeline: Simplify → Translate
        
        Args:
            complex_text: Complex English text to process
        
        Returns:
            Dictionary with all pipeline outputs
        """
        
        print("\n" + "-"*70)
        print("PROCESSING TEXT")
        print("-"*70)
        
        # Step 1: Simplify
        print("\n[1/3] Simplifying complex English text...")
        simplified_text = self.simplifier.simplify_text(complex_text)
        print(f"✓ Simplification complete")
        
        # Step 2: Translate to Hindi
        print("\n[2/3] Translating simplified English to Hindi...")
        hindi_text = self.translator.translate_to_hindi(simplified_text)
        print(f"✓ Translation complete")
        
        # Step 3: Create results
        print("\n[3/3] Pipeline complete!")
        
        results = {
            'original_complex': complex_text,
            'simplified_english': simplified_text,
            'hindi_translation': hindi_text
        }
        
        return results
    
    def display_results(self, results: dict):
        """Pretty print the pipeline results"""
        
        print("\n" + "="*70)
        print("PIPELINE RESULTS")
        print("="*70)
        
        print(f"\n📌 ORIGINAL COMPLEX TEXT:")
        print(f"   {results['original_complex']}")
        
        print(f"\n✓ SIMPLIFIED ENGLISH:")
        print(f"   {results['simplified_english']}")
        
        print(f"\n🇮🇳 HINDI TRANSLATION:")
        print(f"   {results['hindi_translation']}")
        
        print("\n" + "="*70 + "\n")


def main():
    """Main execution function"""
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Selected runtime device preference: {device}")
    pipeline = TextSimplificationPipeline(device=device)
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Type your complex English text and press Enter.")
    print("Type 'quit' to stop. Type 'examples' to run built-in demo texts.\n")

    while True:
        user_text = input("Enter complex English text: ").strip()

        if not user_text:
            print("Please enter some text.\n")
            continue

        if user_text.lower() in {"quit", "exit", "q"}:
            print("\nExiting. Goodbye!")
            break

        if user_text.lower() == "examples":
            medical_text = (
                "The administration of comprehensive pharmacological interventions necessitates "
                "meticulous monitoring of patient vitals and laboratory parameters to obviate "
                "potential contraindications and adverse pharmaceutical interactions."
            )
            government_text = (
                "The promulgation of legislative provisions pertaining to fiscal accountability "
                "and governmental transparency endeavors to ameliorate bureaucratic efficacy "
                "and foster unprecedented public confidence in institutional mechanisms."
            )

            print("\n" + "#"*70)
            print("EXAMPLE 1: MEDICAL TEXT")
            print("#"*70)
            example_1 = pipeline.run_pipeline(medical_text)
            pipeline.display_results(example_1)

            print("\n" + "#"*70)
            print("EXAMPLE 2: GOVERNMENT POLICY TEXT")
            print("#"*70)
            example_2 = pipeline.run_pipeline(government_text)
            pipeline.display_results(example_2)
            continue

        results = pipeline.run_pipeline(user_text)
        pipeline.display_results(results)


if __name__ == '__main__':
    main()

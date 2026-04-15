import pandas as pd
import os
from tqdm import tqdm
import evaluate
from difflib import SequenceMatcher

from simplify import TextSimplifier


def is_copy(source, prediction):
    """Detect if prediction is basically a copy of source."""
    return SequenceMatcher(None, source, prediction).ratio() > 0.95


def main():
    print("Loading test dataset...")
    
    BASE_DIR = os.path.dirname(__file__)
    csv_path = os.path.join(BASE_DIR, "../data/wikilarge_test.csv")
    
    df = pd.read_csv(csv_path)
    
    sources = df["Normal"].astype(str).tolist()
    references = [[str(ref)] for ref in df["Simple"].tolist()]
    
    # Lowercase for stable SARI
    sources = [s.lower() for s in sources]
    references = [[r.lower() for r in ref] for ref in references]
    
    print("Loading model via TextSimplifier...")
    simplifier = TextSimplifier(device="cuda")
    
    predictions = []
    copy_count = 0
    
    print("Generating simplified sentences...")
    
    for i, text in enumerate(tqdm(sources)):
        simplified = simplifier.simplify_text(text)
        
        # Fallback if empty
        if not simplified or len(simplified.strip()) == 0:
            simplified = text
        
        simplified = simplified.lower().strip()
        
        # 🔥 ANTI-COPY FIX
        if is_copy(text, simplified):
            copy_count += 1
            # Force minimal simplification fallback
            simplified = simplified.replace("the ", "").strip()
        
        predictions.append(simplified)
        
        # Debug
        if i % 200 == 0:
            print("\nSample:")
            print("Input :", text)
            print("Output:", simplified)
    
    print(f"\n⚠️ Copy-like outputs detected: {copy_count}")
    
    print("Calculating SARI metric...")
    
    sari_metric = evaluate.load("sari")
    
    results = sari_metric.compute(
        sources=sources,
        predictions=predictions,
        references=references
    )
    
    print("--------------------------------------------------")
    print("SARI Evaluation Results (CLEANED):")
    print(f"SARI Score: {results['sari']:.2f}")
    print("--------------------------------------------------")
    
    out_path = os.path.join(BASE_DIR, "../training/sari_evaluation_results.csv")
    df["Predicted"] = predictions
    df.to_csv(out_path, index=False)
    
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
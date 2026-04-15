import pandas as pd
import os
import textstat
from bert_score import BERTScorer
from difflib import SequenceMatcher
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# -----------------------------
# Extra Metrics
# -----------------------------

def copy_rate(sources, predictions):
    count = 0
    for s, p in zip(sources, predictions):
        if SequenceMatcher(None, s, p).ratio() > 0.95:
            count += 1
    return count / len(sources)

def length_ratio(sources, predictions):
    ratios = []
    for s, p in zip(sources, predictions):
        s_len = max(len(s.split()), 1)
        p_len = len(p.split())
        ratios.append(p_len / s_len)
    return sum(ratios) / len(ratios)

# -----------------------------
# Main
# -----------------------------

def main():
    print("Loading sari_evaluation_results.csv...")
    
    BASE_DIR = os.path.dirname(__file__)
    csv_path = os.path.join(BASE_DIR, "../training/sari_evaluation_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    predictions = df["Predicted"].astype(str).tolist()
    sources = df["Normal"].astype(str).tolist()
    references = df["Simple"].astype(str).tolist()
    
    print(f"Evaluating {len(predictions)} sentences...")
    
    # FKGL
    print("\nCalculating FKGL (Readability)...")
    source_fkgl = sum(textstat.flesch_kincaid_grade(text) for text in sources) / len(sources)
    pred_fkgl   = sum(textstat.flesch_kincaid_grade(text) for text in predictions) / len(predictions)
    
    print("--------------------------------------------------")
    print("FKGL Results:")
    print(f"Source Grade Level:    {source_fkgl:.2f}")
    print(f"Predicted Grade Level: {pred_fkgl:.2f} (lower = simpler)")
    print("--------------------------------------------------")

    # BERTScore
    print("\nCalculating BERTScore (Meaning)...")
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    _, _, F1_ref = scorer.score(predictions, references)
    _, _, F1_src = scorer.score(predictions, sources)
    
    print("--------------------------------------------------")
    print("BERTScore Results:")
    print(f"F1 vs Reference: {F1_ref.mean().item():.4f}")
    print(f"F1 vs Source   : {F1_src.mean().item():.4f}")
    print("--------------------------------------------------")

    # Copy Rate
    cr = copy_rate(sources, predictions)
    print(f"\nCopy Rate: {cr*100:.2f}% of outputs are near copies")

    # Length Ratio
    lr = length_ratio(sources, predictions)
    print(f"Length Ratio: {lr:.2f} (target: 0.7–0.9)")

    # Final Interpretation
    print("\n--------------------------------------------------")
    print("FINAL INTERPRETATION:")
    if cr > 0.4:
        print("[FAIL] Model is copying too much")
    elif lr > 0.95:
        print("[WARN] Not simplifying enough (length ratio too high)")
    elif pred_fkgl >= source_fkgl:
        print("[FAIL] Not reducing complexity")
    elif F1_src.mean().item() < 0.85:
        print("[FAIL] Losing too much meaning")
    else:
        print("[PASS] Good simplification system")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()

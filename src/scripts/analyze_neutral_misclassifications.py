import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
PRED_FILE = OUTPUT_DIR / "predicted_data_multi.csv"

def top_key_terms(target_texts, baseline_texts, top_n=25):
    """
    Returns terms overrepresented in target_texts vs baseline_texts
    using smoothed log-probability difference.
    """
    if len(target_texts) == 0 or len(baseline_texts) == 0:
        return pd.DataFrame(columns=["term", "score"])

    vec = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_features=6000
    )

    combined = pd.concat([target_texts, baseline_texts], ignore_index=True)
    X = vec.fit_transform(combined)

    X_target = X[: len(target_texts)]
    X_base = X[len(target_texts):]

    t_counts = np.asarray(X_target.sum(axis=0)).ravel() + 1.0
    b_counts = np.asarray(X_base.sum(axis=0)).ravel() + 1.0

    t_prob = t_counts / t_counts.sum()
    b_prob = b_counts / b_counts.sum()

    scores = np.log(t_prob) - np.log(b_prob)
    terms = np.array(vec.get_feature_names_out())

    idx = np.argsort(scores)[::-1][:top_n]
    return pd.DataFrame({"term": terms[idx], "score": scores[idx]})

def main():
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PRED_FILE}")

    df = pd.read_csv(PRED_FILE)

    required_cols = {"text", "actual_sentiment", "predicted_sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    neutral_all = df[df["actual_sentiment"] == "neutral"].copy()
    neutral_correct = neutral_all[neutral_all["predicted_sentiment"] == "neutral"].copy()
    neutral_mis = neutral_all[neutral_all["predicted_sentiment"] != "neutral"].copy()

    neutral_to_pos = neutral_mis[neutral_mis["predicted_sentiment"] == "positive"].copy()
    neutral_to_neg = neutral_mis[neutral_mis["predicted_sentiment"] == "negative"].copy()

    # Save slices for manual inspection
    neutral_correct.to_csv(OUTPUT_DIR / "neutral_correct_predictions.csv", index=False)
    neutral_mis.to_csv(OUTPUT_DIR / "neutral_misclassified_reviews.csv", index=False)
    neutral_to_pos.to_csv(OUTPUT_DIR / "neutral_to_positive_reviews.csv", index=False)
    neutral_to_neg.to_csv(OUTPUT_DIR / "neutral_to_negative_reviews.csv", index=False)

    # Summary
    summary = {
        "neutral_total": len(neutral_all),
        "neutral_correct": len(neutral_correct),
        "neutral_misclassified": len(neutral_mis),
        "neutral_recall": (len(neutral_correct) / len(neutral_all)) if len(neutral_all) else np.nan,
        "neutral_to_positive": len(neutral_to_pos),
        "neutral_to_negative": len(neutral_to_neg),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_DIR / "neutral_error_summary.csv", index=False)

    # Term analysis: what language pushes neutral toward positive/negative?
    pos_terms = top_key_terms(neutral_to_pos["text"].fillna(""), neutral_correct["text"].fillna(""), top_n=30)
    neg_terms = top_key_terms(neutral_to_neg["text"].fillna(""), neutral_correct["text"].fillna(""), top_n=30)

    pos_terms.to_csv(OUTPUT_DIR / "neutral_to_positive_top_terms.csv", index=False)
    neg_terms.to_csv(OUTPUT_DIR / "neutral_to_negative_top_terms.csv", index=False)

    # Plot misclassification breakdown
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=neutral_mis,
        x="predicted_sentiment",
        order=["negative", "positive"],
        palette="Set2"
    )
    plt.title("Neutral Reviews Misclassified As")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "neutral_misclassification_breakdown.png", dpi=300)
    plt.close()

    # Confidence plot if available
    if "confidence" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(neutral_mis["confidence"], bins=20, kde=True, color="tomato")
        plt.title("Confidence Distribution: Misclassified Neutral Reviews")
        plt.xlabel("Model Confidence")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "neutral_misclassified_confidence_hist.png", dpi=300)
        plt.close()

    print("Neutral misclassification analysis complete.")
    print(f"Saved outputs to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
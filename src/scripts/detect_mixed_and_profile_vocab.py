import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_FILE = OUTPUT_DIR / "predicted_data_multi.csv"
OUTPUT_FILE = OUTPUT_DIR / "predicted_data_with_mixed_flag.csv"

# Tunable thresholds
POS_MIN = 0.30
NEG_MIN = 0.30
DIFF_MAX = 0.25

CONTRAST_WORDS = {
    "but", "however", "though", "although", "yet", "except", "overall", "while"
}

POS_CUES = {
    "good", "great", "nice", "friendly", "fast", "clean", "love", "excellent", "amazing", "enjoy"
}
NEG_CUES = {
    "bad", "slow", "rude", "wrong", "dirty", "hate", "awful", "terrible", "issue", "problem"
}

def has_contrast(text: str) -> bool:
    t = f" {text.lower()} "
    return any(f" {w} " in t for w in CONTRAST_WORDS)

def has_dual_polarity_words(text: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return (len(tokens & POS_CUES) > 0) and (len(tokens & NEG_CUES) > 0)

def mixed_rule(row) -> bool:
    text = str(row.get("text", ""))

    # Probability condition (if available)
    prob_cond = False
    if {"prob_positive", "prob_negative"}.issubset(row.index):
        p_pos = float(row.get("prob_positive", 0.0))
        p_neg = float(row.get("prob_negative", 0.0))
        prob_cond = (p_pos >= POS_MIN) and (p_neg >= NEG_MIN) and (abs(p_pos - p_neg) <= DIFF_MAX)

    contrast_cond = has_contrast(text)
    lex_cond = has_dual_polarity_words(text)

    # Strong rule: probabilities + contrast, or fallback lexical mixed signal
    return (prob_cond and contrast_cond) or (contrast_cond and lex_cond)

def top_terms(target_texts: pd.Series, baseline_texts: pd.Series, top_n=30):
    if len(target_texts) == 0 or len(baseline_texts) == 0:
        return pd.DataFrame(columns=["term", "score"])

    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=8000)
    all_text = pd.concat([target_texts.fillna(""), baseline_texts.fillna("")], ignore_index=True)
    X = vec.fit_transform(all_text)

    Xt = X[:len(target_texts)]
    Xb = X[len(target_texts):]

    t_counts = np.asarray(Xt.sum(axis=0)).ravel() + 1.0
    b_counts = np.asarray(Xb.sum(axis=0)).ravel() + 1.0

    t_prob = t_counts / t_counts.sum()
    b_prob = b_counts / b_counts.sum()
    scores = np.log(t_prob) - np.log(b_prob)

    terms = np.array(vec.get_feature_names_out())
    idx = np.argsort(scores)[::-1][:top_n]
    return pd.DataFrame({"term": terms[idx], "score": scores[idx]})

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"text", "actual_sentiment", "predicted_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Identify middle class rows from actual labels (handles both names)
    middle_labels = {"neutral", "neutral/mixed"}
    middle = df[df["actual_sentiment"].astype(str).str.lower().isin(middle_labels)].copy()

    middle["mixed_detected"] = middle.apply(mixed_rule, axis=1)
    middle["middle_subtype"] = np.where(middle["mixed_detected"], "mixed_candidate", "neutral_candidate")

    # Merge flag back
    out = df.copy()
    out["mixed_detected"] = False
    out["middle_subtype"] = ""
    out.loc[middle.index, "mixed_detected"] = middle["mixed_detected"].values
    out.loc[middle.index, "middle_subtype"] = middle["middle_subtype"].values
    out.to_csv(OUTPUT_FILE, index=False)

    # Summary
    summary = pd.DataFrame([{
        "middle_total": len(middle),
        "mixed_candidate": int((middle["mixed_detected"] == True).sum()),
        "neutral_candidate": int((middle["mixed_detected"] == False).sum()),
        "mixed_rate": float((middle["mixed_detected"] == True).mean()) if len(middle) else np.nan
    }])
    summary.to_csv(OUTPUT_DIR / "neutral_mixed_split_summary.csv", index=False)

    # Vocab profiles
    mixed_text = middle.loc[middle["mixed_detected"], "text"].fillna("")
    neutral_text = middle.loc[~middle["mixed_detected"], "text"].fillna("")

    top_terms(mixed_text, neutral_text, top_n=40).to_csv(
        OUTPUT_DIR / "mixed_candidate_top_terms.csv", index=False
    )
    top_terms(neutral_text, mixed_text, top_n=40).to_csv(
        OUTPUT_DIR / "neutral_candidate_top_terms.csv", index=False
    )

    # Save review slices for manual spot-check
    middle.loc[middle["mixed_detected"]].to_csv(
        OUTPUT_DIR / "mixed_candidate_reviews.csv", index=False
    )
    middle.loc[~middle["mixed_detected"]].to_csv(
        OUTPUT_DIR / "neutral_candidate_reviews.csv", index=False
    )

    print("Done.")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved summary/term files in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
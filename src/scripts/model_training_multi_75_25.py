"""
Sentiment Analysis Model Training Script — 75/25 Split

This script trains a machine learning model to predict sentiment (positive/negative/neutral)
from customer reviews based on their text content and star ratings.

This version uses a 75/25 train-test split instead of 80/20, providing a larger
test set for more robust evaluation at the cost of slightly less training data.

Model Type: Logistic Regression with TF-IDF Vectorization
Split: 75% Training / 25% Testing
"""

import sys
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, log_loss
)
from sklearn.feature_extraction import text as sk_text
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR     = PROJECT_ROOT / 'src' / 'sample_data'
OUTPUT_DIR   = PROJECT_ROOT / 'output'

# ── Mixed-signal detection helpers ────────────────────────────────────────────
CONTRAST_WORDS = {"but", "however", "though", "although", "yet", "except", "overall", "while"}
POS_CUES = {"good", "great", "nice", "friendly", "fast", "clean", "love",
            "excellent", "amazing", "enjoy", "helpful", "professional"}
NEG_CUES = {"bad", "slow", "rude", "wrong", "dirty", "hate", "awful",
            "terrible", "issue", "problem"}

def has_contrast(text: str) -> bool:
    t = f" {str(text).lower()} "
    return any(f" {w} " in t for w in CONTRAST_WORDS)

def has_dual_polarity_words(text: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", str(text).lower()))
    return bool(tokens & POS_CUES) and bool(tokens & NEG_CUES)

def mixed_rule(row) -> bool:
    text  = str(row.get("text", ""))
    p_pos = float(row.get("prob_positive", 0.0))
    p_neg = float(row.get("prob_negative", 0.0))
    prob_cond     = (p_pos >= 0.30) and (p_neg >= 0.30) and (abs(p_pos - p_neg) <= 0.25)
    contrast_cond = has_contrast(text)
    lex_cond      = has_dual_polarity_words(text)
    return (prob_cond and contrast_cond) or (contrast_cond and lex_cond)

# ── Sentiment labelling ────────────────────────────────────────────────────────
def sentiments_from_stars(stars, classification_type='three_class'):
    """Convert star ratings to sentiment labels."""
    if classification_type == 'binary':
        if stars >= 4:   return 'positive'
        elif stars <= 2: return 'negative'
        else:            return None
    else:
        if stars >= 4:   return 'positive'
        elif stars == 3: return 'neutral/mixed'
        else:            return 'negative'

def add_sentiment_values_to_file():
    """Add sentiment labels to training data file."""
    file = DATA_DIR / 'training_testing_data.csv'
    df   = pd.read_csv(file)
    df['sentiment'] = df['stars'].apply(
        lambda x: sentiments_from_stars(x, 'three_class'))
    df.to_csv(DATA_DIR / 'training_testing_data.csv', index=False)
    print("Sentiment labels added to training data.")

# ── Main training pipeline ─────────────────────────────────────────────────────
def main():
    """
    75/25 Train-Test Split Training Pipeline:
    1. Load preprocessed data with sentiment labels
    2. Split data 75% training / 25% testing  ← KEY DIFFERENCE from 80/20
    3. Vectorize text using TF-IDF
    4. Train Logistic Regression classifier
    5. Run 5-Fold Stratified Cross-Validation for robust evaluation
    6. Evaluate on held-out test set
    7. Save model, metrics, charts, and log
    """

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    file = DATA_DIR / 'training_testing_data.csv'
    df   = pd.read_csv(file)
    df   = df[df['sentiment'].notna()].copy()

    content = df['clean_text']
    sent    = df['sentiment']

    print(f"Total samples: {len(df)}")
    print(f"Sentiment distribution:\n{sent.value_counts()}\n")

    # ── Step 2: 75/25 Train-Test Split ────────────────────────────────────────
    # Using test_size=0.25 instead of 0.20
    # Larger test set = more reliable evaluation metrics
    # Tradeoff: 5% less training data compared to 80/20
    content_train, content_test, sent_train, sent_test = train_test_split(
        content,
        sent,
        test_size=0.25,       # 25% for testing (vs 20% in the 80/20 version)
        random_state=2016,    # Same seed for fair comparison with 80/20 results
        stratify=sent         # Maintain class proportions in both sets
    )

    print(f"Train set: {len(content_train)} samples ({len(content_train)/len(df)*100:.0f}%)")
    print(f"Test  set: {len(content_test)}  samples ({len(content_test)/len(df)*100:.0f}%)")
    print(f"Train sentiment distribution:\n{sent_train.value_counts()}")
    print(f"Test  sentiment distribution:\n{sent_test.value_counts()}\n")

    # ── Step 3: TF-IDF Vectorization ──────────────────────────────────────────
    extra_stop = {'review', 'user', 'star', 'stars', 'https', 'http', 'amp'}
    stop_words = set(sk_text.ENGLISH_STOP_WORDS) | extra_stop

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words=list(stop_words)
    )

    content_train_tfidf = vectorizer.fit_transform(content_train)
    content_test_tfidf  = vectorizer.transform(content_test)

    print(f"TF-IDF feature matrix: {content_train_tfidf.shape}")

    # ── Step 4: Train Logistic Regression ─────────────────────────────────────
    model = LogisticRegression(
        max_iter=1000,
        random_state=2016,
        C=0.8,
        class_weight='balanced'
    )
    model.fit(content_train_tfidf, sent_train)
    print("Model training complete.\n")

    # ── Step 5: 5-Fold Stratified Cross-Validation ────────────────────────────
    # This gives a more reliable accuracy estimate than a single split
    # Uses the full dataset (not just training set) for cross-validation
    print("Running 5-Fold Stratified Cross-Validation...")
    all_tfidf = vectorizer.transform(content)   # transform full dataset
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=2016)
    cv_scores = cross_val_score(model, all_tfidf, sent, cv=cv, scoring='accuracy')
    cv_f1     = cross_val_score(model, all_tfidf, sent, cv=cv, scoring='f1_macro')

    print(f"CV Accuracy:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"CV Macro F1:  {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})\n")

    # ── Step 6: Evaluate on held-out test set ─────────────────────────────────
    sent_predict = model.predict(content_test_tfidf)
    sent_proba   = model.predict_proba(content_test_tfidf)
    confidence   = sent_proba.max(axis=1)

    accuracy        = accuracy_score(sent_test, sent_predict)
    balanced_acc    = balanced_accuracy_score(sent_test, sent_predict)
    macro_f1        = f1_score(sent_test, sent_predict, average='macro')
    weighted_f1     = f1_score(sent_test, sent_predict, average='weighted')
    model_log_loss  = log_loss(sent_test, sent_proba, labels=model.classes_)

    cm = confusion_matrix(sent_test, sent_predict, labels=model.classes_)

    print(f"Test Accuracy:          {accuracy:.4f}")
    print(f"Test Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Test Macro F1:          {macro_f1:.4f}")
    print(f"Test Weighted F1:       {weighted_f1:.4f}")
    print(f"Test Log Loss:          {model_log_loss:.4f}\n")

    # ── Step 7: Save confusion matrix chart ───────────────────────────────────
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title('Confusion Matrix — Logistic Regression (75/25 Split)')
    plt.ylabel('Actual Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_multi_75_25.png', dpi=300)
    plt.close()
    print("Confusion matrix saved to confusion_matrix_multi_75_25.png")

    # ── Step 8: Feature importance chart ──────────────────────────────────────
    feature_names = vectorizer.get_feature_names_out()
    coefficients  = model.coef_

    vocab_data = []
    for idx, feature in enumerate(feature_names):
        tfidf_value    = content_train_tfidf.mean(axis=0).A1[idx]
        feature_coefs  = coefficients[:, idx]
        max_sent_idx   = np.argmax(np.abs(feature_coefs))
        primary_sent   = model.classes_[max_sent_idx]
        coef_dict      = {f'{s}_coef': feature_coefs[i]
                          for i, s in enumerate(model.classes_)}
        vocab_data.append({
            'Word/Phrase':          feature,
            'TF-IDF_Average':       tfidf_value,
            'Primary_Sentiment':    primary_sent,
            'Sentiment_Coefficient': feature_coefs[max_sent_idx],
            **coef_dict
        })

    vocab_df = pd.DataFrame(vocab_data).sort_values('TF-IDF_Average', ascending=False)
    vocab_df.to_csv(OUTPUT_DIR / 'vocab_data_multi_75_25.csv', index=False)
    print("Vocabulary data saved to vocab_data_multi_75_25.csv")

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    base_colors = {
        'positive':     (0.0, 0.6, 0.0),
        'neutral/mixed':(0.5, 0.5, 0.5),
        'negative':     (0.8, 0.0, 0.0),
    }
    for idx, sentiment in enumerate(model.classes_):
        sentiment_col = f'{sentiment}_coef'
        top_features  = vocab_df.nlargest(15, sentiment_col)
        n             = len(top_features)
        if n == 0:
            axes[idx].set_visible(False)
            continue
        intensities = np.linspace(1.0, 0.4, n)
        base        = np.array(base_colors.get(str(sentiment).lower(), (0.5, 0.5, 0.5)))
        colors      = [tuple(1 - inten * (1 - base)) for inten in intensities]
        axes[idx].barh(range(n), top_features[sentiment_col], color=colors)
        axes[idx].set_yticks(range(n))
        axes[idx].set_yticklabels(top_features['Word/Phrase'], fontsize=9)
        axes[idx].set_xlabel('Coefficient Value', fontsize=10)
        axes[idx].set_title(f'Top Features — {sentiment.capitalize()}',
                             fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'features_by_sentiment_75_25.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print("Feature importance chart saved to features_by_sentiment_75_25.png")

    # ── Step 9: Save predictions CSV ──────────────────────────────────────────
    predictions_df = pd.DataFrame({
        'text':                content_test.values,
        'actual_sentiment':    sent_test.values,
        'predicted_sentiment': sent_predict,
        'confidence':          confidence,
        'correct_prediction':  (sent_test.values == sent_predict),
    })
    for i, cls in enumerate(model.classes_):
        predictions_df[f'prob_{cls}'] = sent_proba[:, i]

    predictions_df['is_mixed'] = False
    mid_mask = predictions_df['predicted_sentiment'].isin(['neutral', 'neutral/mixed'])
    if mid_mask.any():
        predictions_df.loc[mid_mask, 'is_mixed'] = \
            predictions_df[mid_mask].apply(mixed_rule, axis=1)

    predictions_df.to_csv(OUTPUT_DIR / 'predicted_data_multi_75_25.csv', index=False)
    print("Predictions saved to predicted_data_multi_75_25.csv")

    # ── Step 10: Save model and vectorizer ────────────────────────────────────
    joblib.dump(model,      OUTPUT_DIR / 'sentiment_model_75_25.pkl')
    joblib.dump(vectorizer, OUTPUT_DIR / 'tfidf_vectorizer_75_25.pkl')
    print("Model saved to sentiment_model_75_25.pkl")
    print("Vectorizer saved to tfidf_vectorizer_75_25.pkl")

    # ── Step 11: Write training log ───────────────────────────────────────────
    original_stdout = sys.stdout
    with open(OUTPUT_DIR / 'training_multi_75_25.log', 'w') as log_file:
        sys.stdout = log_file
        print("=" * 60)
        print("SENTIMENT ANALYSIS MODEL TRAINING REPORT — 75/25 SPLIT")
        print("=" * 60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nSplit: 75% Training / 25% Testing")
        print(f"Training samples: {len(content_train)}")
        print(f"Test samples:     {len(content_test)}")
        print(f"\nTraining sentiment distribution:")
        print(sent_train.value_counts().to_string())
        print(f"\nTest sentiment distribution:")
        print(sent_test.value_counts().to_string())
        print(f"\nTF-IDF feature matrix shape: {content_train_tfidf.shape}")
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)} terms")
        print("\n" + "=" * 60)
        print("5-FOLD STRATIFIED CROSS-VALIDATION RESULTS")
        print("=" * 60)
        print(f"Accuracy scores: {cv_scores.round(4)}")
        print(f"Mean Accuracy:   {cv_scores.mean():.4f}")
        print(f"Std Accuracy:    {cv_scores.std():.4f}")
        print(f"Macro F1 scores: {cv_f1.round(4)}")
        print(f"Mean Macro F1:   {cv_f1.mean():.4f}")
        print(f"Std Macro F1:    {cv_f1.std():.4f}")
        print("\n" + "=" * 60)
        print("TEST SET PERFORMANCE METRICS")
        print("=" * 60)
        print(classification_report(sent_test, sent_predict))
        print(f"Accuracy:          {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Macro F1:          {macro_f1:.4f}")
        print(f"Weighted F1:       {weighted_f1:.4f}")
        print(f"Log Loss:          {model_log_loss:.4f}")
        print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
        print(cm)
        print("\n" + "=" * 60)
        print("COMPARISON NOTE")
        print("=" * 60)
        print("Compare these results against training_multi.log (80/20 split).")
        print("A higher test accuracy in the 80/20 model is expected since it")
        print("trained on more data. The 75/25 model has a larger, more")
        print("representative test set making its metrics more conservative")
        print("and potentially more realistic.")

    sys.stdout = original_stdout
    print("\nTraining log saved to training_multi_75_25.log")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR / 'training_testing_data.csv'):
        print("Error: training_testing_data.csv not found!")
        sys.exit(1)

    test_df = pd.read_csv(DATA_DIR / 'training_testing_data.csv')
    if 'sentiment' not in test_df.columns or 'neutral' in test_df['sentiment'].values:
        print("Preprocessing: Generating sentiment-labeled dataset...")
        add_sentiment_values_to_file()
        print("Dataset creation complete.\n")

    print("Training start — 75/25 split...")
    main()
    print("\nTraining complete. All results saved to the output/ folder.")

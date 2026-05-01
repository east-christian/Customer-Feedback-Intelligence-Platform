"""
pipeline_ml.py
Sentiment Analysis — Machine Learning Pipeline

Handles model training, text preprocessing, and sentiment prediction.
Uses Logistic Regression with TF-IDF vectorization.

Author: Christian East; February 22 2026
Collaborators: Birajman Tamang, Kelsang Yonjan
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import re

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DATA_DIR        = PROJECT_ROOT / "src" / "sample_data"
OUTPUT_DIR      = PROJECT_ROOT / "output"
MODEL_FILE      = OUTPUT_DIR / "sentiment_model.pkl"
VECTORIZER_FILE = OUTPUT_DIR / "tfidf_vectorizer.pkl"


def ensure_output_dir():
    """Create the output folder if it does not already exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sentiments_from_stars(stars, classification_type="three_class"):
    """
    Convert a star rating (1-5) into a sentiment label.
    *** 4-5 stars = positive, 3 stars = neutral/mixed, 1-2 stars = negative ***
    Used to automatically label the training data before the model is trained.
    """
    try:
        stars = float(stars)
    except (TypeError, ValueError):
        return None
    if classification_type == "binary":
        if stars >= 4: return "positive"
        if stars <= 2: return "negative"
        return None
    if stars >= 4:   return "positive"
    if stars == 3:   return "neutral/mixed"
    return "negative"


def prepare_training_data():
    """
    Load the training CSV and prepare it for model training.
    Adds a sentiment column based on star ratings if one does not already exist.
    Also creates a clean_text column (lowercased review text) if missing.
    """
    train_file = DATA_DIR / "training_testing_data.csv"
    if not train_file.exists():
        candidates = sorted(DATA_DIR.glob("*.csv"))
        if candidates:
            train_file = candidates[0]
        else:
            raise FileNotFoundError("No training CSV found in src/sample_data")
    df = pd.read_csv(train_file)
    if "sentiment" not in df.columns:
        df["sentiment"] = df["stars"].apply(lambda x: sentiments_from_stars(x, "three_class"))
    if "clean_text" not in df.columns:
        if "text" in df.columns:
            df["clean_text"] = df["text"].fillna("").astype(str).str.lower()
        elif "raw_text" in df.columns:
            df["clean_text"] = df["raw_text"].fillna("").astype(str).str.lower()
        else:
            raise ValueError("Training data must contain 'clean_text', 'text', or 'raw_text'")
    return df[df["sentiment"].notna()].copy()


def train_model():
    """
    Train the sentiment classification model from scratch.

    Process:
    *** Step 1 — Load labelled review data
    *** Step 2 — Split into 80% training / 20% testing (stratified)
    *** Step 3 — Convert text to numbers using TF-IDF
    *** Step 4 — Train a Logistic Regression classifier
    *** Step 5 — Measure accuracy on the test set
    *** Step 6 — Save the trained model to disk for reuse

    Returns the trained model, vectorizer, and accuracy score.
    """
    df = prepare_training_data()
    content, sent = df["clean_text"], df["sentiment"]

    content_train, content_test, sent_train, sent_test = train_test_split(
        content, sent, test_size=0.2, random_state=2016, stratify=sent
    )

    # common words removed to improve accuracy
    extra_stop = {"review", "user", "star", "stars", "https", "http", "amp"}
    stop_words  = set(sk_text.ENGLISH_STOP_WORDS) | extra_stop

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words=list(stop_words),
    )
    X_train = vectorizer.fit_transform(content_train)
    X_test  = vectorizer.transform(content_test)

    model = LogisticRegression(max_iter=1000, random_state=2016, C=0.8)
    model.fit(X_train, sent_train)

    accuracy = accuracy_score(sent_test, model.predict(X_test))

    joblib.dump(model,      MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return model, vectorizer, accuracy


@st.cache_resource
def load_or_train_model():
    """
    Load the sentiment model from disk if it already exists, otherwise train a new one.
    The @st.cache_resource decorator keeps the model in memory for the whole session
    so it is not reloaded from disk every time you click a button.
    """
    ensure_output_dir()
    if MODEL_FILE.exists() and VECTORIZER_FILE.exists():
        return joblib.load(MODEL_FILE), joblib.load(VECTORIZER_FILE), None
    with st.spinner("Training model from sample data..."):
        return train_model()


def preprocess_reviews(df):
    """
    Clean and prepare an uploaded review CSV for analysis.

    *** Automatically renames columns from common scraper formats
        (Instant Data Scraper, Octoparse, Apify, etc.)
    *** Assigns a review_id if the column is missing
    *** Lowercases all review text so the model treats 'Good' and 'good' the same
    *** Raises a clear error if no review text column can be found
    """
    # assign review_id if missing
    if "review_id" not in df.columns:
        df["review_id"] = [f"review_{i}" for i in range(len(df))]

    # auto-detect and rename columns from scraper exports
    col_map    = {}
    cols_lower = {c.lower(): c for c in df.columns}

    if "text" not in df.columns:
        text_candidates = [
            "comment", "review", "review_text", "review text", "body",
            "content", "description", "reviewbody", "review_body",
            "raw_text", "clean_text", "reviewtext", "reviewcontent",
        ]
        for candidate in text_candidates:
            if candidate in cols_lower:
                col_map[cols_lower[candidate]] = "text"
                break
        if "text" not in col_map.values():
            for c in df.columns:
                if "comment" in c.lower() or "review" in c.lower():
                    col_map[c] = "text"
                    break

    if "stars" not in df.columns:
        star_candidates = [
            "rating", "star", "stars", "star_rating", "starrating",
            "score", "rate", "reviewrating", "review_rating",
        ]
        for candidate in star_candidates:
            if candidate in cols_lower:
                col_map[cols_lower[candidate]] = "stars"
                break

    if "date" not in df.columns:
        date_candidates = [
            "datetime", "created", "created_at", "createdat",
            "review_date", "reviewdate", "posted", "posted_at",
            "time", "timestamp", "published", "date_created",
        ]
        for candidate in date_candidates:
            if candidate in cols_lower:
                col_map[cols_lower[candidate]] = "date"
                break

    if col_map:
        df = df.rename(columns=col_map)

    # clean the text column
    if "clean_text" in df.columns:
        df["clean_text"] = df["clean_text"].fillna("").astype(str).str.lower()
    elif "text" in df.columns:
        df["clean_text"] = df["text"].fillna("").astype(str).str.lower()
    elif "raw_text" in df.columns:
        df["clean_text"] = df["raw_text"].fillna("").astype(str).str.lower()
    else:
        raise ValueError(
            "Could not find a review text column. "
            "Please rename your text column to 'text' before uploading."
        )
    return df


# Mixed-signal detection helpers
# These check if a review contains both positive and negative language
# e.g. "Great coffee but terrible service" — this is a mixed-signal review

CONTRAST_WORDS = {"but","however","though","although","yet","except","overall","while"}
POS_CUES = {"good","great","nice","friendly","fast","clean","love","excellent","amazing","enjoy"}
NEG_CUES = {"bad","slow","rude","wrong","dirty","hate","awful","terrible","issue","problem"}


def has_contrast(text: str) -> bool:
    """Check if the review contains a contrast word like 'but' or 'however'."""
    t = f" {text.lower()} "
    return any(f" {w} " in t for w in CONTRAST_WORDS)


def has_dual_polarity_words(text: str) -> bool:
    """Check if the review uses both positive and negative words."""
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return bool(tokens & POS_CUES) and bool(tokens & NEG_CUES)


def mixed_rule(row) -> bool:
    """
    Decide if a neutral review is actually a mixed-signal review.
    A review is flagged as mixed if:
    *** The model was moderately confident in both positive and negative
        AND the review contains a contrast word
    *** OR the review contains a contrast word AND both positive and negative vocabulary
    """
    text  = str(row.get("clean_text", ""))
    p_pos = float(row.get("prob_positive", 0.0))
    p_neg = float(row.get("prob_negative", 0.0))
    prob_cond     = (p_pos >= 0.30) and (p_neg >= 0.30) and (abs(p_pos - p_neg) <= 0.25)
    contrast_cond = has_contrast(text)
    lex_cond      = has_dual_polarity_words(text)
    return (prob_cond and contrast_cond) or (contrast_cond and lex_cond)


def predict_reviews(df, model, vectorizer):
    """
    Run the sentiment model on every review in the uploaded CSV.

    *** Converts each review to a TF-IDF number vector
    *** The model predicts positive, negative, or neutral for each review
    *** Records the confidence score (how certain the model was)
    *** Flags mixed-signal reviews using the contrast word + polarity vocabulary rules
    """
    tfidf = vectorizer.transform(df["clean_text"])
    preds = model.predict(tfidf)
    probs = model.predict_proba(tfidf)

    df["predicted_sentiment"] = preds
    df["confidence"]           = probs.max(axis=1)
    for idx, cls in enumerate(model.classes_):
        df[f"prob_{cls}"] = probs[:, idx]

    df["is_mixed"]  = False
    middle_mask = df["predicted_sentiment"].isin(["neutral", "neutral/mixed"])
    if middle_mask.any():
        df.loc[middle_mask, "is_mixed"] = df[middle_mask].apply(mixed_rule, axis=1)

    return df

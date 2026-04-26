import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import ast
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "sample_data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_FILE = OUTPUT_DIR / "sentiment_model.pkl"
VECTORIZER_FILE = OUTPUT_DIR / "tfidf_vectorizer.pkl"

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def sentiments_from_stars(stars, classification_type="three_class"):
    try:
        stars = float(stars)
    except (TypeError, ValueError):
        return None

    if classification_type == "binary":
        if stars >= 4:
            return "positive"
        if stars <= 2:
            return "negative"
        return None

    if stars >= 4:
        return "positive"
    if stars == 3:
        return "neutral/mixed"
    return "negative"

def prepare_training_data():
    train_file = DATA_DIR / "training_testing_data.csv"
    if not train_file.exists():
        csv_candidates = sorted(DATA_DIR.glob("*.csv"))
        if csv_candidates:
            train_file = csv_candidates[0]
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
    df = prepare_training_data()
    content = df["clean_text"]
    sent = df["sentiment"]

    content_train, content_test, sent_train, sent_test = train_test_split(
        content, sent, test_size=0.2, random_state=2016, stratify=sent
    )

    extra_stop = {"review", "user", "star", "stars", "https", "http", "amp"}
    stop_words = set(sk_text.ENGLISH_STOP_WORDS) | extra_stop

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words=list(stop_words),
    )
    X_train = vectorizer.fit_transform(content_train)
    X_test = vectorizer.transform(content_test)

    model = LogisticRegression(max_iter=1000, random_state=2016, C=0.8)
    model.fit(X_train, sent_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(sent_test, preds)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return model, vectorizer, accuracy

def load_or_train_model():
    ensure_output_dir()
    if MODEL_FILE.exists() and VECTORIZER_FILE.exists():
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer, None

    with st.spinner("Training model from sample data..."):
        return train_model()

def preprocess_reviews(df):
    if "clean_text" in df.columns:
        df["clean_text"] = df["clean_text"].fillna("").astype(str).str.lower()
    elif "text" in df.columns:
        df["clean_text"] = df["text"].fillna("").astype(str).str.lower()
    elif "raw_text" in df.columns:
        df["clean_text"] = df["raw_text"].fillna("").astype(str).str.lower()
    else:
        raise ValueError("Uploaded CSV must contain a 'text', 'raw_text', or 'clean_text' column")
    return df

CONTRAST_WORDS = {"but", "however", "though", "although", "yet", "except", "overall", "while"}
POS_CUES = {"good", "great", "nice", "friendly", "fast", "clean", "love", "excellent", "amazing", "enjoy"}
NEG_CUES = {"bad", "slow", "rude", "wrong", "dirty", "hate", "awful", "terrible", "issue", "problem"}

def has_contrast(text: str) -> bool:
    t = f" {text.lower()} "
    return any(f" {w} " in t for w in CONTRAST_WORDS)

def has_dual_polarity_words(text: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return (len(tokens & POS_CUES) > 0) and (len(tokens & NEG_CUES) > 0)

def mixed_rule(row) -> bool:
    text = str(row.get("clean_text", ""))
    p_pos = float(row.get("prob_positive", 0.0))
    p_neg = float(row.get("prob_negative", 0.0))
    prob_cond = (p_pos >= 0.30) and (p_neg >= 0.30) and (abs(p_pos - p_neg) <= 0.25)
    contrast_cond = has_contrast(text)
    lex_cond = has_dual_polarity_words(text)
    return (prob_cond and contrast_cond) or (contrast_cond and lex_cond)

def predict_reviews(df, model, vectorizer):
    tfidf = vectorizer.transform(df["clean_text"])
    preds = model.predict(tfidf)
    probs = model.predict_proba(tfidf)

    df["predicted_sentiment"] = preds
    df["confidence"] = probs.max(axis=1)
    for idx, cls in enumerate(model.classes_):
        df[f"prob_{cls}"] = probs[:, idx]
        
    df["is_mixed"] = False
    middle_mask = df["predicted_sentiment"].isin(["neutral", "neutral/mixed"])
    if middle_mask.any():
        df.loc[middle_mask, "is_mixed"] = df[middle_mask].apply(mixed_rule, axis=1)
        
    return df

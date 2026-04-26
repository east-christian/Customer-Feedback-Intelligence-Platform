import io
import hashlib
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import json
import time
import ollama
import ast
import re

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, Image as RLImage,
)

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DATA_DIR        = PROJECT_ROOT / "src" / "sample_data"
OUTPUT_DIR      = PROJECT_ROOT / "output"
MODEL_FILE      = OUTPUT_DIR / "sentiment_model.pkl"
VECTORIZER_FILE = OUTPUT_DIR / "tfidf_vectorizer.pkl"

THEMES = [
    "Product Quality",
    "Product Availability",
    "Customer Service",
    "Speed of Service",
    "Store Environment",
    "Price & Value",
    "Digital & Rewards",
    "Policies & Safety",
]

st.set_page_config(page_title="Customer Feedback Intelligence Platform", layout="wide")

COLOUR_MAP = {
    "positive":      "#16a34a",
    "negative":      "#dc2626",
    "neutral":       "#6b7280",
    "neutral/mixed": "#9ca3af",
}

def _text_col(df):
    """
    Find the column that contains the review text.
    Checks for 'text', 'raw_text', or 'clean_text' — returns whichever exists first.
    """
    for c in ["text", "raw_text", "clean_text"]:
        if c in df.columns:
            return c
    return "clean_text"

def _csv_hash(df):
    """
    Create a unique fingerprint for a dataframe.
    Used to detect if the same CSV is uploaded twice so we can skip re-running the LLM.
    """
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


### LLM

def call_llm(prompt, model="gemma3:4b"):
    """
    Send a prompt to the local Ollama language model and return its response.
    Uses Gemma 3 4B by default — a small, fast model that runs entirely on your Mac.
    If Ollama is not running this will raise an error.
    """
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise e


### Model

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
        return "positive" if stars >= 4 else ("negative" if stars <= 2 else None)
    return "positive" if stars >= 4 else ("neutral/mixed" if stars == 3 else "negative")

def prepare_training_data():
    """
    Load the training CSV from the sample_data folder and prepare it for model training.
    Adds a 'sentiment' column based on star ratings if one does not already exist.
    Also creates a 'clean_text' column (lowercased review text) if missing.
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
    *** Step 2 — Split into 80% training / 20% testing
    *** Step 3 — Convert text to numbers using TF-IDF
                 (words that appear often in one review but rarely across all reviews
                  get a higher score — this helps the model focus on meaningful words)
    *** Step 4 — Train a Logistic Regression classifier
                 (a fast, reliable algorithm for text classification)
    *** Step 5 — Measure accuracy on the test set
    *** Step 6 — Save the trained model to disk so it can be reused

    Returns the trained model, vectorizer, and accuracy score.
    """
    df = prepare_training_data()
    content, sent = df["clean_text"], df["sentiment"]
    c_train, c_test, s_train, s_test = train_test_split(
        content, sent, test_size=0.2, random_state=2016, stratify=sent)
    extra_stop = {"review", "user", "star", "stars", "https", "http", "amp"}
    stop_words  = set(sk_text.ENGLISH_STOP_WORDS) | extra_stop
    vectorizer  = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                   min_df=2, max_df=0.8, stop_words=list(stop_words))
    X_train = vectorizer.fit_transform(c_train)
    X_test  = vectorizer.transform(c_test)
    model   = LogisticRegression(max_iter=1000, random_state=2016, C=0.8)
    model.fit(X_train, s_train)
    accuracy = accuracy_score(s_test, model.predict(X_test))
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    return model, vectorizer, accuracy

@st.cache_resource
def load_or_train_model():
    """
    Load the sentiment model from disk if it already exists, otherwise train a new one.
    The @st.cache_resource decorator means the model is loaded only once per session
    and kept in memory — so switching tabs or clicking buttons does not reload it from disk.
    """
    ensure_output_dir()
    if MODEL_FILE.exists() and VECTORIZER_FILE.exists():
        return joblib.load(MODEL_FILE), joblib.load(VECTORIZER_FILE), None
    return train_model()


### Preprocessing

def preprocess_reviews(df):
    """
    Clean and prepare an uploaded review CSV for analysis.

    *** Automatically renames columns from common scraper formats
        (e.g. Instant Data Scraper, Octoparse, Apify)
    *** Lowercases all review text so the model treats 'Good' and 'good' the same
    *** Raises a clear error if no review text column can be found
    """
    ### Auto-detect and rename common column name variations
    # Handles CSVs from Instant Data Scraper, Octoparse, Apify, etc.
    col_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    # Text column — look for any column that likely contains review text
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
        # Also catch Yelp/Instant Data Scraper dynamic class names like comment__373c0
        if "text" not in col_map.values():
            for c in df.columns:
                if "comment" in c.lower() or "review" in c.lower():
                    if c not in col_map:
                        col_map[c] = "text"
                        break

    # Stars column
    if "stars" not in df.columns:
        star_candidates = [
            "rating", "star", "stars", "star_rating", "starrating",
            "score", "rate", "reviewrating", "review_rating",
        ]
        for candidate in star_candidates:
            if candidate in cols_lower:
                col_map[cols_lower[candidate]] = "stars"
                break

    # Date column
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

    ### Clean the text column
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

### Mixed-signal detection helpers
# These functions check if a review contains both positive and negative language
# e.g. "Great coffee but terrible service" — this is a mixed-signal review

CONTRAST_WORDS = {"but","however","though","although","yet","except","overall","while"}
POS_CUES = {"good","great","nice","friendly","fast","clean","love","excellent","amazing","enjoy"}
NEG_CUES = {"bad","slow","rude","wrong","dirty","hate","awful","terrible","issue","problem"}

def has_contrast(text):
    """Check if the review contains a contrast word like 'but' or 'however'."""
    t = f" {text.lower()} "
    return any(f" {w} " in t for w in CONTRAST_WORDS)

def has_dual_polarity_words(text):
    """Check if the review uses both positive words (e.g. 'great') and negative words (e.g. 'slow')."""
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return bool(tokens & POS_CUES) and bool(tokens & NEG_CUES)

def mixed_rule(row):
    """
    Decide if a neutral review is actually a mixed-signal review.
    A review is flagged as mixed if it meets at least one of these conditions:
    *** The model gave it a moderate probability for both positive and negative
        AND the review contains a contrast word
    *** The review contains a contrast word AND uses both positive and negative vocabulary
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
    df["confidence"] = probs.max(axis=1)
    for idx, cls in enumerate(model.classes_):
        df[f"prob_{cls}"] = probs[:, idx]
    df["is_mixed"] = False
    mask = df["predicted_sentiment"].isin(["neutral", "neutral/mixed"])
    if mask.any():
        df.loc[mask, "is_mixed"] = df[mask].apply(mixed_rule, axis=1)
    return df


### Theme extraction
# Sends reviews to the local LLM in batches and asks it to assign business themes.
# Each batch is retried up to 5 times if the LLM response is invalid or contains invented themes.

def build_prompt(batch, themes):
    """
    Build the instruction message sent to the LLM for a batch of reviews.
    Tells the LLM exactly which themes are allowed and asks it to return valid JSON only.
    """
    numbered = "\n".join([f"Review {i+1}:\n{str(r)[:250]}" for i, r in enumerate(batch)])
    return f"""You are a professional theme classifier for customer reviews.
Available themes: {themes}
RULES:
- Only assign themes from the available list. NEVER invent themes.
- Every review MUST have at least one theme.
- Return ONLY a valid JSON dictionary where keys are Review Numbers and values are arrays of themes.
- Generate exactly {len(batch)} keys. No extra explanation.
Example: {{"1": ["Customer Service"], "2": ["Speed of Service", "Price & Value"]}}
Reviews:
{numbered}
Return ONLY valid JSON."""

def extract_themes_with_retry(batch_info, themes_list, max_retries=5):
    """
    Send one batch of reviews to the LLM and extract their themes.
    *** Retries up to 5 times if the LLM returns invalid JSON or invents themes not in the list
    *** Rejects any theme not in the approved list (hallucination guard)
    *** Returns the batch index, reviews, assigned themes, and success/failure status
    """
    batch_idx, batch = batch_info
    prompt = build_prompt(batch, themes_list)
    for attempt in range(1, max_retries + 1):
        try:
            raw         = call_llm(prompt)
            first_brace = raw.find("{")
            first_brack = raw.find("[")
            start = min(i for i in [first_brace, first_brack] if i != -1) \
                    if any(i != -1 for i in [first_brace, first_brack]) else -1
            end   = max(raw.rfind("}"), raw.rfind("]")) + 1
            if start == -1 or end <= start:
                raise ValueError("No valid JSON in LLM output")
            clean = raw[start:end]
            try:
                parsed_data = json.loads(clean)
            except json.JSONDecodeError:
                try:
                    parsed_data = ast.literal_eval(clean)
                except Exception:
                    raise ValueError(f"Could not parse: {clean}")
            parsed_dict = {}
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    if isinstance(item, dict):
                        parsed_dict.update(item)
            elif isinstance(parsed_data, dict):
                parsed_dict = parsed_data
            else:
                raise ValueError("Response is not valid JSON")
            themes = []
            for idx in range(1, len(batch) + 1):
                t = parsed_dict.get(str(idx), [])
                if not isinstance(t, list): t = [t]
                if not t: t = ["Customer Service"]
                themes.append(t)
            if len(themes) != len(batch):
                raise ValueError(f"Count mismatch: got {len(themes)}, expected {len(batch)}")
            validated = []
            for tlist in themes:
                safe  = [str(list(t.values())[0]) if isinstance(t, dict) and t else str(t) for t in tlist]
                valid = []
                for st_t in safe:
                    for real in themes_list:
                        if st_t.strip().lower() == real.lower():
                            if real not in valid: valid.append(real)
                            break
                if not valid:
                    raise ValueError(f"Hallucination: '{tlist}'")
                validated.append(valid)
            return batch_idx, batch, validated, "success"
        except Exception as e:
            print(f"Batch {batch_idx} attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
    return batch_idx, batch, None, "failed"

def extract_themes(df, themes_list, batch_size=30, max_workers=2):
    """
    Assign business themes to every review using the local LLM.

    *** Splits reviews into batches of 30 so the LLM does not get overwhelmed
    *** Runs 2 batches at the same time (parallel processing) to save time
    *** Shows a progress bar while processing
    *** Any batch that fails after 5 retries is dropped from the results
    *** Results are cached — if you upload the same CSV again the LLM is skipped
    """
    reviews = df["clean_text"].fillna("").tolist()
    batches = [(i, reviews[i:i+batch_size]) for i in range(0, len(reviews), batch_size)]
    success, failed = [], []
    pbar   = st.progress(0)
    status = st.empty()
    total  = len(batches)
    done   = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(extract_themes_with_retry, b, themes_list): b for b in batches}
        for future in as_completed(futures):
            try:
                bidx, batch, bthemes, bstatus = future.result(timeout=180)
            except TimeoutError:
                bidx, batch = futures[future]; bstatus, bthemes = "failed", [["FAILED"]]*len(batch)
            except Exception:
                bidx, batch = futures[future]; bstatus, bthemes = "failed", [["FAILED"]]*len(batch)
            if bstatus == "success":
                for i, (_, vt) in enumerate(zip(batch, bthemes)):
                    success.append({"original_idx": bidx+i, "themes": ", ".join(vt)})
            else:
                for i in range(len(batch)):
                    failed.append({"original_idx": bidx+i, "themes": "FAILED"})
            done += 1
            if done % max(1, total//100) == 0 or done == total:
                pbar.progress(done/total)
                status.text(f"Processed batch {done}/{total} — please wait...")
    lookup = {r["original_idx"]: r["themes"] for r in success + failed}
    df["themes"] = [lookup.get(i, "FAILED") for i in range(len(df))]
    before = len(df)
    df = df[~df["themes"].str.contains("FAILED", na=False)]
    if len(df) < before:
        print(f"Dropped {before-len(df)} reviews due to extraction failure.")
    pbar.progress(1.0)
    status.text("Complete!")
    return df


def generate_executive_summary(df):
    """
    Use the local LLM to write a plain-English business summary of the analysis results.
    *** Calculates key stats (positive %, negative %, top themes) from the data
    *** Sends those stats to Ollama as a structured prompt
    *** The LLM writes 3 short paragraphs: what is going well, what needs attention,
        and what to do next
    *** All of this runs locally — no data is sent to the internet
    """
    total    = len(df)
    pos      = int((df["predicted_sentiment"] == "positive").sum())
    neg      = int((df["predicted_sentiment"] == "negative").sum())
    mixed    = int(df["predicted_sentiment"].isin(["neutral","neutral/mixed"]).sum())
    avg_conf = df["confidence"].mean() * 100 if "confidence" in df.columns else None
    top_themes, top_neg_theme = "", ""
    if "themes" in df.columns:
        exp = df["themes"].str.split(r",\s*").explode().str.strip()
        exp = exp[~exp.isin(["FAILED","","NOT PROCESSED"])]
        top_themes = ", ".join(exp.value_counts().head(5).index.tolist())
        neg_df = df[df["predicted_sentiment"] == "negative"]
        if not neg_df.empty:
            neg_exp = neg_df["themes"].str.split(r",\s*").explode().str.strip()
            neg_exp = neg_exp[~neg_exp.isin(["FAILED","","NOT PROCESSED"])]
            if not neg_exp.empty:
                top_neg_theme = neg_exp.value_counts().idxmax()
    prompt = f"""You are a business analyst writing an executive summary for a customer feedback report.
Statistics:
- Total reviews: {total}
- Positive: {pos} ({pos/total*100:.1f}%)
- Negative: {neg} ({neg/total*100:.1f}%)
- Neutral/Mixed: {mixed} ({mixed/total*100:.1f}%)
- Avg model confidence: {f"{avg_conf:.1f}%" if avg_conf else "N/A"}
- Top themes: {top_themes if top_themes else "N/A"}
- Top negative theme: {top_neg_theme if top_neg_theme else "N/A"}

Write exactly 3 short paragraphs:
1. What is going well
2. What needs attention
3. What to do next

2-3 sentences each. Use the numbers. No bullet points. No headings. Professional tone."""
    return call_llm(prompt)


### Shared CSS

METRIC_CSS = """
<style>
/* ── Import clean font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 17px !important;
}
.stMarkdown p, .stMarkdown li {
    font-size: 17px !important;
    line-height: 1.8 !important;
    color: #1f2937 !important;
}
.stMarkdown h1 { font-size: 2.2rem !important; font-weight: 700 !important; color: #1e3a5f !important; }
.stMarkdown h2 { font-size: 1.8rem !important; font-weight: 700 !important; color: #1e3a5f !important; }
.stMarkdown h3 { font-size: 1.4rem !important; font-weight: 600 !important; color: #1e3a5f !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #065a82 100%) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span:not(.st-emotion-cache-1kyxreq),
[data-testid="stSidebar"] p {
    color: #ffffff !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
/* Run Analysis and Reset buttons — white bg, dark navy text */
[data-testid="stSidebar"] .stButton > button {
    background-color: #ffffff !important;
    color: #1e3a5f !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    border: 2px solid #ffffff !important;
    padding: 10px 16px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #cfe8ff !important;
    color: #1e3a5f !important;
    border-color: #cfe8ff !important;
}
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span {
    color: #1e3a5f !important;
    font-weight: 700 !important;
}
/* File uploader area */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.12) !important;
    border: 1.5px solid rgba(255,255,255,0.4) !important;
    border-radius: 8px !important;
    padding: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #ffffff !important;
}
/* Upload button inside file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: rgba(255,255,255,0.18) !important;
    border: 2px dashed rgba(255,255,255,0.6) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background-color: #ffffff !important;
    color: #1e3a5f !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    border: none !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button p {
    color: #1e3a5f !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: #ffffff !important;
}
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption {
    color: #c7d7f0 !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #f0f4ff !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #374151 !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
}

/* ── Buttons ── */
.stButton button {
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 2px solid #1e3a5f !important;
    color: #1e3a5f !important;
    background-color: #ffffff !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, .stSelectbox div {
    font-size: 16px !important;
    border-radius: 8px !important;
    border: 1.5px solid #d1d5db !important;
}
label, .stSelectbox label, .stTextInput label,
.stSlider label, .stCheckbox label,
.stTextArea label {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* ── Expander ── */
.stExpander {
    border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important;
}
.stExpander summary {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #1e3a5f !important;
}

/* ── Alert boxes ── */
.stInfo  { font-size: 16px !important; border-radius: 8px !important; }
.stSuccess { font-size: 16px !important; border-radius: 8px !important; }
.stWarning { font-size: 16px !important; border-radius: 8px !important; }
.stError   { font-size: 16px !important; border-radius: 8px !important; }

/* ── Dataframe ── */
.stDataFrame { font-size: 15px !important; border-radius: 8px !important; }

/* ── Caption ── */
.stCaption, [data-testid="stCaptionContainer"] { font-size: 14px !important; color: #6b7280 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
    border: 1.5px solid #c7d7f0;
    border-radius: 12px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 2px 8px rgba(30,58,95,0.08);
}
[data-testid="stMetricLabel"] {
    font-size: 14px !important; font-weight: 600 !important; color: #6b7280 !important;
    text-transform: uppercase; letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] { font-size: 30px !important; font-weight: 700 !important; color: #1e3a5f !important; }
[data-testid="stMetricDelta"] { font-size: 13px !important; color: #6b7280 !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Remove outline box from selectbox display text ── */
[data-baseweb="select"] > div {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] {
    border: none !important;
}
/* Keep the outer selectbox container border clean */
[data-baseweb="select"] {
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
}
/* Remove inner highlighted box around selected value text */
[data-baseweb="select"] span {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
</style>
"""


### Chart / axis helpers

CHART_FONT = dict(family="Inter, sans-serif", size=14, color="#1f2937")
AXIS_TITLE_FONT = dict(family="Inter, sans-serif", size=15, color="#1e3a5f")
TICK_FONT = dict(family="Inter, sans-serif", size=13, color="#374151")

def _vertical_bar_axis():
    return dict(
        xaxis=dict(
            tickformat=".0%",
            range=[-0.05, 1.05],
            tickvals=[i/10 for i in range(0, 11)],
            ticktext=[f"{i*10}%" for i in range(0, 11)],
            tickfont=TICK_FONT,
            title_standoff=12,
        ),
        yaxis=dict(
            rangemode="tozero", dtick=1, tickformat="d",
            tickfont=TICK_FONT,
        ),
    )

def _horizontal_bar_axis(max_val, title=""):
    nice_max = max(1, int(max_val) + 1)
    step = max(1, int(nice_max / 8))
    return dict(
        xaxis=dict(
            range=[0, nice_max + step * 0.5],
            tick0=0, dtick=step, tickformat="d",
            rangemode="tozero", tickfont=TICK_FONT,
            title=dict(text=title, font=AXIS_TITLE_FONT) if title else {},
        ),
        yaxis=dict(
            categoryorder="total ascending",
            tickfont=TICK_FONT,
        ),
    )

def _base_layout(**extra):
    layout = dict(
        paper_bgcolor="white",
        plot_bgcolor="#fafbff",
        margin=dict(l=10, r=20, t=55, b=80),
        bargap=0.2,
        bargroupgap=0.05,
        font=CHART_FONT,
        title_font=dict(family="Inter, sans-serif", size=17, color="#1e3a5f"),
        legend=dict(
            font=dict(family="Inter, sans-serif", size=13, color="#374151"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
    )
    layout.update(extra)
    return layout

def _hist_yaxis(df):
    n = len(df)
    est_max = max(20, n // 4)
    ceiling = ((est_max // 20) + 1) * 20
    return dict(
        rangemode="tozero", dtick=20, tickformat="d",
        range=[0, ceiling], tickfont=TICK_FONT,
    )

def _chart_download(fig, filename, label="Download chart as PNG"):
    try:
        png_bytes = fig.to_image(format="png", scale=2)
        st.download_button(label=label, data=png_bytes,
                           file_name=filename, mime="image/png")
    except Exception:
        st.caption("Install kaleido for chart downloads: pip install kaleido")


### PDF helpers

def _pdf_styles():
    base = getSampleStyleSheet()
    HDR  = rl_colors.HexColor("#1e3a5f")
    return (
        ParagraphStyle("T",  parent=base["Title"],   fontSize=20, textColor=HDR,
                       fontName="Helvetica-Bold", spaceAfter=4),
        ParagraphStyle("S",  parent=base["Normal"],  fontSize=10,
                       textColor=rl_colors.HexColor("#6b7280"), spaceAfter=14),
        ParagraphStyle("H2", parent=base["Heading2"],fontSize=13, textColor=HDR,
                       fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6),
        ParagraphStyle("B",  parent=base["Normal"],  fontSize=9, leading=13,
                       textColor=rl_colors.HexColor("#374151")),
    )

def _pdf_kpi_table(df, HDR, RULE, BG):
    total    = len(df)
    pos      = int((df["predicted_sentiment"] == "positive").sum())
    neg      = int((df["predicted_sentiment"] == "negative").sum())
    mixed    = int(df["predicted_sentiment"].isin(["neutral","neutral/mixed"]).sum())
    avg_conf = df["confidence"].mean()*100 if "confidence" in df.columns else None
    GRN = rl_colors.HexColor("#16a34a")
    RED = rl_colors.HexColor("#dc2626")
    rows = [["Metric","Count","Share"],
            ["Total Reviews", f"{total:,}", "100%"]]
    if pos   > 0: rows.append(["Positive",        f"{pos:,}",   f"{pos/total*100:.1f}%"])
    if neg   > 0: rows.append(["Negative",         f"{neg:,}",   f"{neg/total*100:.1f}%"])
    if mixed > 0: rows.append(["Neutral / Mixed",  f"{mixed:,}", f"{mixed/total*100:.1f}%"])
    if avg_conf:   rows.append(["Avg Confidence",  f"{avg_conf:.1f}%", "—"])
    tbl = Table(rows, colWidths=[3.0*inch, 1.5*inch, 1.5*inch])
    style = [
        ("BACKGROUND",    (0,0),(-1,0),BG), ("TEXTCOLOR",(0,0),(-1,0),HDR),
        ("FONTNAME",      (0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,0),9),
        ("FONTNAME",      (0,1),(-1,-1),"Helvetica"), ("FONTSIZE",(0,1),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white, rl_colors.HexColor("#f9fafb")]),
        ("TEXTCOLOR",     (0,1),(-1,-1),rl_colors.HexColor("#374151")),
        ("GRID",          (0,0),(-1,-1),0.4,RULE),
        ("TOPPADDING",    (0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",   (0,0),(-1,-1),8),
    ]
    if pos > 0: style.append(("TEXTCOLOR",(2,2),(2,2),GRN))
    if neg > 0: style.append(("TEXTCOLOR",(2,3),(2,3),RED))
    tbl.setStyle(TableStyle(style))
    return tbl

def _pdf_theme_table(df, HDR, RULE, BG, body_s):
    exp_all = df["themes"].str.split(r",\s*").explode().str.strip()
    exp_all = exp_all[~exp_all.isin(["FAILED","","NOT PROCESSED"])]
    if exp_all.empty:
        return Paragraph("No theme data available for selected filters.", body_s)
    tc = exp_all.value_counts().head(8).reset_index()
    tc.columns = ["Theme","Mentions"]
    theme_rows = [["Theme","Mentions"]] + [[str(c) for c in row] for row in tc.values.tolist()]
    tbl = Table(theme_rows, colWidths=[4.0*inch, 2.0*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),BG), ("TEXTCOLOR",(0,0),(-1,0),HDR),
        ("FONTNAME",      (0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,0),9),
        ("FONTNAME",      (0,1),(-1,-1),"Helvetica"), ("FONTSIZE",(0,1),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white, rl_colors.HexColor("#f9fafb")]),
        ("TEXTCOLOR",     (0,1),(-1,-1),rl_colors.HexColor("#374151")),
        ("GRID",          (0,0),(-1,-1),0.4,RULE),
        ("TOPPADDING",    (0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",   (0,0),(-1,-1),8),
    ]))
    return tbl


def build_custom_pdf(df, report_title="Customer Feedback Report",
                     inc_kpi=True, inc_pie=True,
                     inc_trend=True, inc_themes=True) -> bytes:
    HDR  = rl_colors.HexColor("#1e3a5f")
    RULE = rl_colors.HexColor("#e5e7eb")
    BG   = rl_colors.HexColor("#f0f4ff")

    title_s, sub_s, h2_s, body_s = _pdf_styles()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.85*inch,  bottomMargin=0.75*inch)
    story = []

    story.append(Paragraph(report_title, title_s))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=12))

    # Summary Metrics
    if inc_kpi:
        story.append(Paragraph("Summary", h2_s))
        story.append(_pdf_kpi_table(df, HDR, RULE, BG))
        story.append(Spacer(1, 14))

    # Sentiment Pie Chart
    if inc_pie:
        story.append(Paragraph("Sentiment Distribution", h2_s))
        try:
            counts = df["predicted_sentiment"].value_counts().reset_index()
            counts.columns = ["sentiment","count"]
            fig_pie = px.pie(counts, names="sentiment", values="count",
                             color="sentiment", color_discrete_map=COLOUR_MAP)
            fig_pie.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="white",
                legend=dict(font=dict(size=12)),
            )
            png = fig_pie.to_image(format="png", width=500, height=320, scale=2)
            story.append(RLImage(io.BytesIO(png), width=4.5*inch, height=2.9*inch))
            story.append(Spacer(1, 10))
        except Exception as e:
            story.append(Paragraph(f"Pie chart unavailable: {e}", body_s))

    # Per-sentiment theme bar charts
    if "themes" in df.columns:
        sentiment_colours = {
            "positive":      "#16a34a",
            "negative":      "#dc2626",
            "neutral":       "#6b7280",
            "neutral/mixed": "#9ca3af",
        }
        for sent, clr in sentiment_colours.items():
            sent_df = df[df["predicted_sentiment"] == sent]
            if sent_df.empty:
                continue
            exp = sent_df["themes"].str.split(r",\s*").explode().str.strip()
            exp = exp[~exp.isin(["FAILED","","NOT PROCESSED"])]
            if exp.empty:
                continue
            tc = exp.value_counts().head(8).reset_index()
            tc.columns = ["Theme","Count"]
            try:
                fig_bar = px.bar(tc, x="Count", y="Theme", orientation="h",
                                 title=f"Top Themes — {sent.capitalize()} Reviews",
                                 color_discrete_sequence=[clr])
                fig_bar.update_traces(marker_line_color="white", marker_line_width=1)
                fig_bar.update_layout(
                    bargap=0.2, paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=200, r=40, t=50, b=60),
                    xaxis=dict(
                        rangemode="tozero", tickformat="d",
                        title=dict(text="Number of Mentions", font=dict(size=13, color="#1e3a5f")),
                        tickfont=dict(size=11),
                    ),
                    yaxis=dict(
                        categoryorder="total ascending",
                        title=dict(text="", font=dict(size=13, color="#1e3a5f")),
                        tickfont=dict(size=11),
                    ),
                )
                png = fig_bar.to_image(format="png", width=800, height=380, scale=2)
                story.append(Paragraph(
                    f"Top Themes — {sent.capitalize()} Reviews", h2_s))
                story.append(RLImage(io.BytesIO(png), width=6.5*inch, height=3.1*inch))
                story.append(Spacer(1, 10))
            except Exception as e:
                story.append(Paragraph(
                    f"Bar chart unavailable for {sent}: {e}", body_s))

    # Sentiment Over Time
    if inc_trend and "date" in df.columns:
        story.append(Paragraph("Sentiment Over Time", h2_s))
        try:
            d  = df.copy()
            d["date"] = pd.to_datetime(d["date"])
            td = (d.groupby([pd.Grouper(key="date", freq="ME"), "predicted_sentiment"])
                   .size().reset_index(name="count"))
            if len(td) > 1:
                fig_t = px.line(td, x="date", y="count", color="predicted_sentiment",
                                color_discrete_map=COLOUR_MAP)
                fig_t.update_layout(
                    margin=dict(l=80, r=30, t=30, b=70),
                    paper_bgcolor="white",
                    xaxis=dict(
                        title=dict(text="Date", font=dict(size=13, color="#1e3a5f")),
                        tickfont=dict(size=11),
                    ),
                    yaxis=dict(
                        rangemode="tozero", range=[0, None],
                        dtick=1, tickformat="d",
                        title=dict(text="Number of Reviews", font=dict(size=13, color="#1e3a5f")),
                        tickfont=dict(size=11),
                    ),
                    legend=dict(title=dict(text="Sentiment"), font=dict(size=11)),
                )
                png = fig_t.to_image(format="png", width=700, height=320, scale=2)
                story.append(RLImage(io.BytesIO(png), width=6.0*inch, height=2.7*inch))
                story.append(Spacer(1, 10))
            else:
                story.append(Paragraph(
                    "Not enough date range for a trend chart.", body_s))
        except Exception as e:
            story.append(Paragraph(f"Trend chart unavailable: {e}", body_s))

    # Top Themes Table
    if inc_themes and "themes" in df.columns:
        story.append(Paragraph("Top Themes Summary", h2_s))
        story.append(_pdf_theme_table(df, HDR, RULE, BG, body_s))

    doc.build(story)
    return buf.getvalue()


### 
#  TAB PAGE RENDERERS
### 

def page_overview(df):
    st.markdown(METRIC_CSS, unsafe_allow_html=True)

    st.markdown("---")

    ### Month selector
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            if min_date < max_date:
                st.markdown("**Filter by date range**")
                date_range = st.slider(
                    "Select date range:", min_value=min_date, max_value=max_date,
                    value=(min_date, max_date), format="MMM YYYY",
                    key="date_slider_overview")
                df = df[(df["date"].dt.date >= date_range[0]) &
                        (df["date"].dt.date <= date_range[1])]
                st.caption(
                    f"Showing {len(df):,} reviews from "
                    f"{date_range[0].strftime('%b %Y')} to "
                    f"{date_range[1].strftime('%b %Y')}")
        except Exception:
            pass

    ### KPI cards
    total    = len(df)
    pos      = int((df["predicted_sentiment"] == "positive").sum())
    neg      = int((df["predicted_sentiment"] == "negative").sum())
    mixed    = int(df["predicted_sentiment"].isin(["neutral","neutral/mixed"]).sum())
    avg_conf = df["confidence"].mean() if "confidence" in df.columns else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews",   f"{total:,}",                    "100% of reviews")
    c2.metric("Positive",        f"{pos:,}",                      f"{pos/total*100:.1f}% of reviews")
    c3.metric("Negative",        f"{neg:,}",                      f"{neg/total*100:.1f}% of reviews")
    c4.metric("Neutral / Mixed", f"{mixed:,}",                    f"{mixed/total*100:.1f}% of reviews")
    c5.metric("Avg Confidence",  f"{avg_conf*100:.1f}%" if avg_conf else "N/A", "model certainty")

    st.markdown("---")

    ### Keyword search
    st.markdown("**Keyword Search**")
    keyword = st.text_input("Search reviews for a keyword or phrase:",
                            placeholder="e.g. wait, rude, coffee, parking...",
                            key="keyword_overview")
    if keyword.strip():
        tc_name = _text_col(df)
        mask    = df[tc_name].str.contains(keyword.strip(), case=False, na=False)
        kw_df   = df[mask]
        st.markdown(f"**{len(kw_df):,} reviews** contain the word **'{keyword.strip()}'**")
        if not kw_df.empty:
            kw_counts = kw_df["predicted_sentiment"].value_counts().reset_index()
            kw_counts.columns = ["sentiment","count"]
            fig_kw = px.pie(kw_counts, names="sentiment", values="count",
                            title=f"Sentiment for reviews mentioning '{keyword.strip()}'",
                            color="sentiment", color_discrete_map=COLOUR_MAP)
            st.plotly_chart(fig_kw, use_container_width=True)
            _chart_download(fig_kw, "keyword_sentiment.png", "Download keyword sentiment chart")
            with st.expander(f"Show matching reviews ({len(kw_df)})"):
                show_cols = [tc_name, "predicted_sentiment", "confidence"] + \
                            [c for c in ["themes"] if c in kw_df.columns]
                st.dataframe(kw_df[show_cols].head(50), use_container_width=True)
        st.markdown("---")

    ### Pie chart
    counts = df["predicted_sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment","count"]
    fig_pie = px.pie(counts, names="sentiment", values="count",
                     title="Predicted Sentiment Distribution",
                     color="sentiment", color_discrete_map=COLOUR_MAP)
    st.plotly_chart(fig_pie, use_container_width=True)
    _chart_download(fig_pie, "sentiment_distribution.png", "Download sentiment pie chart")

    ### Time trend
    if "date" in df.columns:
        try:
            td = (df.groupby([pd.Grouper(key="date", freq="ME"), "predicted_sentiment"])
                    .size().reset_index(name="count"))
            if len(td) > 1:
                fig_trend = px.line(td, x="date", y="count", color="predicted_sentiment",
                                    title="Sentiment Over Time", color_discrete_map=COLOUR_MAP)
                fig_trend.update_layout(**_base_layout(
                    xaxis=dict(
                        title=dict(text="Date", font=AXIS_TITLE_FONT),
                        tickfont=TICK_FONT,
                    ),
                    yaxis=dict(
                        title=dict(text="Number of Reviews", font=AXIS_TITLE_FONT),
                        rangemode="tozero", range=[0, None],
                        dtick=1, tickformat="d", tickfont=TICK_FONT,
                    ),
                    legend=dict(
                        title=dict(text="Sentiment", font=AXIS_TITLE_FONT),
                        font=TICK_FONT,
                    ),
                ))
                st.plotly_chart(fig_trend, use_container_width=True)
                _chart_download(fig_trend, "sentiment_trend.png", "Download trend chart")
            else:
                st.info("Not enough date range for a trend chart.")
        except Exception:
            st.info("Date column could not be parsed.")

    ### Data preview
    st.markdown("---")
    st.markdown("**Data preview — first 10 rows**")
    preview_cols = [c for c in ["predicted_sentiment","confidence","is_mixed","themes"]
                    if c in df.columns]
    st.dataframe(df[preview_cols].head(10), use_container_width=True)

    ### AI Executive Summary
    st.markdown("---")
    st.markdown("### Executive Summary")
    if st.button("Generate AI Summary", key="gen_summary"):
        with st.spinner("Asking AI to analyse results..."):
            try:
                summary = generate_executive_summary(df)
                st.session_state["exec_summary"] = summary
            except Exception as e:
                st.error(f"Could not generate summary: {e}")
                st.info("Make sure Ollama is running: ollama run gemma3:4b")

    if "exec_summary" in st.session_state:
        paragraphs = [p.strip() for p in
                      st.session_state["exec_summary"].split("\n\n") if p.strip()]
        labels  = ["What is going well", "What needs attention", "What to do next"]
        bgs     = ["#f0fdf4", "#fef2f2", "#eff6ff"]
        borders = ["#16a34a", "#dc2626", "#2563eb"]
        for i, para in enumerate(paragraphs[:3]):
            label = labels[i] if i < len(labels) else f"Point {i+1}"
            bg    = bgs[i]    if i < len(bgs)    else "#f9fafb"
            bdr   = borders[i] if i < len(borders) else "#6b7280"
            st.markdown(
                f'<div style="background:{bg}; border-left:4px solid {bdr}; '
                f'padding:14px 18px; border-radius:6px; margin-bottom:10px;">'
                f'<strong style="color:{bdr};">{label}</strong><br>'
                f'<span style="font-size:14px; color:#374151;">{para}</span>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.caption("Click Generate AI Summary to get an AI-written analysis of the results.")

    ### Customizable Report Builder
    st.markdown("---")
    st.markdown("### Download Report")
    st.write("Select what to include in your PDF report before downloading.")

    with st.expander("Report Options", expanded=True):
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Sentiment Filter**")
            inc_positive = st.checkbox("Include Positive Reviews",      value=True)
            inc_negative = st.checkbox("Include Negative Reviews",      value=True)
            inc_neutral  = st.checkbox("Include Neutral / Mixed Reviews", value=True)
            st.markdown("**Sections to include**")
            inc_kpi    = st.checkbox("Summary Metrics table",           value=True)
            inc_pie    = st.checkbox("Sentiment pie chart",             value=True)
            inc_trend  = st.checkbox("Sentiment over time chart",       value=True)
            inc_themes = st.checkbox("Top themes table",                value=True)

        with rc2:
            st.markdown("**Date Range Filter**")
            use_date_filter = False
            if "date" in df.columns:
                try:
                    min_d = df["date"].min().date()
                    max_d = df["date"].max().date()
                    report_dates = st.slider(
                        "Include reviews from:",
                        min_value=min_d, max_value=max_d,
                        value=(min_d, max_d), format="MMM YYYY",
                        key="report_date_slider")
                    use_date_filter = True
                except Exception:
                    st.info("Date column could not be parsed.")
            else:
                st.info("No date column in your CSV — date filter not available.")

            st.markdown("**Report Title**")
            report_title = st.text_input(
                "Title shown on the report:",
                value="Customer Feedback Report",
                key="report_title_custom")

    if st.button("Generate PDF Report", key="gen_pdf_overview"):
        with st.spinner("Building your custom report..."):
            try:
                report_df = df.copy()
                if use_date_filter and "date" in report_df.columns:
                    report_df = report_df[
                        (report_df["date"].dt.date >= report_dates[0]) &
                        (report_df["date"].dt.date <= report_dates[1])]
                keep = []
                if inc_positive: keep.append("positive")
                if inc_negative: keep.append("negative")
                if inc_neutral:  keep.extend(["neutral","neutral/mixed"])
                if keep:
                    report_df = report_df[report_df["predicted_sentiment"].isin(keep)]
                if report_df.empty:
                    st.error("No reviews match your filters. Adjust options and try again.")
                else:
                    pdf_bytes = build_custom_pdf(
                        report_df, report_title=report_title,
                        inc_kpi=inc_kpi, inc_pie=inc_pie,
                        inc_trend=inc_trend, inc_themes=inc_themes)
                    st.session_state["pdf_bytes_overview"]  = pdf_bytes
                    st.session_state["pdf_review_count"]    = len(report_df)
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.info("Run: pip install reportlab kaleido")

    if "pdf_bytes_overview" in st.session_state:
        st.success("Report ready.")
        st.download_button(
            label="Download PDF Report",
            data=st.session_state["pdf_bytes_overview"],
            file_name=f"feedback_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            key="dl_pdf_overview")


def page_positive(df):
    pos_df = df[df["predicted_sentiment"] == "positive"].copy()
    st.markdown(f"### Positive Reviews — {len(pos_df):,} total")
    if pos_df.empty:
        st.info("No positive reviews found.")
        return

    if "confidence" in pos_df.columns:
        fig_hist = px.histogram(pos_df, x="confidence",
                                title="Confidence Score Distribution — Positive Reviews",
                                color_discrete_sequence=["#16a34a"])
        fig_hist.update_traces(marker_line_color="white", marker_line_width=2,
                               xbins=dict(start=0.0, end=1.01, size=0.10))
        fig_hist.update_layout(**_base_layout(
            xaxis=dict(
                title=dict(text="Model Confidence Score (higher = more certain)", font=AXIS_TITLE_FONT),
                tickformat=".0%", range=[0, 1.05],
                tickvals=[i/10 for i in range(0, 11)],
                ticktext=[f"{i*10}%" for i in range(0, 11)],
                tickfont=TICK_FONT,
            ),
            yaxis=dict(
                title=dict(text="Number of Reviews", font=AXIS_TITLE_FONT),
                **_hist_yaxis(pos_df),
            ),
        ))
        st.plotly_chart(fig_hist, use_container_width=True)
        _chart_download(fig_hist, "positive_confidence.png", "Download confidence chart")

    if "themes" in pos_df.columns:
        st.markdown("**Most mentioned themes in positive reviews**")
        exp = pos_df["themes"].str.split(r",\s*").explode().str.strip()
        exp = exp[~exp.isin(["FAILED","","NOT PROCESSED"])]
        tc  = exp.value_counts().head(8).reset_index(); tc.columns = ["Theme","Count"]
        fig_bar = px.bar(tc, x="Count", y="Theme", orientation="h",
                         color_discrete_sequence=["#16a34a"])
        fig_bar.update_traces(marker_line_color="white", marker_line_width=1)
        fig_bar.update_layout(**_base_layout(
            **_horizontal_bar_axis(tc["Count"].max(), title="Number of reviews mentioning this theme")))
        st.plotly_chart(fig_bar, use_container_width=True)
        _chart_download(fig_bar, "positive_themes.png", "Download themes chart")

    st.markdown("**Sample positive reviews**")
    tc_name = _text_col(pos_df)
    hc, bc  = st.columns([3,1])
    with bc: st.button("Refresh", key="pos_refresh")
    for _, row in pos_df.sample(min(10, len(pos_df))).iterrows():
        conf = f" *(confidence: {row['confidence']*100:.0f}%)*" if "confidence" in row else ""
        st.success(f'"{row[tc_name]}"{conf}')


def page_negative(df):
    neg_df = df[df["predicted_sentiment"] == "negative"].copy()
    st.markdown(f"### Negative Reviews — {len(neg_df):,} total")
    if neg_df.empty:
        st.info("No negative reviews found.")
        return

    if "confidence" in neg_df.columns:
        fig_hist = px.histogram(neg_df, x="confidence",
                                title="Confidence Score Distribution — Negative Reviews",
                                color_discrete_sequence=["#dc2626"])
        fig_hist.update_traces(marker_line_color="white", marker_line_width=2,
                               xbins=dict(start=0.0, end=1.01, size=0.10))
        fig_hist.update_layout(**_base_layout(
            xaxis=dict(
                title=dict(text="Model Confidence Score (higher = more certain)", font=AXIS_TITLE_FONT),
                tickformat=".0%", range=[0, 1.05],
                tickvals=[i/10 for i in range(0, 11)],
                ticktext=[f"{i*10}%" for i in range(0, 11)],
                tickfont=TICK_FONT,
            ),
            yaxis=dict(
                title=dict(text="Number of Reviews", font=AXIS_TITLE_FONT),
                **_hist_yaxis(neg_df),
            ),
        ))
        st.plotly_chart(fig_hist, use_container_width=True)
        _chart_download(fig_hist, "negative_confidence.png", "Download confidence chart")

    if "themes" in neg_df.columns:
        st.markdown("**Most mentioned themes in negative reviews**")
        exp = neg_df["themes"].str.split(r",\s*").explode().str.strip()
        exp = exp[~exp.isin(["FAILED","","NOT PROCESSED"])]
        tc  = exp.value_counts().head(8).reset_index(); tc.columns = ["Theme","Count"]
        fig_bar = px.bar(tc, x="Count", y="Theme", orientation="h",
                         color_discrete_sequence=["#dc2626"])
        fig_bar.update_traces(marker_line_color="white", marker_line_width=1)
        fig_bar.update_layout(**_base_layout(
            **_horizontal_bar_axis(tc["Count"].max(), title="Number of reviews mentioning this theme")))
        st.plotly_chart(fig_bar, use_container_width=True)
        _chart_download(fig_bar, "negative_themes.png", "Download themes chart")

    st.markdown("**Sample negative reviews**")
    tc_name = _text_col(neg_df)
    hc, bc  = st.columns([3,1])
    with bc: st.button("Refresh", key="neg_refresh")
    for _, row in neg_df.sample(min(10, len(neg_df))).iterrows():
        conf = f" *(confidence: {row['confidence']*100:.0f}%)*" if "confidence" in row else ""
        st.error(f'"{row[tc_name]}"{conf}')


def page_neutral(df):
    neu_df = df[df["predicted_sentiment"].isin(["neutral","neutral/mixed"])].copy()
    st.markdown(f"### Neutral / Mixed Reviews — {len(neu_df):,} total")
    st.write("These reviews sit in the middle — the model did not detect a clearly positive or negative tone.")
    if neu_df.empty:
        st.info("No neutral or mixed reviews found.")
        return

    if "confidence" in neu_df.columns:
        fig_hist = px.histogram(neu_df, x="confidence",
                                title="Confidence Score Distribution — Neutral / Mixed Reviews",
                                color_discrete_sequence=["#6b7280"])
        fig_hist.update_traces(marker_line_color="white", marker_line_width=2,
                               xbins=dict(start=0.0, end=1.01, size=0.10))
        fig_hist.update_layout(**_base_layout(
            xaxis=dict(
                title=dict(text="Model Confidence Score (higher = more certain it is neutral)", font=AXIS_TITLE_FONT),
                tickformat=".0%", range=[0, 1.05],
                tickvals=[i/10 for i in range(0, 11)],
                ticktext=[f"{i*10}%" for i in range(0, 11)],
                tickfont=TICK_FONT,
            ),
            yaxis=dict(
                title=dict(text="Number of Reviews", font=AXIS_TITLE_FONT),
                **_hist_yaxis(neu_df),
            ),
        ))
        st.plotly_chart(fig_hist, use_container_width=True)
        _chart_download(fig_hist, "neutral_confidence.png", "Download confidence chart")

    if "is_mixed" in neu_df.columns:
        truly_neutral = neu_df[neu_df["is_mixed"] == False]
        mixed_signal  = neu_df[neu_df["is_mixed"] == True]
        col_a, col_b  = st.columns(2)
        with col_a:
            st.metric("Purely Neutral", f"{len(truly_neutral):,}",
                      "No strong positive or negative language")
        with col_b:
            st.metric("Mixed Signal", f"{len(mixed_signal):,}",
                      "Contains both positive and negative language")

    if "themes" in neu_df.columns:
        st.markdown("**Most mentioned themes in neutral / mixed reviews**")
        exp = neu_df["themes"].str.split(r",\s*").explode().str.strip()
        exp = exp[~exp.isin(["FAILED","","NOT PROCESSED"])]
        tc  = exp.value_counts().head(8).reset_index(); tc.columns = ["Theme","Count"]
        fig_bar = px.bar(tc, x="Count", y="Theme", orientation="h",
                         color_discrete_sequence=["#6b7280"])
        fig_bar.update_traces(marker_line_color="white", marker_line_width=1)
        fig_bar.update_layout(**_base_layout(
            **_horizontal_bar_axis(tc["Count"].max(), title="Number of reviews mentioning this theme")))
        st.plotly_chart(fig_bar, use_container_width=True)
        _chart_download(fig_bar, "neutral_themes.png", "Download themes chart")

    st.markdown("**Sample neutral / mixed reviews**")
    tc_name = _text_col(neu_df)
    hc, bc  = st.columns([3,1])
    with bc: st.button("Refresh", key="neu_refresh")
    for _, row in neu_df.sample(min(10, len(neu_df))).iterrows():
        conf  = f" *(confidence: {row['confidence']*100:.0f}%)*" if "confidence" in row else ""
        label = "Mixed" if ("is_mixed" in row and row["is_mixed"]) else "Neutral"
        st.warning(f'**[{label}]** "{row[tc_name]}"{conf}')


def page_themes(df):
    if "themes" not in df.columns:
        st.warning("No themes column found. Run the analysis pipeline first.")
        return

    st.markdown("**Keyword Search**")
    kw = st.text_input("Search reviews for a keyword:",
                       placeholder="e.g. wait, rude, coffee...", key="keyword_themes")
    if kw.strip():
        tc_name = _text_col(df)
        mask    = df[tc_name].str.contains(kw.strip(), case=False, na=False)
        kw_df   = df[mask]
        st.markdown(f"**{len(kw_df):,} reviews** mention '{kw.strip()}'")
        if not kw_df.empty and "themes" in kw_df.columns:
            kw_exp = kw_df["themes"].str.split(r",\s*").explode().str.strip()
            kw_exp = kw_exp[~kw_exp.isin(["FAILED","","NOT PROCESSED"])]
            kw_tc  = kw_exp.value_counts().head(8).reset_index()
            kw_tc.columns = ["Theme","Count"]
            fig_kw = px.bar(kw_tc, x="Count", y="Theme", orientation="h",
                            title=f"Themes for reviews mentioning '{kw.strip()}'",
                            color_discrete_sequence=["#065a82"])
            fig_kw.update_traces(marker_line_color="white", marker_line_width=1)
            fig_kw.update_layout(**_base_layout(**_horizontal_bar_axis(kw_tc["Count"].max())))
            st.plotly_chart(fig_kw, use_container_width=True)
        st.markdown("---")

    exp_all = df["themes"].str.split(r",\s*").explode().str.strip()
    exp_all = exp_all[~exp_all.isin(["FAILED","","NOT PROCESSED"])]
    theme_counts = exp_all.value_counts().head(20).reset_index()
    theme_counts.columns = ["Theme","Count"]

    fig_theme_bar = px.bar(theme_counts, x="Count", y="Theme", orientation="h",
                           title="Most Common Themes", color="Count",
                           color_continuous_scale="Viridis")
    fig_theme_bar.update_traces(marker_line_color="white", marker_line_width=1)
    fig_theme_bar.update_layout(**_base_layout(
        coloraxis_showscale=False,
        **_horizontal_bar_axis(theme_counts["Count"].max(), title="Number of mentions")))
    st.plotly_chart(fig_theme_bar, use_container_width=True)
    _chart_download(fig_theme_bar, "top_themes.png", "Download themes chart")

    st.markdown("**Theme mention counts**")
    st.dataframe(theme_counts, use_container_width=True)

    st.markdown("---")
    st.markdown("**Sentiment Breakdown by Theme**")
    df_exp = df.assign(Theme=df["themes"].str.split(r",\s*")).explode("Theme")
    df_exp["Theme"] = df_exp["Theme"].str.strip()
    df_exp = df_exp[~df_exp["Theme"].isin(["FAILED","","NOT PROCESSED"])].reset_index(drop=True)

    if not df_exp.empty:
        pivot = pd.crosstab(df_exp["Theme"], df_exp["predicted_sentiment"],
                            margins=True, margins_name="Total")
        scols = [c for c in ["positive","negative","neutral","neutral/mixed"] if c in pivot.columns]
        for c in scols:
            pivot[c+" (%)"] = (pivot[c] / pivot["Total"] * 100).round(1)
        ordered = []
        for c in scols: ordered.extend([c, c+" (%)"])
        ordered.append("Total")
        st.dataframe(pivot[ordered], use_container_width=True)

        st.markdown("---")
        st.markdown("**Interactive Theme Sentiment Distribution**")
        unique_themes = sorted(df_exp["Theme"].unique().tolist())
        selected      = st.selectbox("Select a theme:", unique_themes, key="theme_select")
        t_data        = df_exp[df_exp["Theme"] == selected]
        t_counts      = t_data["predicted_sentiment"].value_counts().reset_index()
        t_counts.columns = ["sentiment","count"]

        fig_pie = px.pie(t_counts, names="sentiment", values="count",
                         title=f"Sentiment split for '{selected}'",
                         color="sentiment", color_discrete_map=COLOUR_MAP)
        st.plotly_chart(fig_pie, use_container_width=True)
        _chart_download(fig_pie, f"theme_{selected.replace(' ','_')}_sentiment.png",
                        "Download theme sentiment chart")

        st.markdown(f"**{selected} — review counts by sentiment**")
        st.dataframe(t_counts.rename(columns={"sentiment":"Sentiment","count":"Reviews"}),
                     use_container_width=True)

        st.markdown("---")
        st.markdown(f"**Top phrases in '{selected}'**")
        try:
            sent_docs = df_exp.groupby("Theme")["clean_text"].apply(
                lambda x: " ".join(x.dropna())).to_dict()
            if selected in sent_docs and sent_docs[selected].strip():
                extra_stop = {"review","user","star","stars","https","http","amp","just","like","im"}
                sw  = list(set(sk_text.ENGLISH_STOP_WORDS) | extra_stop)
                tv  = TfidfVectorizer(ngram_range=(2,3), stop_words=sw)
                mat = tv.fit_transform(list(sent_docs.values()))
                idx = list(sent_docs.keys()).index(selected)
                dense  = mat.toarray()
                scores = dense[idx] * (dense.argmax(axis=0) == idx)
                top_i  = scores.argsort()[-10:][::-1]
                phrases = [tv.get_feature_names_out()[i] for i in top_i if scores[i] > 0]
                pcounts = [sent_docs[selected].count(p) for p in phrases]
                pf = pd.DataFrame({"Phrase":phrases,"Count":pcounts}).sort_values("Count",ascending=True)
                fig3 = px.bar(pf, x="Count", y="Phrase", orientation="h")
                fig3.update_traces(marker_line_color="white", marker_line_width=1)
                fig3.update_layout(**_base_layout(
                    **_horizontal_bar_axis(pf["Count"].max(), title="Mentions")))
                st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.info(f"Could not extract phrases: {e}")

        if "date" in df.columns:
            st.markdown("---")
            st.subheader("Time-Based & Emergent Trends")
            try:
                theme_time = (df_exp.groupby([pd.Grouper(key="date", freq="ME"), "Theme"])
                              .size().reset_index(name="count"))
                if len(theme_time["date"].unique()) > 1:
                    st.markdown(f"**Monthly volume for '{selected}'**")
                    sel_tt  = theme_time[theme_time["Theme"] == selected]
                    n_bars  = len(sel_tt)
                    chart_w = max(900, n_bars * 55)
                    fig_t = px.bar(sel_tt, x="date", y="count",
                                   labels={"date":"Month","count":"Mentions"})
                    fig_t.update_traces(marker_color="#065a82", marker_line_color="white",
                                        marker_line_width=2,
                                        width=1000*60*60*24*20)
                    fig_t.update_layout(
                        width=chart_w, height=450, bargap=0.4,
                        paper_bgcolor="white", plot_bgcolor="white",
                        margin=dict(l=50,r=20,t=45,b=130),
                        xaxis=dict(tickformat="%b %Y", tickmode="array",
                                   tickvals=sel_tt["date"].tolist(),
                                   tickangle=-45, tickfont=dict(size=11),
                                   title="Month", showgrid=False),
                        yaxis=dict(rangemode="tozero", range=[0,20], dtick=2,
                                   tickformat="d", title="Mentions",
                                   showgrid=True, gridcolor="#e5e7eb"))
                    st.markdown(
                        '<div style="overflow-x:auto; overflow-y:hidden; width:100%; '
                        'border:1px solid #e5e7eb; border-radius:6px; padding:4px;">',
                        unsafe_allow_html=True)
                    st.plotly_chart(fig_t, use_container_width=False)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("**Emergent Themes (month-over-month)**")
                    st.write("Change in mention volume between the two most recent months.")
                    months     = sorted(theme_time["date"].unique())
                    curr_month = months[-1]
                    prev_month = months[-2]
                    curr_df = theme_time[theme_time["date"]==curr_month].set_index("Theme")
                    prev_df = theme_time[theme_time["date"]==prev_month].set_index("Theme")
                    emer = curr_df[["count"]].join(prev_df[["count"]],
                                                    lsuffix="_curr", rsuffix="_prev",
                                                    how="outer").fillna(0)
                    emer["Change"]     = (emer["count_curr"]-emer["count_prev"]).astype(int)
                    emer["count_curr"] = emer["count_curr"].astype(int)
                    emer["count_prev"] = emer["count_prev"].astype(int)
                    st.dataframe(
                        emer.sort_values("Change",ascending=False)
                        [["count_prev","count_curr","Change"]]
                        .rename(columns={"count_prev":"Prev Month",
                                         "count_curr":"Current Month",
                                         "Change":"Momentum"}),
                        use_container_width=True)
                else:
                    st.info("Need at least 2 months of data for trend momentum.")
            except Exception as e:
                st.warning(f"Could not calculate trends: {e}")

        st.markdown("---")
        st.markdown("**Read actual reviews by theme + sentiment**")
        dd_col1, dd_col2 = st.columns(2)
        with dd_col1:
            dd_theme = st.selectbox("Theme:", unique_themes, key="dd_theme")
        with dd_col2:
            dd_sent  = st.selectbox("Sentiment:",
                                    ["negative","positive","neutral","neutral/mixed"],
                                    key="dd_sent")
        dd_data = df_exp[(df_exp["Theme"]==dd_theme) & (df_exp["predicted_sentiment"]==dd_sent)]
        if dd_data.empty:
            st.info(f"No {dd_sent} reviews found for '{dd_theme}'.")
        else:
            hc, bc = st.columns([3,1])
            with bc: st.button("Refresh", key="dd_refresh")
            tc_name = _text_col(dd_data)
            for rev in dd_data.sample(min(5,len(dd_data)))[tc_name].tolist():
                st.info(f'"{rev}"')


def page_outliers(df):
    st.markdown("### Outlier & Edge-Case Reviews")
    st.write("**Low-confidence predictions** are reviews the model was unsure about. "
             "**Mixed-signal reviews** contain both positive and negative language.")

    if "confidence" in df.columns:
        st.markdown("---")
        st.markdown("#### Low-Confidence Predictions")
        threshold = st.slider("Show reviews below this confidence level:",
                              0.40, 0.90, 0.60, 0.05,
                              help="60% means the model was only 60% sure of its prediction.")
        low_conf = df[df["confidence"] < threshold].copy()
        st.markdown(f"**{len(low_conf):,} reviews** predicted with less than "
                    f"**{threshold*100:.0f}% confidence** — may be worth checking manually.")

        if not low_conf.empty:
            fig_out = px.histogram(low_conf, x="confidence", color="predicted_sentiment",
                                   title="How uncertain was the model?",
                                   color_discrete_map=COLOUR_MAP, barmode="group")
            fig_out.update_traces(marker_line_color="white", marker_line_width=2,
                                  xbins=dict(start=0.0, end=1.01, size=0.05))
            fig_out.update_layout(**_base_layout(
                xaxis=dict(
                    title=dict(text="Model Confidence Score", font=AXIS_TITLE_FONT),
                    tickformat=".0%", range=[0, 1.05],
                    tickvals=[i/10 for i in range(0, 11)],
                    ticktext=[f"{i*10}%" for i in range(0, 11)],
                    tickfont=TICK_FONT,
                ),
                yaxis=dict(
                    title=dict(text="Number of Reviews", font=AXIS_TITLE_FONT),
                    **_hist_yaxis(low_conf),
                ),
                legend=dict(
                    title=dict(text="Predicted Sentiment", font=AXIS_TITLE_FONT),
                    font=TICK_FONT,
                ),
            ))
            st.plotly_chart(fig_out, use_container_width=True)
            _chart_download(fig_out, "low_confidence_distribution.png",
                            "Download low-confidence chart")

            tc_name   = _text_col(low_conf)
            show_cols = [tc_name, "predicted_sentiment", "confidence"] + \
                        [c for c in ["themes","is_mixed"] if c in low_conf.columns]
            st.markdown("**Reviews sorted by lowest confidence first:**")
            st.dataframe(low_conf[show_cols].sort_values("confidence").head(30),
                         use_container_width=True)
            st.download_button("Download low-confidence reviews",
                               data=low_conf.to_csv(index=False),
                               file_name="low_confidence_reviews.csv",
                               mime="text/csv", key="dl_low_conf")

    st.markdown("---")

    if "is_mixed" in df.columns:
        st.markdown("#### Mixed-Signal Reviews")
        st.write("Reviews containing contrast words like *'but', 'however', 'although'* "
                 "alongside both positive and negative vocabulary.")
        mixed_df = df[df["is_mixed"] == True].copy()
        st.markdown(f"**{len(mixed_df):,} mixed-signal reviews** detected")
        if not mixed_df.empty:
            tc_name   = _text_col(mixed_df)
            show_cols = [tc_name, "predicted_sentiment", "confidence"] + \
                        [c for c in ["themes"] if c in mixed_df.columns]
            st.dataframe(mixed_df[show_cols].head(30), use_container_width=True)
            st.download_button("Download mixed-signal reviews",
                               data=mixed_df.to_csv(index=False),
                               file_name="mixed_signal_reviews.csv",
                               mime="text/csv", key="dl_mixed")
        else:
            st.info("No mixed-signal reviews detected in this dataset.")


def page_trends(df):
    st.markdown("### Trends & Insights")
    st.write("Advanced analytics — spike detection, theme lifecycle, compliments vs concerns, word clouds and emergent themes.")

    df_exploded = get_exploded_themes(df)

    ### Top Compliments & Concerns
    st.markdown("---")
    st.markdown("#### Top Compliments vs Concerns")
    st.caption("Themes with the highest positive % vs highest negative % — at least 5 reviews required.")
    if not df_exploded.empty:
        render_top_compliments_concerns(df_exploded)
    else:
        st.info("No theme data available.")

    ### Theme × Sentiment Heatmap
    st.markdown("---")
    st.markdown("#### Theme × Sentiment Heatmap")
    st.caption("Visual breakdown of how each theme is perceived. Darker red = higher negative concentration.")
    if not df_exploded.empty:
        render_heatmap(df_exploded)
    else:
        st.info("No theme data available.")

    ### Spike Detection
    st.markdown("---")
    st.markdown("#### Negative Review Spike Detection")
    st.caption("Months where negative reviews exceeded 1.5 standard deviations above average — likely indicates an incident.")
    render_spike_detection(df)

    ### Theme Lifecycle
    st.markdown("---")
    render_theme_lifecycle(df_exploded)

    ### Emergent Themes
    st.markdown("---")
    st.markdown("#### Month-over-Month Theme Momentum")
    render_emergent_themes(df_exploded)

    ### Word Cloud
    st.markdown("---")
    st.markdown("#### Word Cloud by Sentiment")
    st.caption("Most frequent words per sentiment class. Larger = more frequent.")
    wc_sentiment = st.radio(
        "Choose sentiment:",
        ["positive","negative","neutral"],
        horizontal=True, key="wc_sentiment_trends",
    )
    render_word_cloud(df, wc_sentiment)


def normalize_sentiment_label(s):
    if isinstance(s, str) and s.lower() in ("neutral/mixed", "neutral"):
        return "neutral"
    return s


def get_exploded_themes(df):
    if "themes" not in df.columns:
        return pd.DataFrame(columns=["Theme", "predicted_sentiment", "clean_text"])
    df_exp = df.assign(Theme=df["themes"].str.split(r",\s*")).explode("Theme")
    df_exp["Theme"] = df_exp["Theme"].str.strip()
    df_exp = df_exp[~df_exp["Theme"].isin(["FAILED", "", "NOT PROCESSED"])]
    return df_exp[df_exp["Theme"].notna()].reset_index(drop=True)


def render_top_compliments_concerns(df_exploded):
    pivot = pd.crosstab(
        df_exploded["Theme"],
        df_exploded["predicted_sentiment"].apply(normalize_sentiment_label)
    )
    if pivot.empty:
        st.info("Not enough data for compliments/concerns analysis.")
        return
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot[pivot["Total"] >= 5]
    pivot["positive_pct"] = (pivot.get("positive", 0) / pivot["Total"] * 100).round(1)
    pivot["negative_pct"] = (pivot.get("negative", 0) / pivot["Total"] * 100).round(1)

    top_compliments = pivot.sort_values("positive_pct", ascending=False).head(5)
    top_concerns    = pivot.sort_values("negative_pct",  ascending=False).head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Compliments** — themes customers love most")
        comp_df = top_compliments.reset_index()[["Theme","positive_pct","Total"]]
        comp_df.columns = ["Theme","Positive %","Reviews"]
        fig = px.bar(comp_df, x="Positive %", y="Theme", orientation="h",
                     color="Positive %",
                     color_continuous_scale=["#D1FAE5","#10B981","#047857"],
                     text="Positive %")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=320, margin=dict(t=10,b=10,l=0,r=60),
                          yaxis=dict(categoryorder="total ascending", automargin=True),
                          coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top Concerns** — themes customers complain about most")
        conc_df = top_concerns.reset_index()[["Theme","negative_pct","Total"]]
        conc_df.columns = ["Theme","Negative %","Reviews"]
        fig = px.bar(conc_df, x="Negative %", y="Theme", orientation="h",
                     color="Negative %",
                     color_continuous_scale=["#FEE2E2","#EF4444","#991B1B"],
                     text="Negative %")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=320, margin=dict(t=10,b=10,l=0,r=60),
                          yaxis=dict(categoryorder="total ascending", automargin=True),
                          coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


def render_heatmap(df_exploded):
    pivot = pd.crosstab(
        df_exploded["Theme"],
        df_exploded["predicted_sentiment"].apply(normalize_sentiment_label),
        normalize="index"
    ) * 100
    for col in ["positive","neutral","negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["positive","neutral","negative"]]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=["Positive","Neutral","Negative"],
        y=pivot.index,
        colorscale=[[0,"#FFFFFF"],[0.5,"#FED7AA"],[1,"#DC2626"]],
        text=pivot.values.round(1),
        texttemplate="%{text}%",
        textfont={"size":12,"color":"black"},
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="% of reviews"),
    ))
    fig.update_layout(height=400, margin=dict(t=20,b=20,l=20,r=20),
                      xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Darker red cells = higher concentration of that sentiment for that theme.")


def render_spike_detection(df):
    if "date" not in df.columns:
        st.info("No date column found — spike detection unavailable.")
        return
    try:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[d["date"].notna()]
        if d.empty:
            return
        d["sentiment_clean"] = d["predicted_sentiment"].apply(normalize_sentiment_label)
        neg_df  = d[d["sentiment_clean"] == "negative"]
        monthly = neg_df.groupby(pd.Grouper(key="date", freq="ME")).size().reset_index(name="count")
        if len(monthly) < 3:
            st.info("Need at least 3 months of data to detect spikes.")
            return
        mean      = monthly["count"].mean()
        std       = monthly["count"].std()
        threshold = mean + 1.5 * std
        spikes    = monthly[monthly["count"] > threshold].sort_values("count", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["date"], y=monthly["count"],
            marker_color=["#F59E0B" if c > threshold else "#dc2626" for c in monthly["count"]],
            hovertemplate="<b>%{x|%b %Y}</b><br>Negative reviews: %{y}<extra></extra>",
        ))
        fig.add_hline(y=threshold, line_dash="dash", line_color="#F59E0B",
                      annotation_text=f"Spike threshold ({threshold:.0f})",
                      annotation_position="top right")
        fig.add_hline(y=mean, line_dash="dot", line_color="#9CA3AF",
                      annotation_text=f"Average ({mean:.0f})",
                      annotation_position="bottom right")
        fig.update_layout(height=400, margin=dict(t=30,b=20,l=20,r=20),
                          xaxis_title="Month", yaxis_title="Negative review count",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if not spikes.empty:
            st.markdown("**Detected Spikes:**")
            for _, row in spikes.head(5).iterrows():
                pct_above = (row["count"] - mean) / mean * 100 if mean > 0 else 0
                st.warning(
                    f"**{row['date'].strftime('%B %Y')}** — "
                    f"{int(row['count'])} negative reviews "
                    f"({pct_above:+.0f}% above average of {mean:.0f})"
                )
        else:
            st.success("No significant spikes detected. Negative review volume is stable.")
    except Exception as e:
        st.warning(f"Could not run spike detection: {e}")


def classify_theme_lifecycle(df_exploded, sentiment_filter="negative"):
    if "date" not in df_exploded.columns:
        return None
    try:
        d = df_exploded.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[d["date"].notna()]
        if d.empty:
            return None
        d["sentiment_clean"] = d["predicted_sentiment"].apply(normalize_sentiment_label)
        if sentiment_filter in ("positive","negative"):
            d = d[d["sentiment_clean"] == sentiment_filter]
        if d.empty:
            return None

        theme_time = (d.groupby([pd.Grouper(key="date", freq="ME"), "Theme"])
                       .size().reset_index(name="count"))
        months = sorted(theme_time["date"].unique())
        if len(months) < 4:
            return None

        midpoint           = len(months) // 2
        first_half_months  = months[:midpoint]
        second_half_months = months[midpoint:]

        results = []
        for theme in theme_time["Theme"].unique():
            t_data = theme_time[theme_time["Theme"] == theme]
            fh = t_data[t_data["date"].isin(first_half_months)]["count"]
            sh = t_data[t_data["date"].isin(second_half_months)]["count"]
            fh_avg = fh.mean() if not fh.empty else 0.0
            sh_avg = sh.mean() if not sh.empty else 0.0
            fh_avg = 0.0 if pd.isna(fh_avg) else fh_avg
            sh_avg = 0.0 if pd.isna(sh_avg) else sh_avg
            fh_presence = (fh > 0).sum()

            if fh_presence <= 1 and sh_avg > 0:
                lifecycle, color = "Emerging",  "#3B82F6"
            elif sh_avg > fh_avg * 1.3:
                lifecycle, color = "Growing",   "#10B981"
            elif sh_avg < fh_avg * 0.7:
                lifecycle, color = "Declining", "#F59E0B"
            else:
                lifecycle, color = "Stable",    "#6B7280"

            change_pct = ((sh_avg - fh_avg) / fh_avg * 100) if fh_avg > 0 else 100
            results.append({
                "Theme": theme, "Lifecycle": lifecycle, "Color": color,
                "First Half Avg": round(fh_avg, 1),
                "Second Half Avg": round(sh_avg, 1),
                "Change %": round(change_pct, 1),
            })
        return pd.DataFrame(results).sort_values("Change %", ascending=False)
    except Exception:
        return None


def render_theme_lifecycle(df_exploded):
    st.markdown("**Theme Lifecycle Classification**")
    st.caption("How each theme's mention volume changed from the first half to the second half of the period.")
    sentiment_filter = st.radio(
        "Filter by sentiment:",
        ["negative","positive"],
        format_func=lambda x: x.title(),
        horizontal=True, key="lifecycle_filter",
    )
    lifecycle_df = classify_theme_lifecycle(df_exploded, sentiment_filter=sentiment_filter)
    if lifecycle_df is None or lifecycle_df.empty:
        st.info(f"Not enough {sentiment_filter} data for lifecycle analysis (need at least 4 months).")
        return
    for _, row in lifecycle_df.iterrows():
        cols = st.columns([2, 1.5, 2, 2])
        with cols[0]: st.markdown(f"**{row['Theme']}**")
        with cols[1]:
            st.markdown(
                f'<span style="background:{row["Color"]}20; color:{row["Color"]}; '
                f'padding:3px 10px; border-radius:10px; font-weight:600; font-size:13px;">'
                f'{row["Lifecycle"]}</span>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(f"Avg/mo: **{row['First Half Avg']} → {row['Second Half Avg']}**")
        with cols[3]:
            clr = "#10B981" if row["Change %"] > 0 else "#EF4444" if row["Change %"] < 0 else "#6B7280"
            st.markdown(
                f'<span style="color:{clr}; font-weight:600;">{row["Change %"]:+.1f}%</span>',
                unsafe_allow_html=True)
        st.markdown("<hr style='margin:0.3rem 0; border-color:#F3F4F6;'>", unsafe_allow_html=True)


def render_emergent_themes(df_exploded):
    if "date" not in df_exploded.columns:
        return
    try:
        d = df_exploded.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[d["date"].notna()]
        if d.empty:
            return
        theme_time = (d.groupby([pd.Grouper(key="date", freq="ME"), "Theme"])
                       .size().reset_index(name="count"))
        months = sorted(theme_time["date"].unique())
        if len(months) < 2:
            st.info("Need at least 2 months for momentum analysis.")
            return
        curr = theme_time[theme_time["date"] == months[-1]].set_index("Theme")
        prev = theme_time[theme_time["date"] == months[-2]].set_index("Theme")
        emer = curr[["count"]].join(prev[["count"]], lsuffix="_curr", rsuffix="_prev", how="outer").fillna(0)
        emer["Change"] = (emer["count_curr"] - emer["count_prev"]).astype(int)
        emer["count_curr"] = emer["count_curr"].astype(int)
        emer["count_prev"] = emer["count_prev"].astype(int)
        st.markdown(f"**Momentum: {months[-2].strftime('%b %Y')} → {months[-1].strftime('%b %Y')}**")
        st.dataframe(
            emer.sort_values("Change", ascending=False)
            [["count_prev","count_curr","Change"]]
            .rename(columns={"count_prev":"Prev Month","count_curr":"Current Month","Change":"Momentum"}),
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Could not compute emergent themes: {e}")


def render_word_cloud(df, sentiment):
    if not WORDCLOUD_AVAILABLE:
        st.info("Word cloud requires: pip install wordcloud")
        return
    df_f = df[df["predicted_sentiment"].apply(normalize_sentiment_label) == sentiment]
    if df_f.empty:
        st.info(f"No {sentiment} reviews for word cloud.")
        return
    text = " ".join(df_f["clean_text"].fillna("").astype(str).tolist())
    if not text.strip():
        return
    extra_stop = {"review","user","star","stars","https","http","amp","just","like","im",
                  "ive","got","get","go","going","went","told","said","would","could","still"}
    stop_words = set(sk_text.ENGLISH_STOP_WORDS) | extra_stop
    color_map  = {"positive":"Greens","negative":"Reds","neutral":"Greys"}
    try:
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap=color_map.get(sentiment,"viridis"),
                       stopwords=stop_words, max_words=80,
                       relative_scaling=0.5, min_font_size=10).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not generate word cloud: {e}")


def page_methodology():
    st.markdown("## About & Methodology")

    ### Credits
    st.markdown(
        '<div style="background:#f0f4ff; border-left:4px solid #1e3a5f; '
        'padding:18px 22px; border-radius:8px; margin-bottom:16px;">'
        '<p style="font-size:17px; font-weight:700; color:#1e3a5f; margin:0 0 6px 0;">'
        'Project By Group 5</p>'
        '<p style="font-size:15px; color:#374151; margin:0 0 10px 0;">'
        'Christian East &nbsp;·&nbsp; Birajman Tamang &nbsp;·&nbsp; Kelsang Yonjan</p>'
        '<p style="font-size:14px; color:#6b7280; margin:0 0 4px 0;">'
        '<strong>CSCI 491</strong></p>'
        '<p style="font-size:14px; color:#6b7280; margin:0;">'
        'Special thanks to <strong>Dr. Jennifer Lavergne</strong> and '
        '<strong>Dr. Lasang Tamang</strong></p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    ### What it does
    st.markdown("### What This Platform Does")
    st.markdown(
        """
        The Customer Feedback Intelligence Platform analyses customer reviews using two AI systems:

        - **Sentiment Classification** — predicts whether a review is positive, negative, or neutral
        - **Theme Extraction** — identifies which business topics each review is about
        """
    )

    ### Sentiment Model
    st.markdown("---")
    st.markdown("### Sentiment Model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Algorithm:** Logistic Regression")
        st.markdown("**Vectorizer:** TF-IDF (5,000 features, 1–2 word phrases)")
        st.markdown("**Training split:** 80% train / 20% test")
    with col2:
        st.markdown("**Classes:** Positive · Negative · Neutral/Mixed")
        st.markdown("**Mixed-signal detection:** contrast words + dual polarity vocabulary")
        st.markdown("**Confidence score:** model's probability for its predicted class")

    st.markdown(
        """
        **Confidence score guide:**
        | Score | Meaning |
        |---|---|
        | 90–100% | Very certain |
        | 70–89% | Confident |
        | 60–69% | Moderate |
        | Below 60% | Low — flagged in Outliers tab |
        """
    )

    ### Theme Extraction
    st.markdown("---")
    st.markdown("### Theme Extraction")
    st.markdown(
        """
        Reviews are sent in batches to a locally-running LLM (Gemma 3 4B via Ollama).
        The model assigns 1–3 themes per review from the approved list only — any invented
        themes are rejected and retried up to 5 times. Results are cached so the same CSV
        skips LLM processing on re-upload.

        **The 8 themes:**
        """
    )
    theme_descriptions = {
        "Product Quality":      "Food, drink, or item quality and standards",
        "Product Availability": "Out of stock items or limited menu",
        "Customer Service":     "Staff attitude, helpfulness, complaint handling",
        "Speed of Service":     "Wait times, queue length, order delays",
        "Store Environment":    "Cleanliness, atmosphere, seating, parking",
        "Price & Value":        "Cost, affordability, value for money",
        "Digital & Rewards":    "App, online ordering, loyalty points",
        "Policies & Safety":    "Return policies, hygiene, health precautions",
    }
    for theme, desc in theme_descriptions.items():
        st.markdown(
            f'<div style="background:#f8f9fa; border-left:3px solid #1e3a5f; '
            f'padding:8px 14px; border-radius:4px; margin-bottom:5px;">'
            f'<strong style="color:#1e3a5f;">{theme}</strong> — '
            f'<span style="color:#374151; font-size:14px;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    ### Tech Stack & Privacy
    st.markdown("---")
    st.markdown("### Technology Stack")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("**ML**\nscikit-learn · Logistic Regression · TF-IDF")
    with tc2:
        st.markdown("**LLM**\nOllama · Gemma 3 4B")
    with tc3:
        st.markdown("**Dashboard**\nStreamlit · Plotly · ReportLab · pandas")

    st.markdown("---")
    st.caption("Customer Feedback Intelligence Platform — CSCI 491 · Group 5")


### 
#  MAIN
### 

def main():
    st.markdown(METRIC_CSS, unsafe_allow_html=True)
    st.markdown("# Customer Feedback Intelligence Platform")
    st.markdown("Upload a review CSV in the sidebar, click **Run Analysis**, "
                "then navigate results using the tabs below.")

    model, vectorizer, train_accuracy = load_or_train_model()
    if model is None or vectorizer is None:
        st.error("Model could not be loaded or trained.")
        return
    if train_accuracy is not None:
        st.success(f"Model trained — accuracy: {train_accuracy:.4f}")

    st.sidebar.header("Upload & Run")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV for analysis", type="csv",
        help="CSV must have a 'text', 'clean_text', or 'raw_text' column.")
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run Analysis", use_container_width=True)
    if st.sidebar.button("Reset / Clear Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if uploaded_file and run_button:
        try:
            df = pd.read_csv(uploaded_file)
            df = preprocess_reviews(df)
            df = predict_reviews(df, model, vectorizer)
            cache_key = f"themes_{_csv_hash(df)}"
            if cache_key in st.session_state:
                df["themes"] = st.session_state[cache_key]
                st.info("Themes loaded from cache — skipping LLM extraction.")
            else:
                df = extract_themes(df, THEMES)
                st.session_state[cache_key] = df["themes"].copy()
            st.session_state.analyzed_df = df
        except Exception as exc:
            st.error(f"Error during analysis: {exc}")

    if "analyzed_df" in st.session_state:
        df = st.session_state.analyzed_df
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Overview", "Positive Reviews", "Neutral Reviews",
            "Negative Reviews", "Theme Extraction", "Trends & Insights",
            "Outliers", "About",
        ])
        with tab1: page_overview(df)
        with tab2: page_positive(df)
        with tab3: page_neutral(df)
        with tab4: page_negative(df)
        with tab5: page_themes(df)
        with tab6: page_trends(df)
        with tab7: page_outliers(df)
        with tab8: page_methodology()

    elif not uploaded_file:
        st.info("Upload a CSV file in the sidebar and click Run Analysis to get started.")
        st.markdown("---")
        page_methodology()


if __name__ == "__main__":
    main()

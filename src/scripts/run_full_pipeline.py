import argparse
import subprocess
import sys
import io
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

# PDF export imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "sample_data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_FILE = OUTPUT_DIR / "sentiment_model.pkl"
VECTORIZER_FILE = OUTPUT_DIR / "tfidf_vectorizer.pkl"

THEMES = [
    "Product Quality",        # Item quality, taste, order accuracy
    "Product Availability",   # Stock availability
    "Customer Service",       # Staff attitude, friendliness, support, issue resolution
    "Speed of Service",       # Wait times, drive-thru speed, delivery speed, queues
    "Store Environment",      # Cleanliness, atmosphere, lighting, parking, location
    "Price & Value",          # Cost, affordability, value for money
    "Digital & Rewards",      # App functionality, website, online ordering, loyalty points
    "Policies & Safety",      # Return policies, health precautions, hygiene standards
]

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")


# ── LLM ───────────────────────────────────────────────────────────────────────

def call_llm(prompt, model="gemma2:9b"):
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise e


# ── Model training / loading ──────────────────────────────────────────────────

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


# ── Review preprocessing & prediction ────────────────────────────────────────

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


# ── Theme extraction ──────────────────────────────────────────────────────────

def build_prompt(batch, themes):
    numbered = "\n".join([
        f"Review {i+1}:\n{str(r)[:250]}"
        for i, r in enumerate(batch)
    ])
    prompt = f"""You are a professional theme classifier for customer reviews.
You work for a Feedback Intelligence Platform that analyzes reviews.

Available themes: {themes}

RULES:
- Only assign themes from the available list above. NEVER invent or create your own themes.
- Every review MUST have at least one theme assigned. If a review is vague, pick the single closest matching theme. Do not leave the array empty.
- Return ONLY a valid JSON dictionary where the keys are the Review Numbers ("1", "2", etc.) and the values are arrays of themes.
- You MUST generate exactly {len(batch)} keys in your dictionary, one for every review provided.
- Do not provide any extra explanation or text outside of the JSON block.

Example for 3 reviews:
{{
  "1": ["Customer Service", "Product Quality"],
  "2": ["Speed of Service"],
  "3": ["Price & Value"]
}}

Reviews to classify:
{numbered}

Return ONLY valid JSON.
"""
    return prompt


def extract_themes_with_retry(batch_info, themes_list, max_retries=5):
    batch_idx, batch = batch_info
    prompt = build_prompt(batch, themes_list)

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(prompt)

            first_brace = raw.find("{")
            first_bracket = raw.find("[")
            start = min(i for i in [first_brace, first_bracket] if i != -1) if any(i != -1 for i in [first_brace, first_bracket]) else -1

            last_brace = raw.rfind("}")
            last_bracket = raw.rfind("]")
            end = max(last_brace, last_bracket) + 1

            if start == -1 or end <= start:
                raise ValueError("Warning: LLM output did not contain valid JSON wrapped in {} or []")
            else:
                clean = raw[start:end]
                try:
                    parsed_data = json.loads(clean)
                except json.JSONDecodeError:
                    try:
                        parsed_data = ast.literal_eval(clean)
                    except Exception:
                        try:
                            if clean.strip().startswith("{") and clean.strip().endswith("}"):
                                parsed_data = json.loads(f"[{clean}]")
                            else:
                                raise ValueError()
                        except Exception:
                            raise ValueError(f"Could not parse response dictionary: {clean}")

                parsed_dict = {}
                if isinstance(parsed_data, list):
                    for item in parsed_data:
                        if isinstance(item, dict):
                            parsed_dict.update(item)
                elif isinstance(parsed_data, dict):
                    parsed_dict = parsed_data
                else:
                    raise ValueError("Response is not a valid structured JSON.")

                themes = []
                for idx in range(1, len(batch) + 1):
                    key = str(idx)
                    idx_themes = parsed_dict.get(key, [])
                    if not isinstance(idx_themes, list):
                        idx_themes = [idx_themes]
                    if not idx_themes or idx_themes == []:
                        idx_themes = ["Customer Service"]
                    themes.append(idx_themes)

            if len(themes) != len(batch):
                raise ValueError(f"Batch mismatch: LLM returned {len(themes)} exact theme arrays, but there are {len(batch)} reviews.")

            validated_themes = []
            for theme_list in themes:
                valid_for_review = []
                safe_themes = [str(list(t.values())[0]) if isinstance(t, dict) and t else str(t) for t in theme_list]
                for st_theme in safe_themes:
                    st_clean = st_theme.strip().lower()
                    for real_theme in themes_list:
                        if st_clean == real_theme.lower():
                            if real_theme not in valid_for_review:
                                valid_for_review.append(real_theme)
                            break
                if not valid_for_review:
                    raise ValueError(f"Hallucination detected: The LLM returned '{theme_list}' which is not in the allowed list: {themes_list}")
                validated_themes.append(valid_for_review)

            return batch_idx, batch, validated_themes, "success"

        except Exception as e:
            print(f"Batch {batch_idx} attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)

    return batch_idx, batch, None, "failed"


def extract_themes(df, themes_list, batch_size=30, max_workers=2):
    reviews = df["clean_text"].fillna("").tolist()
    batches = [(i, reviews[i:i + batch_size]) for i in range(0, len(reviews), batch_size)]

    successful_results = []
    failed_results = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_batches = len(batches)
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_themes_with_retry, b, themes_list): b
            for b in batches
        }

        for future in as_completed(futures):
            try:
                batch_idx, batch, batch_themes, status = future.result(timeout=180)
            except TimeoutError:
                b = futures[future]
                batch_idx, batch = b
                status = "failed"
                batch_themes = [["FAILED (TIMEOUT)"]] * len(batch)
                print(f"Batch {batch_idx} timed out after 3 minutes and was marked as failed.")
            except Exception as e:
                b = futures[future]
                batch_idx, batch = b
                status = "failed"
                batch_themes = [["FAILED (ERROR)"]] * len(batch)
                print(f"Batch {batch_idx} crashed unexpectedly: {e}")

            if status == "success":
                for i, (review, valid_themes) in enumerate(zip(batch, batch_themes)):
                    joined_themes = ", ".join(valid_themes)
                    successful_results.append({
                        "original_idx": batch_idx + i,
                        "themes": joined_themes
                    })
            else:
                for i, review in enumerate(batch):
                    failed_results.append({
                        "original_idx": batch_idx + i,
                        "themes": "FAILED"
                    })

            completed_batches += 1
            if completed_batches % max(1, (total_batches // 100)) == 0 or completed_batches == total_batches:
                progress_bar.progress(completed_batches / total_batches)
                status_text.text(f"Processed review batch {completed_batches}/{total_batches}. Please wait, local LLM parsing is intensive...")

    themes_lookup = {r["original_idx"]: r["themes"] for r in successful_results + failed_results}
    df["themes"] = [themes_lookup.get(i, "FAILED") for i in range(len(df))]

    initial_len = len(df)
    df = df[~df["themes"].str.contains("FAILED", na=False)]
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} reviews because the LLM repeatedly hallucinated or timed out.")

    progress_bar.progress(1.0)
    status_text.text("Theme extraction complete!")

    return df


# ── PDF export ────────────────────────────────────────────────────────────────

_CLR_POS    = colors.HexColor("#16a34a")
_CLR_NEG    = colors.HexColor("#dc2626")
_CLR_NEU    = colors.HexColor("#6b7280")
_CLR_HEADER = colors.HexColor("#1e3a5f")
_CLR_RULE   = colors.HexColor("#e5e7eb")
_CLR_BG     = colors.HexColor("#f0f4ff")
_COLOUR_MAP = {
    "positive":      "#16a34a",
    "negative":      "#dc2626",
    "neutral":       "#6b7280",
    "neutral/mixed": "#9ca3af",
}


def _build_pdf_styles():
    base = getSampleStyleSheet()
    title_s = ParagraphStyle("ReportTitle", parent=base["Title"],
        fontSize=22, textColor=_CLR_HEADER, spaceAfter=4, fontName="Helvetica-Bold")
    subtitle_s = ParagraphStyle("ReportSubtitle", parent=base["Normal"],
        fontSize=10, textColor=colors.HexColor("#6b7280"), spaceAfter=16)
    h2_s = ParagraphStyle("H2", parent=base["Heading2"],
        fontSize=13, textColor=_CLR_HEADER, spaceBefore=18, spaceAfter=6, fontName="Helvetica-Bold")
    body_s = ParagraphStyle("Body", parent=base["Normal"],
        fontSize=9, leading=14, textColor=colors.HexColor("#374151"))
    small_s = ParagraphStyle("Small", parent=base["Normal"],
        fontSize=8, leading=12, textColor=colors.HexColor("#6b7280"))
    return title_s, subtitle_s, h2_s, body_s, small_s


def _fig_to_rl_image(fig, width_inch=6.5, height_inch=3.2):
    png_bytes = fig.to_image(format="png", width=int(width_inch * 100),
                              height=int(height_inch * 100), scale=2)
    buf = io.BytesIO(png_bytes)
    return RLImage(buf, width=width_inch * inch, height=height_inch * inch)


def _pdf_sentiment_pie(df):
    counts = df["predicted_sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(counts, names="sentiment", values="count",
                 color="sentiment", color_discrete_map=_COLOUR_MAP,
                 title="Overall Sentiment Distribution")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Helvetica", size=11), title_font_size=13)
    return _fig_to_rl_image(fig, width_inch=3.0, height_inch=2.8)


def _pdf_theme_bar(df):
    exploded = df["themes"].str.split(r",\s*").explode().str.strip()
    exploded = exploded[~exploded.isin(["FAILED", "", "NOT PROCESSED"])]
    theme_counts = exploded.value_counts().head(8).reset_index()
    theme_counts.columns = ["Theme", "Count"]
    fig = px.bar(theme_counts, x="Count", y="Theme", orientation="h",
                 title="Top Themes by Mention Volume",
                 color="Count", color_continuous_scale="Blues")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Helvetica", size=10), title_font_size=13,
                      yaxis=dict(categoryorder="total ascending"),
                      coloraxis_showscale=False)
    return _fig_to_rl_image(fig, width_inch=3.8, height_inch=2.8)


def _pdf_time_trend(df):
    if "date" not in df.columns:
        return None
    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        time_df = (df.groupby([pd.Grouper(key="date", freq="ME"), "predicted_sentiment"])
                   .size().reset_index(name="count"))
        if len(time_df) <= 1:
            return None
        fig = px.line(time_df, x="date", y="count", color="predicted_sentiment",
                      title="Monthly Sentiment Trend", color_discrete_map=_COLOUR_MAP)
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20),
                          paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(family="Helvetica", size=10), title_font_size=13,
                          legend_title_text="")
        return _fig_to_rl_image(fig, width_inch=6.5, height_inch=2.8)
    except Exception:
        return None


def _pdf_kpi_table(df):
    total    = len(df)
    pos      = int((df["predicted_sentiment"] == "positive").sum())
    neg      = int((df["predicted_sentiment"] == "negative").sum())
    mixed    = int(df["predicted_sentiment"].isin(["neutral", "neutral/mixed"]).sum())
    avg_conf = df["confidence"].mean() if "confidence" in df.columns else None
    flagged  = int((df["is_mixed"] == True).sum()) if "is_mixed" in df.columns else 0

    rows = [
        ["Metric", "Value", "Share"],
        ["Total reviews analysed",        f"{total:,}",   "100%"],
        ["Positive",                       f"{pos:,}",    f"{pos/total*100:.1f}%"],
        ["Negative",                       f"{neg:,}",    f"{neg/total*100:.1f}%"],
        ["Neutral / Mixed",                f"{mixed:,}",  f"{mixed/total*100:.1f}%"],
        ["Mixed-signal reviews flagged",   f"{flagged:,}", f"{flagged/total*100:.1f}%"],
    ]
    if avg_conf is not None:
        rows.append(["Average model confidence", f"{avg_conf*100:.1f}%", "—"])

    tbl = Table(rows, colWidths=[3.2*inch, 1.5*inch, 1.5*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _CLR_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _CLR_HEADER),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("TEXTCOLOR",     (0, 1), (-1, -1), colors.HexColor("#374151")),
        ("TEXTCOLOR",     (2, 2), (2, 2),   _CLR_POS),
        ("TEXTCOLOR",     (2, 3), (2, 3),   _CLR_NEG),
        ("GRID",          (0, 0), (-1, -1), 0.4, _CLR_RULE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    return tbl


def _pdf_theme_sentiment_table(df):
    if "themes" not in df.columns:
        return None
    df_exp = df.assign(Theme=df["themes"].str.split(r",\s*")).explode("Theme")
    df_exp["Theme"] = df_exp["Theme"].str.strip()
    df_exp = df_exp[~df_exp["Theme"].isin(["FAILED", "", "NOT PROCESSED"])]
    if df_exp.empty:
        return None

    pivot = pd.crosstab(df_exp["Theme"], df_exp["predicted_sentiment"])
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).head(8)
    sentiment_cols = [c for c in ["positive", "negative", "neutral", "neutral/mixed"] if c in pivot.columns]

    header = ["Theme"] + [c.capitalize() for c in sentiment_cols] + ["Total"]
    rows = [header]
    for theme, row in pivot.iterrows():
        data_row = [str(theme)]
        for sc in sentiment_cols:
            val = int(row.get(sc, 0))
            pct = val / row["Total"] * 100 if row["Total"] > 0 else 0
            data_row.append(f"{val} ({pct:.0f}%)")
        data_row.append(str(int(row["Total"])))
        rows.append(data_row)

    col_w = [2.5*inch] + [1.0*inch]*len(sentiment_cols) + [0.8*inch]
    tbl = Table(rows, colWidths=col_w)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0), _CLR_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _CLR_HEADER),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("GRID",          (0, 0), (-1, -1), 0.4, _CLR_RULE),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
    ]
    for i, sc in enumerate(sentiment_cols):
        col_idx = i + 1
        if sc == "positive":
            style.append(("TEXTCOLOR", (col_idx, 0), (col_idx, 0), _CLR_POS))
        elif sc == "negative":
            style.append(("TEXTCOLOR", (col_idx, 0), (col_idx, 0), _CLR_NEG))
    tbl.setStyle(TableStyle(style))
    return tbl


def _pdf_sample_reviews_table(df):
    text_col = next((c for c in ["text", "raw_text", "clean_text"] if c in df.columns), None)
    if text_col is None:
        return None

    pos_sample = (df[df["predicted_sentiment"] == "positive"].nlargest(3, "confidence")
                  if "confidence" in df.columns
                  else df[df["predicted_sentiment"] == "positive"].head(3))
    neg_sample = (df[df["predicted_sentiment"] == "negative"].nlargest(3, "confidence")
                  if "confidence" in df.columns
                  else df[df["predicted_sentiment"] == "negative"].head(3))
    sample = pd.concat([pos_sample, neg_sample])

    rows = [["Sentiment", "Review excerpt", "Confidence"]]
    for _, row in sample.iterrows():
        excerpt = str(row[text_col])[:160].strip()
        if len(str(row[text_col])) > 160:
            excerpt += "…"
        conf = f"{row['confidence']*100:.0f}%" if "confidence" in row else "—"
        rows.append([row["predicted_sentiment"].capitalize(), excerpt, conf])

    tbl = Table(rows, colWidths=[1.0*inch, 4.8*inch, 0.9*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _CLR_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _CLR_HEADER),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("GRID",          (0, 0), (-1, -1), 0.4, _CLR_RULE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("WORDWRAP",      (1, 1), (1, -1),  True),
    ]))
    return tbl


def _pdf_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#9ca3af"))
    canvas.drawString(0.75*inch, 0.5*inch,
                      "Customer Feedback Intelligence Platform  |  Confidential")
    canvas.drawRightString(letter[0] - 0.75*inch, 0.5*inch, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(df: pd.DataFrame, report_title: str = "Feedback Intelligence Report") -> bytes:
    """
    Build a multi-page PDF report from the analysed dataframe and return raw bytes.
    Pass the result directly to st.download_button(data=...).
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.85*inch, bottomMargin=0.75*inch)

    title_s, subtitle_s, h2_s, body_s, small_s = _build_pdf_styles()
    story = []

    # Cover
    story.append(Paragraph(report_title, title_s))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  |  "
        f"{len(df):,} reviews analysed",
        subtitle_s,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=_CLR_RULE, spaceAfter=12))

    # Section 1 — KPI table
    story.append(Paragraph("1. Summary Metrics", h2_s))
    story.append(_pdf_kpi_table(df))
    story.append(Spacer(1, 14))

    # Section 2 — Charts
    story.append(Paragraph("2. Sentiment & Theme Overview", h2_s))
    has_themes = "themes" in df.columns
    try:
        pie_img = _pdf_sentiment_pie(df)
        theme_img = _pdf_theme_bar(df) if has_themes else None
        if theme_img:
            chart_row = Table([[pie_img, theme_img]], colWidths=[3.2*inch, 4.0*inch])
            chart_row.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ]))
            story.append(chart_row)
        else:
            story.append(pie_img)
    except Exception as e:
        story.append(Paragraph(
            f"Charts could not be generated: {e}. "
            "Ensure kaleido is installed: pip install kaleido",
            small_s,
        ))
    story.append(Spacer(1, 10))

    # Section 3 — Time trend (optional)
    trend_img = _pdf_time_trend(df)
    section_offset = 0
    if trend_img:
        story.append(Paragraph("3. Monthly Sentiment Trend", h2_s))
        story.append(trend_img)
        story.append(Spacer(1, 10))
        section_offset = 1

    # Section 4 — Theme breakdown table
    theme_tbl = _pdf_theme_sentiment_table(df)
    if theme_tbl:
        story.append(Paragraph(f"{3 + section_offset}. Theme Sentiment Breakdown", h2_s))
        story.append(Paragraph(
            "Count (%) of reviews per sentiment class for the top 8 themes.", body_s))
        story.append(Spacer(1, 6))
        story.append(theme_tbl)
        story.append(Spacer(1, 10))

    # Section 5 — Sample verbatims
    sample_tbl = _pdf_sample_reviews_table(df)
    if sample_tbl:
        story.append(PageBreak())
        story.append(Paragraph(f"{4 + section_offset}. Sample Review Verbatims", h2_s))
        story.append(Paragraph(
            "Highest-confidence positive and negative reviews from this dataset.", body_s))
        story.append(Spacer(1, 6))
        story.append(sample_tbl)

    # Disclaimer footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_CLR_RULE, spaceAfter=6))
    story.append(Paragraph(
        "This report was generated automatically by the Customer Feedback Intelligence Platform. "
        "Sentiment predictions are produced by a Logistic Regression model trained on labelled review data. "
        "Theme labels are assigned by a locally-hosted LLM (Ollama / Gemma). "
        "Results should be reviewed alongside the raw data before business decisions are made.",
        small_s,
    ))

    doc.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    return buf.getvalue()


# ── Dashboard rendering ───────────────────────────────────────────────────────

def render_dashboard(df):
    st.subheader("Prediction Summary")
    st.write(df[["predicted_sentiment", "confidence", "is_mixed", "themes"]].head(10))

    sentiment_counts = df["predicted_sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        title="Predicted Sentiment Distribution",
        color="sentiment",
        color_discrete_map={"positive": "green", "neutral": "gray", "neutral/mixed": "gray", "negative": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)

    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            time_df = (
                df.groupby([pd.Grouper(key="date", freq="ME"), "predicted_sentiment"])
                .size()
                .reset_index(name="count")
            )
            if len(time_df) > 1:
                fig = px.line(
                    time_df,
                    x="date",
                    y="count",
                    color="predicted_sentiment",
                    title="Overall Sentiment Over Time (Monthly)",
                    color_discrete_map={"positive": "green", "neutral": "gray", "neutral/mixed": "gray", "negative": "red"}
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Date column found but could not be parsed as datetime.")

    st.subheader("Top Extracted Themes")

    if "themes" in df.columns:
        exploded = df["themes"].str.split(",\\s*").explode().str.strip()
        exploded = exploded[~exploded.isin(["FAILED", "", "NOT PROCESSED"])]

        theme_summary_series = exploded.value_counts().head(20)
        theme_summary = theme_summary_series.reset_index()
        theme_summary.columns = ["Theme", "Count"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(theme_summary, use_container_width=True)
        with col2:
            fig = px.bar(
                theme_summary,
                x="Count",
                y="Theme",
                orientation='h',
                title="Most Common Review Themes",
                color="Count",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Theme Sentiment Breakdown & Distribution")

        df_exploded = df.assign(Theme=df['themes'].str.split(",\\s*")).explode('Theme')
        df_exploded['Theme'] = df_exploded['Theme'].str.strip()
        df_exploded = df_exploded[~df_exploded['Theme'].isin(["FAILED", "", "NOT PROCESSED"])]
        df_exploded = df_exploded.reset_index(drop=True)

        if not df_exploded.empty:
            st.markdown("**Interactive Theme Sentiment Distribution**")
            unique_themes = sorted(df_exploded['Theme'].unique().tolist())
            selected_theme = st.selectbox("Select a Theme:", unique_themes)

            theme_data = df_exploded[df_exploded['Theme'] == selected_theme]
            theme_sentiment_counts = theme_data['predicted_sentiment'].value_counts().reset_index()
            theme_sentiment_counts.columns = ["sentiment", "count"]

            fig_dist = px.pie(
                theme_sentiment_counts,
                names="sentiment",
                values="count",
                title=f"Sentiment Distribution for '{selected_theme}'",
                color="sentiment",
                color_discrete_map={"positive": "green", "neutral": "gray", "neutral/mixed": "gray", "negative": "red"}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("**Detailed Data: Sentiment Breakdown by Theme (Counts & Percentages)**")
            pivot_df = pd.crosstab(df_exploded['Theme'], df_exploded['predicted_sentiment'],
                                   margins=True, margins_name="Total")

            cols_to_percent = [col for col in ["positive", "negative", "neutral", "neutral/mixed"]
                               if col in pivot_df.columns]
            for col in cols_to_percent:
                pivot_df[col + " (%)"] = (pivot_df[col] / pivot_df["Total"] * 100).round(1)

            ordered_cols = []
            for col in cols_to_percent:
                ordered_cols.extend([col, col + " (%)"])
            if "Total" in pivot_df.columns:
                ordered_cols.append("Total")

            st.dataframe(pivot_df[ordered_cols], use_container_width=True)

            if "date" in df.columns:
                st.markdown("---")
                st.subheader("Time-Based & Emergent Trends")
                st.write("Understand which topics are gaining or losing momentum.")

                try:
                    theme_time = (
                        df_exploded.groupby([pd.Grouper(key="date", freq="ME"), "Theme"])
                        .size()
                        .reset_index(name="count")
                    )

                    if len(theme_time['date'].unique()) > 1:
                        col_tr1, col_tr2 = st.columns([1, 1])

                        with col_tr1:
                            st.markdown(f"**Monthly Volume Trend for '{selected_theme}'**")
                            sel_theme_time = theme_time[theme_time["Theme"] == selected_theme]
                            fig_trend = px.bar(
                                sel_theme_time,
                                x="date",
                                y="count",
                                title=f"Review mentions of '{selected_theme}' alone",
                                labels={"date": "Month", "count": "Mentions"}
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)

                        with col_tr2:
                            st.markdown("**Emergent Themes (Most Recent Month)**")
                            months = sorted(theme_time['date'].unique())
                            curr_month = months[-1]
                            prev_month = months[-2]

                            curr_df = theme_time[theme_time['date'] == curr_month].set_index("Theme")
                            prev_df = theme_time[theme_time['date'] == prev_month].set_index("Theme")

                            emergent_df = curr_df[['count']].join(
                                prev_df[['count']],
                                lsuffix='_curr',
                                rsuffix='_prev',
                                how='outer'
                            ).fillna(0)

                            emergent_df['Change'] = emergent_df['count_curr'] - emergent_df['count_prev']
                            rising_themes = emergent_df.sort_values(by='Change', ascending=False)

                            st.write(f"Change in conversational volume between **{prev_month.strftime('%b %Y')}** and **{curr_month.strftime('%b %Y')}**:")
                            st.dataframe(
                                rising_themes[['count_prev', 'count_curr', 'Change']]
                                .rename(columns={'count_prev': 'Prev. Mentions',
                                                 'count_curr': 'Current Mentions',
                                                 'Change': 'Momentum'}),
                                use_container_width=True
                            )
                    else:
                        st.info("The dataset spans less than a full month. Trend momentum cannot be established.")
                except Exception as e:
                    st.warning(f"Could not calculate emergent theme trends: {e}")

            st.markdown("---")
            st.subheader("Deep Dive: What are customers actually saying?")
            st.write("Understand the specific subtopics and read actual reviews driving the sentiment for a theme.")

            col_dd1, col_dd2 = st.columns(2)
            with col_dd1:
                dd_theme = st.selectbox("Select Theme for Deep Dive:", unique_themes, key="dd_theme")
            with col_dd2:
                dd_sentiment = st.selectbox("Select Sentiment:", ["negative", "positive", "neutral", "neutral/mixed"], key="dd_sentiment")

            dd_data = df_exploded[(df_exploded['Theme'] == dd_theme) & (df_exploded['predicted_sentiment'] == dd_sentiment)]

            if dd_data.empty:
                st.info(f"No {dd_sentiment} reviews found for '{dd_theme}'.")
            else:
                dd_col1, dd_col2 = st.columns([1, 1])
                with dd_col1:
                    st.markdown(f"**Top Phrases driving {dd_sentiment} sentiment in {dd_theme}**")
                    try:
                        sent_data = df_exploded[df_exploded['predicted_sentiment'] == dd_sentiment]
                        theme_docs = sent_data.groupby('Theme')['clean_text'].apply(
                            lambda texts: ' '.join(texts.dropna())).to_dict()

                        if dd_theme in theme_docs and len(theme_docs[dd_theme].strip()) > 0:
                            extra_stop = {"review", "user", "star", "stars", "https", "http", "amp", "just", "like", "im"}
                            stop_words = list(set(sk_text.ENGLISH_STOP_WORDS) | extra_stop)

                            corpus_themes = list(theme_docs.keys())
                            corpus_texts = list(theme_docs.values())

                            tv = TfidfVectorizer(ngram_range=(2, 3), stop_words=stop_words)
                            tfidf_matrix = tv.fit_transform(corpus_texts)

                            theme_idx = corpus_themes.index(dd_theme)
                            feature_names = tv.get_feature_names_out()

                            tfidf_dense = tfidf_matrix.toarray()
                            theme_scores = tfidf_dense[theme_idx]

                            is_primary_theme = (tfidf_dense.argmax(axis=0) == theme_idx)
                            exclusive_scores = theme_scores * is_primary_theme

                            top_indices = exclusive_scores.argsort()[-10:][::-1]
                            top_phrases = [feature_names[i] for i in top_indices if exclusive_scores[i] > 0]

                            phrase_counts = []
                            for p in top_phrases:
                                count = theme_docs[dd_theme].count(p)
                                phrase_counts.append(count)

                            phrase_df = pd.DataFrame({"Phrase": top_phrases, "Count": phrase_counts})
                            phrase_df = phrase_df.sort_values(by="Count", ascending=True)

                            color_map = {"positive": "green", "neutral": "gray", "neutral/mixed": "gray", "negative": "red"}
                            fig_phrases = px.bar(phrase_df, x="Count", y="Phrase", orientation='h')
                            fig_phrases.update_layout(xaxis_title="Mentions", yaxis_title="")
                            fig_phrases.update_traces(marker_color=color_map.get(dd_sentiment, "blue"))
                            st.plotly_chart(fig_phrases, use_container_width=True)
                        else:
                            st.info("Not enough valid text to extract phrases.")
                    except ValueError:
                        st.info("Not enough words to extract meaningful multi-word phrases. Try another theme or sentiment.")
                    except Exception as e:
                        st.warning(f"Could not extract phrases: {e}")

                with dd_col2:
                    header_col, btn_col = st.columns([2, 1])
                    with header_col:
                        st.markdown(f"**Sample {dd_sentiment.title()} Reviews**")
                    with btn_col:
                        st.button("🔄 Refresh", help="Load different random reviews", use_container_width=True)

                    sample_reviews = dd_data.sample(min(5, len(dd_data)))
                    text_col = ("text" if "text" in dd_data.columns
                                else "raw_text" if "raw_text" in dd_data.columns
                                else "clean_text")
                    for review_text in sample_reviews[text_col].tolist():
                        st.info(f'"{review_text}"')

    else:
        st.warning("No themes column found to display statistics.")

    st.download_button(
        label="Download analyzed data as CSV",
        data=df.to_csv(index=False),
        file_name="analysis_results.csv",
        mime="text/csv",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("Sentiment Analysis + Theme Extraction Pipeline")
    st.markdown(
        "This app trains the sentiment model automatically from sample data, "
        "accepts a review CSV upload, predicts sentiment, extracts themes, "
        "and shows charts in a single interface."
    )

    model, vectorizer, train_accuracy = load_or_train_model()
    if model is None or vectorizer is None:
        st.error("Model could not be loaded or trained.")
        return

    if train_accuracy is not None:
        st.success(f"Model trained successfully with accuracy {train_accuracy:.4f}")

    # ── Sidebar controls ───────────────────────────────────────────────────────
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV for analysis",
        type="csv",
        help="Your CSV should include 'text', 'clean_text', or 'raw_text'."
    )

    run_button = st.sidebar.button("Run analysis")

    if st.sidebar.button("Reset / Clear Data"):
        st.session_state.clear()
        st.success("App cache cleared! You can start fresh.")
        return

    # ── PDF export (only visible after analysis is complete) ──────────────────
    if "analyzed_df" in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Export Report**")

        custom_title = st.sidebar.text_input(
            "Report title",
            value="Feedback Intelligence Report",
            help="This title appears on the PDF cover page",
        )

        if st.sidebar.button("Generate PDF Report"):
            with st.spinner("Building PDF report — this takes a few seconds..."):
                try:
                    pdf_bytes = generate_pdf_report(
                        st.session_state.analyzed_df,
                        report_title=custom_title,
                    )
                    st.sidebar.download_button(
                        label="⬇ Download PDF",
                        data=pdf_bytes,
                        file_name=f"feedback_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.sidebar.success("PDF ready — click Download PDF above.")
                except Exception as e:
                    st.sidebar.error(f"PDF generation failed: {e}")
                    st.sidebar.info("Make sure kaleido is installed:\npip install kaleido")

    # ── Run pipeline ───────────────────────────────────────────────────────────
    if uploaded_file and run_button:
        try:
            df = pd.read_csv(uploaded_file)
            df = preprocess_reviews(df)
            df = predict_reviews(df, model, vectorizer)
            df = extract_themes(df, THEMES)
            st.session_state.analyzed_df = df
        except Exception as exc:
            st.error(f"Error processing uploaded file: {exc}")

    if "analyzed_df" in st.session_state:
        render_dashboard(st.session_state.analyzed_df)

    if not uploaded_file and "analyzed_df" not in st.session_state:
        st.info("Upload a CSV file to begin analysis.")


if __name__ == "__main__":
    main()

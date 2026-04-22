import io
import sys
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
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

st.set_page_config(page_title="Feedback Intelligence Platform", layout="wide")


# ── Shared helpers ─────────────────────────────────────────────────────────────

COLOUR_MAP = {
    "positive":      "#16a34a",
    "negative":      "#dc2626",
    "neutral":       "#6b7280",
    "neutral/mixed": "#9ca3af",
}

def _text_col(df):
    for c in ["text", "raw_text", "clean_text"]:
        if c in df.columns:
            return c
    return "clean_text"


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
        return "positive" if stars >= 4 else ("negative" if stars <= 2 else None)
    return "positive" if stars >= 4 else ("neutral/mixed" if stars == 3 else "negative")


def prepare_training_data():
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


def load_or_train_model():
    ensure_output_dir()
    if MODEL_FILE.exists() and VECTORIZER_FILE.exists():
        return joblib.load(MODEL_FILE), joblib.load(VECTORIZER_FILE), None
    with st.spinner("Training model from sample data..."):
        return train_model()


# ── Preprocessing & prediction ────────────────────────────────────────────────

def preprocess_reviews(df):
    if "clean_text" in df.columns:
        df["clean_text"] = df["clean_text"].fillna("").astype(str).str.lower()
    elif "text" in df.columns:
        df["clean_text"] = df["text"].fillna("").astype(str).str.lower()
    elif "raw_text" in df.columns:
        df["clean_text"] = df["raw_text"].fillna("").astype(str).str.lower()
    else:
        raise ValueError("CSV must have a 'text', 'raw_text', or 'clean_text' column")
    return df


CONTRAST_WORDS = {"but", "however", "though", "although", "yet", "except", "overall", "while"}
POS_CUES = {"good", "great", "nice", "friendly", "fast", "clean", "love", "excellent", "amazing", "enjoy"}
NEG_CUES = {"bad", "slow", "rude", "wrong", "dirty", "hate", "awful", "terrible", "issue", "problem"}


def has_contrast(text):
    t = f" {text.lower()} "
    return any(f" {w} " in t for w in CONTRAST_WORDS)


def has_dual_polarity_words(text):
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return bool(tokens & POS_CUES) and bool(tokens & NEG_CUES)


def mixed_rule(row):
    text  = str(row.get("clean_text", ""))
    p_pos = float(row.get("prob_positive", 0.0))
    p_neg = float(row.get("prob_negative", 0.0))
    prob_cond     = (p_pos >= 0.30) and (p_neg >= 0.30) and (abs(p_pos - p_neg) <= 0.25)
    contrast_cond = has_contrast(text)
    lex_cond      = has_dual_polarity_words(text)
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
    mask = df["predicted_sentiment"].isin(["neutral", "neutral/mixed"])
    if mask.any():
        df.loc[mask, "is_mixed"] = df[mask].apply(mixed_rule, axis=1)
    return df


# ── Theme extraction ──────────────────────────────────────────────────────────

def build_prompt(batch, themes):
    numbered = "\n".join([f"Review {i+1}:\n{str(r)[:250]}" for i, r in enumerate(batch)])
    return f"""You are a professional theme classifier for customer reviews.
You work for a Feedback Intelligence Platform that analyzes reviews.

Available themes: {themes}

RULES:
- Only assign themes from the available list above. NEVER invent or create your own themes.
- Every review MUST have at least one theme assigned.
- Return ONLY a valid JSON dictionary where keys are Review Numbers ("1", "2", etc.) and values are arrays of themes.
- You MUST generate exactly {len(batch)} keys.
- No extra explanation outside the JSON block.

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


def extract_themes_with_retry(batch_info, themes_list, max_retries=5):
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
                raise ValueError("LLM output contained no valid JSON")
            clean = raw[start:end]
            try:
                parsed_data = json.loads(clean)
            except json.JSONDecodeError:
                try:
                    parsed_data = ast.literal_eval(clean)
                except Exception:
                    if clean.strip().startswith("{") and clean.strip().endswith("}"):
                        parsed_data = json.loads(f"[{clean}]")
                    else:
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
                if not isinstance(t, list):
                    t = [t]
                if not t:
                    t = ["Customer Service"]
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
                            if real not in valid:
                                valid.append(real)
                            break
                if not valid:
                    raise ValueError(f"Hallucination: '{tlist}' not in allowed list")
                validated.append(valid)
            return batch_idx, batch, validated, "success"
        except Exception as e:
            print(f"Batch {batch_idx} attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
    return batch_idx, batch, None, "failed"


def extract_themes(df, themes_list, batch_size=30, max_workers=2):
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
                bidx, batch = futures[future]
                bstatus, bthemes = "failed", [["FAILED"]] * len(batch)
            except Exception:
                bidx, batch = futures[future]
                bstatus, bthemes = "failed", [["FAILED"]] * len(batch)
            if bstatus == "success":
                for i, (_, vt) in enumerate(zip(batch, bthemes)):
                    success.append({"original_idx": bidx + i, "themes": ", ".join(vt)})
            else:
                for i in range(len(batch)):
                    failed.append({"original_idx": bidx + i, "themes": "FAILED"})
            done += 1
            if done % max(1, total // 100) == 0 or done == total:
                pbar.progress(done / total)
                status.text(f"Processed batch {done}/{total} — please wait...")
    lookup = {r["original_idx"]: r["themes"] for r in success + failed}
    df["themes"] = [lookup.get(i, "FAILED") for i in range(len(df))]
    before = len(df)
    df = df[~df["themes"].str.contains("FAILED", na=False)]
    if len(df) < before:
        print(f"Dropped {before - len(df)} reviews due to extraction failure.")
    pbar.progress(1.0)
    status.text("Theme extraction complete!")
    return df


# ── PDF export ────────────────────────────────────────────────────────────────

_CLR_POS    = colors.HexColor("#16a34a")
_CLR_NEG    = colors.HexColor("#dc2626")
_CLR_HEADER = colors.HexColor("#1e3a5f")
_CLR_RULE   = colors.HexColor("#e5e7eb")
_CLR_BG     = colors.HexColor("#f0f4ff")


def _build_pdf_styles():
    base = getSampleStyleSheet()
    return (
        ParagraphStyle("T",  parent=base["Title"],    fontSize=22, textColor=_CLR_HEADER, spaceAfter=4,   fontName="Helvetica-Bold"),
        ParagraphStyle("ST", parent=base["Normal"],   fontSize=10, textColor=colors.HexColor("#6b7280"),  spaceAfter=16),
        ParagraphStyle("H2", parent=base["Heading2"], fontSize=13, textColor=_CLR_HEADER, spaceBefore=18, spaceAfter=6, fontName="Helvetica-Bold"),
        ParagraphStyle("B",  parent=base["Normal"],   fontSize=9,  leading=14, textColor=colors.HexColor("#374151")),
        ParagraphStyle("S",  parent=base["Normal"],   fontSize=8,  leading=12, textColor=colors.HexColor("#6b7280")),
    )


def _fig_to_img(fig, w=6.5, h=3.2):
    buf = io.BytesIO(fig.to_image(format="png", width=int(w*100), height=int(h*100), scale=2))
    return RLImage(buf, width=w*inch, height=h*inch)


def _pdf_pie(df):
    counts = df["predicted_sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(counts, names="sentiment", values="count",
                 color="sentiment", color_discrete_map=COLOUR_MAP, title="Sentiment Distribution")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="white", font_size=11)
    return _fig_to_img(fig, 3.0, 2.8)


def _pdf_bar(df):
    exp = df["themes"].str.split(r",\s*").explode().str.strip()
    exp = exp[~exp.isin(["FAILED", "", "NOT PROCESSED"])].value_counts().head(8).reset_index()
    exp.columns = ["Theme", "Count"]
    fig = px.bar(exp, x="Count", y="Theme", orientation="h", title="Top Themes",
                 color="Count", color_continuous_scale="Blues")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="white",
                      yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False)
    return _fig_to_img(fig, 3.8, 2.8)


def _pdf_trend(df):
    if "date" not in df.columns:
        return None
    try:
        d  = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        td = d.groupby([pd.Grouper(key="date", freq="ME"), "predicted_sentiment"])\
               .size().reset_index(name="count")
        if len(td) <= 1:
            return None
        fig = px.line(td, x="date", y="count", color="predicted_sentiment",
                      title="Monthly Trend", color_discrete_map=COLOUR_MAP)
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="white")
        return _fig_to_img(fig, 6.5, 2.8)
    except Exception:
        return None


def _pdf_kpi(df):
    total   = len(df)
    pos     = int((df["predicted_sentiment"] == "positive").sum())
    neg     = int((df["predicted_sentiment"] == "negative").sum())
    mixed   = int(df["predicted_sentiment"].isin(["neutral", "neutral/mixed"]).sum())
    conf    = df["confidence"].mean() if "confidence" in df.columns else None
    flagged = int((df["is_mixed"] == True).sum()) if "is_mixed" in df.columns else 0
    rows = [
        ["Metric", "Value", "Share"],
        ["Total reviews analysed",  f"{total:,}",    "100%"],
        ["Positive",                f"{pos:,}",       f"{pos/total*100:.1f}%"],
        ["Negative",                f"{neg:,}",       f"{neg/total*100:.1f}%"],
        ["Neutral / Mixed",         f"{mixed:,}",     f"{mixed/total*100:.1f}%"],
        ["Mixed-signal flagged",    f"{flagged:,}",   f"{flagged/total*100:.1f}%"],
    ]
    if conf:
        rows.append(["Avg model confidence", f"{conf*100:.1f}%", "—"])
    tbl = Table(rows, colWidths=[3.2*inch, 1.5*inch, 1.5*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _CLR_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _CLR_HEADER),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("TEXTCOLOR",     (0, 1), (-1, -1), colors.HexColor("#374151")),
        ("TEXTCOLOR",     (2, 2), (2, 2),  _CLR_POS),
        ("TEXTCOLOR",     (2, 3), (2, 3),  _CLR_NEG),
        ("GRID",          (0, 0), (-1, -1), 0.4, _CLR_RULE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    return tbl


def _pdf_theme_table(df):
    if "themes" not in df.columns:
        return None
    exp = df.assign(Theme=df["themes"].str.split(r",\s*")).explode("Theme")
    exp["Theme"] = exp["Theme"].str.strip()
    exp = exp[~exp["Theme"].isin(["FAILED", "", "NOT PROCESSED"])]
    if exp.empty:
        return None
    pivot = pd.crosstab(exp["Theme"], exp["predicted_sentiment"])
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).head(8)
    scols = [c for c in ["positive", "negative", "neutral", "neutral/mixed"] if c in pivot.columns]
    rows  = [["Theme"] + [c.capitalize() for c in scols] + ["Total"]]
    for theme, row in pivot.iterrows():
        dr = [str(theme)]
        for sc in scols:
            v = int(row.get(sc, 0))
            p = v / row["Total"] * 100 if row["Total"] else 0
            dr.append(f"{v} ({p:.0f}%)")
        dr.append(str(int(row["Total"])))
        rows.append(dr)
    tbl = Table(rows, colWidths=[2.5*inch] + [1.0*inch]*len(scols) + [0.8*inch])
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
    for i, sc in enumerate(scols):
        if sc == "positive": style.append(("TEXTCOLOR", (i+1, 0), (i+1, 0), _CLR_POS))
        if sc == "negative": style.append(("TEXTCOLOR", (i+1, 0), (i+1, 0), _CLR_NEG))
    tbl.setStyle(TableStyle(style))
    return tbl


def _pdf_samples(df):
    tc = _text_col(df)
    get = lambda s: (df[df["predicted_sentiment"] == s].nlargest(3, "confidence")
                     if "confidence" in df.columns
                     else df[df["predicted_sentiment"] == s].head(3))
    sample = pd.concat([get("positive"), get("negative")])
    rows = [["Sentiment", "Review excerpt", "Confidence"]]
    for _, r in sample.iterrows():
        ex   = str(r[tc])[:160].strip() + ("…" if len(str(r[tc])) > 160 else "")
        conf = f"{r['confidence']*100:.0f}%" if "confidence" in r else "—"
        rows.append([r["predicted_sentiment"].capitalize(), ex, conf])
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
    ]))
    return tbl


def _pdf_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#9ca3af"))
    canvas.drawString(0.75*inch, 0.5*inch,
                      "Customer Feedback Intelligence Platform  |  Confidential")
    canvas.drawRightString(letter[0]-0.75*inch, 0.5*inch, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(df, report_title="Feedback Intelligence Report"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.85*inch, bottomMargin=0.75*inch)
    ts, sts, h2s, bs, ss = _build_pdf_styles()
    story = []
    story.append(Paragraph(report_title, ts))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  |  "
        f"{len(df):,} reviews analysed", sts))
    story.append(HRFlowable(width="100%", thickness=1, color=_CLR_RULE, spaceAfter=12))
    story.append(Paragraph("1. Summary Metrics", h2s))
    story.append(_pdf_kpi(df))
    story.append(Spacer(1, 14))
    story.append(Paragraph("2. Sentiment & Theme Overview", h2s))
    try:
        pie = _pdf_pie(df)
        bar = _pdf_bar(df) if "themes" in df.columns else None
        if bar:
            cr = Table([[pie, bar]], colWidths=[3.2*inch, 4.0*inch])
            cr.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                                    ("LEFTPADDING", (0,0), (-1,-1), 0),
                                    ("RIGHTPADDING", (0,0), (-1,-1), 8)]))
            story.append(cr)
        else:
            story.append(pie)
    except Exception as e:
        story.append(Paragraph(f"Charts unavailable: {e} — ensure kaleido is installed.", ss))
    story.append(Spacer(1, 10))
    trend = _pdf_trend(df)
    off   = 0
    if trend:
        story.append(Paragraph("3. Monthly Sentiment Trend", h2s))
        story.append(trend)
        story.append(Spacer(1, 10))
        off = 1
    tt = _pdf_theme_table(df)
    if tt:
        story.append(Paragraph(f"{3+off}. Theme Sentiment Breakdown", h2s))
        story.append(Paragraph("Count (%) of reviews per sentiment for the top 8 themes.", bs))
        story.append(Spacer(1, 6))
        story.append(tt)
        story.append(Spacer(1, 10))
    st_ = _pdf_samples(df)
    if st_:
        story.append(PageBreak())
        story.append(Paragraph(f"{4+off}. Sample Review Verbatims", h2s))
        story.append(Paragraph("Highest-confidence positive and negative reviews.", bs))
        story.append(Spacer(1, 6))
        story.append(st_)
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_CLR_RULE, spaceAfter=6))
    story.append(Paragraph(
        "Auto-generated by the Customer Feedback Intelligence Platform. "
        "Sentiment predictions use Logistic Regression. Themes assigned by local LLM (Ollama/Gemma). "
        "Review alongside raw data before making decisions.", ss))
    doc.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared CSS injected once ───────────────────────────────────────────────────
METRIC_CSS = """
<style>
[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 16px;
    min-height: 110px;
}
[data-testid="stMetricLabel"] {
    font-size: 13px;
    font-weight: 600;
    color: #6b7280;
}
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 700;
}
[data-testid="stMetricDelta"] {
    font-size: 12px;
    color: #6b7280 !important;
}
[data-testid="stMetricDelta"] svg {
    display: none;
}
</style>
"""


def page_overview(df):
    # Inject metric card CSS so all cards sit at equal height
    st.markdown(METRIC_CSS, unsafe_allow_html=True)

    total    = len(df)
    pos      = int((df["predicted_sentiment"] == "positive").sum())
    neg      = int((df["predicted_sentiment"] == "negative").sum())
    mixed    = int(df["predicted_sentiment"].isin(["neutral","neutral/mixed"]).sum())
    avg_conf = df["confidence"].mean() if "confidence" in df.columns else None

    # KPI cards — all deltas use the same label format so height is consistent
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews",      f"{total:,}",                    "100% of reviews")
    c2.metric("✅ Positive",         f"{pos:,}",                      f"{pos/total*100:.1f}% of reviews")
    c3.metric("❌ Negative",         f"{neg:,}",                      f"{neg/total*100:.1f}% of reviews")
    c4.metric("🔀 Neutral / Mixed",  f"{mixed:,}",                    f"{mixed/total*100:.1f}% of reviews")
    c5.metric("🎯 Avg Confidence",   f"{avg_conf*100:.1f}%" if avg_conf else "N/A", "model certainty")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        counts = df["predicted_sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig = px.pie(counts, names="sentiment", values="count",
                     title="Predicted Sentiment Distribution",
                     color="sentiment", color_discrete_map=COLOUR_MAP)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"])
                td = df.groupby([pd.Grouper(key="date", freq="ME"), "predicted_sentiment"])\
                       .size().reset_index(name="count")
                if len(td) > 1:
                    fig2 = px.line(td, x="date", y="count", color="predicted_sentiment",
                                   title="Sentiment Over Time (Monthly)",
                                   color_discrete_map=COLOUR_MAP)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Not enough date range for a trend chart.")
            except Exception:
                st.info("Date column could not be parsed.")
        else:
            st.info("No 'date' column found — add one to your CSV to see trends.")

    st.markdown("---")
    st.markdown("**Data preview — first 10 rows**")
    preview_cols = [c for c in ["predicted_sentiment", "confidence", "is_mixed", "themes"]
                    if c in df.columns]
    st.dataframe(df[preview_cols].head(10), use_container_width=True)

    st.download_button(
        label="⬇ Download full analysed CSV",
        data=df.to_csv(index=False),
        file_name="analysis_results.csv",
        mime="text/csv",
    )


def page_positive(df):
    pos_df = df[df["predicted_sentiment"] == "positive"].copy()
    st.markdown(f"### Positive Reviews — {len(pos_df):,} total")
    if pos_df.empty:
        st.info("No positive reviews found.")
        return

    if "confidence" in pos_df.columns:
        fig = px.histogram(pos_df, x="confidence", nbins=10,
                           title="Confidence Score Distribution — Positive Reviews",
                           color_discrete_sequence=["#16a34a"])
        fig.update_traces(marker_line_color="white", marker_line_width=2)
        fig.update_layout(
            bargap=0.15,
            xaxis_title="Model Confidence Score (higher = more certain)",
            yaxis_title="Number of Reviews",
            xaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)

    if "themes" in pos_df.columns:
        st.markdown("**Most mentioned themes in positive reviews**")
        exp = pos_df["themes"].str.split(r",\s*").explode().str.strip()
        exp = exp[~exp.isin(["FAILED", "", "NOT PROCESSED"])]
        tc  = exp.value_counts().head(8).reset_index()
        tc.columns = ["Theme", "Count"]
        fig2 = px.bar(tc, x="Count", y="Theme", orientation="h",
                      color_discrete_sequence=["#16a34a"])
        fig2.update_traces(marker_line_color="white", marker_line_width=1)
        fig2.update_layout(
            bargap=0.2,
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Number of reviews mentioning this theme",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Sample positive reviews**")
    tc_name = _text_col(pos_df)
    hc, bc  = st.columns([3, 1])
    with bc:
        st.button("🔄 Refresh", key="pos_refresh")
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
        fig = px.histogram(neg_df, x="confidence", nbins=10,
                           title="Confidence Score Distribution — Negative Reviews",
                           color_discrete_sequence=["#dc2626"])
        fig.update_traces(marker_line_color="white", marker_line_width=2)
        fig.update_layout(
            bargap=0.15,
            xaxis_title="Model Confidence Score (higher = more certain)",
            yaxis_title="Number of Reviews",
            xaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)

    if "themes" in neg_df.columns:
        st.markdown("**Most mentioned themes in negative reviews**")
        exp = neg_df["themes"].str.split(r",\s*").explode().str.strip()
        exp = exp[~exp.isin(["FAILED", "", "NOT PROCESSED"])]
        tc  = exp.value_counts().head(8).reset_index()
        tc.columns = ["Theme", "Count"]
        fig2 = px.bar(tc, x="Count", y="Theme", orientation="h",
                      color_discrete_sequence=["#dc2626"])
        fig2.update_traces(marker_line_color="white", marker_line_width=1)
        fig2.update_layout(
            bargap=0.2,
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Number of reviews mentioning this theme",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Sample negative reviews**")
    tc_name = _text_col(neg_df)
    hc, bc  = st.columns([3, 1])
    with bc:
        st.button("🔄 Refresh", key="neg_refresh")
    for _, row in neg_df.sample(min(10, len(neg_df))).iterrows():
        conf = f" *(confidence: {row['confidence']*100:.0f}%)*" if "confidence" in row else ""
        st.error(f'"{row[tc_name]}"{conf}')


def page_themes(df):
    if "themes" not in df.columns:
        st.warning("No themes column found. Run the analysis pipeline first.")
        return

    exp_all = df["themes"].str.split(r",\s*").explode().str.strip()
    exp_all = exp_all[~exp_all.isin(["FAILED", "", "NOT PROCESSED"])]
    theme_counts = exp_all.value_counts().head(20).reset_index()
    theme_counts.columns = ["Theme", "Count"]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(theme_counts, use_container_width=True)
    with col2:
        fig = px.bar(theme_counts, x="Count", y="Theme", orientation="h",
                     title="Most Common Themes", color="Count",
                     color_continuous_scale="Viridis")
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(
            bargap=0.2,
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Sentiment Breakdown by Theme**")
    df_exp = df.assign(Theme=df["themes"].str.split(r",\s*")).explode("Theme")
    df_exp["Theme"] = df_exp["Theme"].str.strip()
    df_exp = df_exp[~df_exp["Theme"].isin(["FAILED", "", "NOT PROCESSED"])].reset_index(drop=True)

    if not df_exp.empty:
        pivot = pd.crosstab(df_exp["Theme"], df_exp["predicted_sentiment"],
                            margins=True, margins_name="Total")
        scols = [c for c in ["positive", "negative", "neutral", "neutral/mixed"] if c in pivot.columns]
        for c in scols:
            pivot[c+" (%)"] = (pivot[c] / pivot["Total"] * 100).round(1)
        ordered = []
        for c in scols:
            ordered.extend([c, c+" (%)"])
        ordered.append("Total")
        st.dataframe(pivot[ordered], use_container_width=True)

        st.markdown("---")
        st.markdown("**Interactive Theme Sentiment Distribution**")
        unique_themes = sorted(df_exp["Theme"].unique().tolist())
        selected      = st.selectbox("Select a theme:", unique_themes, key="theme_select")
        t_data        = df_exp[df_exp["Theme"] == selected]
        t_counts      = t_data["predicted_sentiment"].value_counts().reset_index()
        t_counts.columns = ["sentiment", "count"]
        fig2 = px.pie(t_counts, names="sentiment", values="count",
                      title=f"Sentiment split for '{selected}'",
                      color="sentiment", color_discrete_map=COLOUR_MAP)
        st.plotly_chart(fig2, use_container_width=True)

        # Top phrases
        st.markdown(f"**Top phrases in '{selected}'**")
        try:
            sent_docs = df_exp.groupby("Theme")["clean_text"].apply(
                lambda x: " ".join(x.dropna())).to_dict()
            if selected in sent_docs and sent_docs[selected].strip():
                extra_stop = {"review","user","star","stars","https","http","amp","just","like","im"}
                sw  = list(set(sk_text.ENGLISH_STOP_WORDS) | extra_stop)
                tv  = TfidfVectorizer(ngram_range=(2, 3), stop_words=sw)
                mat = tv.fit_transform(list(sent_docs.values()))
                idx = list(sent_docs.keys()).index(selected)
                dense  = mat.toarray()
                scores = dense[idx] * (dense.argmax(axis=0) == idx)
                top_i  = scores.argsort()[-10:][::-1]
                phrases = [tv.get_feature_names_out()[i] for i in top_i if scores[i] > 0]
                pcounts = [sent_docs[selected].count(p) for p in phrases]
                pf = pd.DataFrame({"Phrase": phrases, "Count": pcounts})\
                       .sort_values("Count", ascending=True)
                fig3 = px.bar(pf, x="Count", y="Phrase", orientation="h")
                fig3.update_traces(marker_line_color="white", marker_line_width=1)
                fig3.update_layout(
                    bargap=0.2,
                    xaxis_title="Mentions",
                    yaxis_title="",
                )
                st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.info(f"Could not extract phrases: {e}")

        # Time trends
        if "date" in df.columns:
            st.markdown("---")
            st.subheader("Time-Based & Emergent Trends")
            try:
                theme_time = (
                    df_exp.groupby([pd.Grouper(key="date", freq="ME"), "Theme"])
                    .size().reset_index(name="count")
                )
                if len(theme_time["date"].unique()) > 1:
                    col_tr1, col_tr2 = st.columns(2)
                    with col_tr1:
                        st.markdown(f"**Monthly volume for '{selected}'**")
                        sel_tt = theme_time[theme_time["Theme"] == selected]
                        fig_t  = px.bar(sel_tt, x="date", y="count",
                                        labels={"date":"Month","count":"Mentions"})
                        fig_t.update_traces(marker_line_color="white", marker_line_width=1)
                        fig_t.update_layout(bargap=0.2)
                        st.plotly_chart(fig_t, use_container_width=True)
                    with col_tr2:
                        st.markdown("**Emergent Themes (month-over-month)**")
                        months     = sorted(theme_time["date"].unique())
                        curr_month = months[-1]
                        prev_month = months[-2]
                        curr_df = theme_time[theme_time["date"]==curr_month].set_index("Theme")
                        prev_df = theme_time[theme_time["date"]==prev_month].set_index("Theme")
                        emer = curr_df[["count"]].join(prev_df[["count"]],
                                                        lsuffix="_curr", rsuffix="_prev",
                                                        how="outer").fillna(0)
                        emer["Change"] = emer["count_curr"] - emer["count_prev"]
                        st.dataframe(
                            emer.sort_values("Change", ascending=False)
                            [["count_prev","count_curr","Change"]]
                            .rename(columns={"count_prev":"Prev Month",
                                             "count_curr":"Current Month",
                                             "Change":"Momentum"}),
                            use_container_width=True)
                else:
                    st.info("Need at least 2 months of data for trend momentum.")
            except Exception as e:
                st.warning(f"Could not calculate trends: {e}")

        # Deep dive verbatims
        st.markdown("---")
        st.markdown("**Read actual reviews by theme + sentiment**")
        dd_col1, dd_col2 = st.columns(2)
        with dd_col1:
            dd_theme = st.selectbox("Theme:", unique_themes, key="dd_theme")
        with dd_col2:
            dd_sent  = st.selectbox("Sentiment:", ["negative","positive","neutral","neutral/mixed"], key="dd_sent")
        dd_data = df_exp[(df_exp["Theme"]==dd_theme) & (df_exp["predicted_sentiment"]==dd_sent)]
        if dd_data.empty:
            st.info(f"No {dd_sent} reviews found for '{dd_theme}'.")
        else:
            hc, bc = st.columns([3, 1])
            with bc:
                st.button("🔄 Refresh", key="dd_refresh")
            tc_name = _text_col(dd_data)
            for rev in dd_data.sample(min(5, len(dd_data)))[tc_name].tolist():
                st.info(f'"{rev}"')


def page_outliers(df):
    st.markdown("### Outlier & Edge-Case Reviews")
    st.write(
        "**Low-confidence predictions** are reviews the model was unsure about — "
        "they scored close to 50/50 between two sentiments. "
        "**Mixed-signal reviews** contain both positive and negative language in the same review."
    )

    if "confidence" in df.columns:
        st.markdown("---")
        st.markdown("#### Low-Confidence Predictions")
        threshold = st.slider(
            "Show reviews below this confidence level:",
            0.40, 0.90, 0.60, 0.05,
            help="Lower = model was very unsure. 60% means the model was only 60% sure of its prediction."
        )
        low_conf = df[df["confidence"] < threshold].copy()
        st.markdown(
            f"**{len(low_conf):,} reviews** were predicted with less than "
            f"**{threshold*100:.0f}% confidence** — these may be worth checking manually."
        )

        if not low_conf.empty:
            fig = px.histogram(
                low_conf, x="confidence", color="predicted_sentiment",
                nbins=8,
                title="How uncertain was the model? (grouped by predicted sentiment)",
                color_discrete_map=COLOUR_MAP,
                barmode="group",
            )
            fig.update_traces(marker_line_color="white", marker_line_width=2)
            fig.update_layout(
                bargap=0.2,
                xaxis_title="Model Confidence Score (e.g. 0.50 = 50% sure, almost a coin flip)",
                yaxis_title="Number of Reviews",
                xaxis=dict(tickformat=".0%"),
                legend_title_text="Predicted Sentiment",
            )
            st.plotly_chart(fig, use_container_width=True)

            tc_name   = _text_col(low_conf)
            show_cols = [tc_name, "predicted_sentiment", "confidence"] + \
                        [c for c in ["themes", "is_mixed"] if c in low_conf.columns]
            st.markdown("**Reviews sorted by lowest confidence first:**")
            st.dataframe(
                low_conf[show_cols].sort_values("confidence").head(30),
                use_container_width=True
            )
            st.download_button(
                "⬇ Download low-confidence reviews",
                data=low_conf.to_csv(index=False),
                file_name="low_confidence_reviews.csv",
                mime="text/csv",
            )

    st.markdown("---")

    if "is_mixed" in df.columns:
        st.markdown("#### Mixed-Signal Reviews")
        st.write(
            "These reviews contain contrast words like *'but', 'however', 'although'* "
            "alongside both positive and negative vocabulary — e.g. *'Great coffee but terrible service'*."
        )
        mixed_df = df[df["is_mixed"] == True].copy()
        st.markdown(f"**{len(mixed_df):,} mixed-signal reviews** detected")

        if not mixed_df.empty:
            tc_name   = _text_col(mixed_df)
            show_cols = [tc_name, "predicted_sentiment", "confidence"] + \
                        [c for c in ["themes"] if c in mixed_df.columns]
            st.dataframe(mixed_df[show_cols].head(30), use_container_width=True)
            st.download_button(
                "⬇ Download mixed-signal reviews",
                data=mixed_df.to_csv(index=False),
                file_name="mixed_signal_reviews.csv",
                mime="text/csv",
            )
        else:
            st.info("No mixed-signal reviews detected in this dataset.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("Feedback Intelligence Platform")
    st.markdown(
        "Upload a review CSV in the sidebar, click **Run Analysis**, "
        "then navigate results using the tabs below."
    )

    model, vectorizer, train_accuracy = load_or_train_model()
    if model is None or vectorizer is None:
        st.error("Model could not be loaded or trained.")
        return
    if train_accuracy is not None:
        st.success(f"Model trained — accuracy: {train_accuracy:.4f}")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("Upload & Run")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV for analysis", type="csv",
        help="CSV must have a 'text', 'clean_text', or 'raw_text' column.")

    run_button = st.sidebar.button("▶ Run Analysis", use_container_width=True)

    if st.sidebar.button("🗑 Reset / Clear Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # PDF export — only appears after analysis
    if "analyzed_df" in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.header("Export")
        custom_title = st.sidebar.text_input("Report title", value="Feedback Intelligence Report")
        if st.sidebar.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Building PDF..."):
                try:
                    pdf_bytes = generate_pdf_report(st.session_state.analyzed_df, custom_title)
                    st.sidebar.download_button(
                        label="⬇ Download PDF",
                        data=pdf_bytes,
                        file_name=f"feedback_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.sidebar.success("PDF ready!")
                except Exception as e:
                    st.sidebar.error(f"PDF failed: {e}")
                    st.sidebar.info("Run: pip install kaleido")

    # ── Run pipeline ───────────────────────────────────────────────────────────
    if uploaded_file and run_button:
        try:
            df = pd.read_csv(uploaded_file)
            df = preprocess_reviews(df)
            df = predict_reviews(df, model, vectorizer)
            df = extract_themes(df, THEMES)
            st.session_state.analyzed_df = df
        except Exception as exc:
            st.error(f"Error during analysis: {exc}")

    # ── Tabbed results ─────────────────────────────────────────────────────────
    if "analyzed_df" in st.session_state:
        df = st.session_state.analyzed_df
        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊  Overview",
            "😊  Positive Reviews",
            "😞  Negative Reviews",
            "🏷️  Theme Extraction",
            "⚠️  Outliers",
        ])

        with tab1:
            page_overview(df)
        with tab2:
            page_positive(df)
        with tab3:
            page_negative(df)
        with tab4:
            page_themes(df)
        with tab5:
            page_outliers(df)

    elif not uploaded_file:
        st.info("👆 Upload a CSV file in the sidebar and click **Run Analysis** to get started.")


if __name__ == "__main__":
    main()

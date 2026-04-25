import io
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
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

# ReportLab — used only in page_overview for the PDF button
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
        df["sentiment"] = df["stars"].apply(
            lambda x: sentiments_from_stars(x, "three_class"))
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


CONTRAST_WORDS = {"but","however","though","although","yet","except","overall","while"}
POS_CUES = {"good","great","nice","friendly","fast","clean","love","excellent","amazing","enjoy"}
NEG_CUES = {"bad","slow","rude","wrong","dirty","hate","awful","terrible","issue","problem"}


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
    numbered = "\n".join([
        f"Review {i+1}:\n{str(r)[:250]}" for i, r in enumerate(batch)])
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
                raise ValueError(
                    f"Count mismatch: got {len(themes)}, expected {len(batch)}")
            validated = []
            for tlist in themes:
                safe  = [str(list(t.values())[0]) if isinstance(t, dict) and t
                          else str(t) for t in tlist]
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
        futures = {ex.submit(extract_themes_with_retry, b, themes_list): b
                   for b in batches}
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
                    success.append({"original_idx": bidx + i,
                                    "themes": ", ".join(vt)})
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


# ── Shared CSS ─────────────────────────────────────────────────────────────────

METRIC_CSS = """
<style>
[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 16px;
    min-height: 110px;
}
[data-testid="stMetricLabel"] { font-size: 13px; font-weight: 600; color: #6b7280; }
[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 12px; color: #6b7280 !important; }
[data-testid="stMetricDelta"] svg { display: none; }
</style>
"""


# ── Axis / layout helpers ──────────────────────────────────────────────────────

def _vertical_bar_axis():
    return dict(
        xaxis=dict(
            tickformat=".0%",
            range=[-0.05, 1.05],
            tickvals=[i / 10 for i in range(0, 11)],
            ticktext=[f"{i*10}%" for i in range(0, 11)],
            title_standoff=10,
        ),
        yaxis=dict(rangemode="tozero", dtick=1, tickformat="d"),
    )


def _horizontal_bar_axis(max_val):
    nice_max = max(1, int(max_val) + 1)
    step = max(1, int(nice_max / 8))
    return dict(
        xaxis=dict(
            range=[0, nice_max + step * 0.5],
            tick0=0, dtick=step, tickformat="d", rangemode="tozero",
        ),
        yaxis=dict(categoryorder="total ascending"),
    )


def _base_layout(**extra):
    layout = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=20, t=45, b=70),
        bargap=0.2, bargroupgap=0.05,
    )
    layout.update(extra)
    return layout


def _hist_yaxis(df):
    """Dynamic y-axis for confidence histograms.
    Always steps of 20 (0, 20, 40...), scales to dataset size, minimum 20."""
    n = len(df)
    # estimate tallest bin: roughly n / number_of_bins, padded generously
    est_max = max(20, n // 4)
    ceiling = ((est_max // 20) + 1) * 20   # round up to next multiple of 20
    return dict(rangemode="tozero", dtick=20, tickformat="d", range=[0, ceiling])


# ── Chart download ─────────────────────────────────────────────────────────────

def _chart_download(fig, filename, label="Download chart as PNG"):
    try:
        png_bytes = fig.to_image(format="png", scale=2)
        st.download_button(label=label, data=png_bytes,
                           file_name=filename, mime="image/png")
    except Exception:
        st.caption("Install kaleido for chart downloads: pip install kaleido")


# ── Report download (used by all tabs) ────────────────────────────────────────

def _report_download(df, key):
    """Renders a Generate Report button that produces a full PDF of all results."""
    st.markdown("---")
    st.markdown("**Download Report**")
    if st.button("Generate Report", key=f"gen_{key}"):
        with st.spinner("Building report..."):
            try:
                pdf_bytes = _build_overview_pdf(df)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"customer_feedback_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    key=f"dl_{key}",
                )
                st.success("Report ready — click Download PDF Report above.")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.info("Make sure reportlab and kaleido are installed:\n"
                        "pip install reportlab kaleido")


# ── Overview PDF builder ───────────────────────────────────────────────────────

def _build_overview_pdf(df) -> bytes:
    """
    Build a clean PDF with:
      - KPI summary table
      - Sentiment pie chart (requires kaleido)
      - Monthly trend chart (requires kaleido, only if date column exists)
    Returns raw PDF bytes.
    """
    total    = len(df)
    pos      = int((df["predicted_sentiment"] == "positive").sum())
    neg      = int((df["predicted_sentiment"] == "negative").sum())
    mixed    = int(df["predicted_sentiment"].isin(["neutral","neutral/mixed"]).sum())
    avg_conf = df["confidence"].mean() * 100 if "confidence" in df.columns else None

    HDR  = rl_colors.HexColor("#1e3a5f")
    RULE = rl_colors.HexColor("#e5e7eb")
    BG   = rl_colors.HexColor("#f0f4ff")
    GRN  = rl_colors.HexColor("#16a34a")
    RED  = rl_colors.HexColor("#dc2626")

    base   = getSampleStyleSheet()
    title_s = ParagraphStyle("T",  parent=base["Title"],   fontSize=20,
                              textColor=HDR, fontName="Helvetica-Bold", spaceAfter=4)
    sub_s   = ParagraphStyle("S",  parent=base["Normal"],  fontSize=10,
                              textColor=rl_colors.HexColor("#6b7280"), spaceAfter=14)
    h2_s    = ParagraphStyle("H2", parent=base["Heading2"],fontSize=13,
                              textColor=HDR, fontName="Helvetica-Bold",
                              spaceBefore=16, spaceAfter=6)
    body_s  = ParagraphStyle("B",  parent=base["Normal"],  fontSize=9,
                              leading=13, textColor=rl_colors.HexColor("#374151"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.85*inch,  bottomMargin=0.75*inch)
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Customer Feedback Report", title_s))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  |  "
        f"{total:,} reviews analysed", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=12))

    # ── KPI table ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Summary Metrics", h2_s))
    kpi_rows = [
        ["Metric",           "Count",        "Share"],
        ["Total Reviews",    f"{total:,}",   "100%"],
        ["Positive",         f"{pos:,}",     f"{pos/total*100:.1f}%"],
        ["Negative",         f"{neg:,}",     f"{neg/total*100:.1f}%"],
        ["Neutral / Mixed",  f"{mixed:,}",   f"{mixed/total*100:.1f}%"],
    ]
    if avg_conf is not None:
        kpi_rows.append(["Avg Model Confidence", f"{avg_conf:.1f}%", "—"])

    kpi_tbl = Table(kpi_rows, colWidths=[3.0*inch, 1.5*inch, 1.5*inch])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), HDR),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("FONTNAME",      (0, 1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1,-1),
         [rl_colors.white, rl_colors.HexColor("#f9fafb")]),
        ("TEXTCOLOR",     (0, 1), (-1,-1), rl_colors.HexColor("#374151")),
        ("TEXTCOLOR",     (2, 2), (2, 2),  GRN),
        ("TEXTCOLOR",     (2, 3), (2, 3),  RED),
        ("GRID",          (0, 0), (-1,-1), 0.4, RULE),
        ("TOPPADDING",    (0, 0), (-1,-1), 5),
        ("BOTTOMPADDING", (0, 0), (-1,-1), 5),
        ("LEFTPADDING",   (0, 0), (-1,-1), 8),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 14))

    # ── Sentiment pie chart ────────────────────────────────────────────────────
    story.append(Paragraph("Sentiment Distribution", h2_s))
    try:
        counts = df["predicted_sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(counts, names="sentiment", values="count",
                         color="sentiment", color_discrete_map=COLOUR_MAP)
        fig_pie.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
        )
        png_pie = fig_pie.to_image(format="png", width=500, height=320, scale=2)
        story.append(RLImage(io.BytesIO(png_pie), width=4.5*inch, height=2.9*inch))
    except Exception as e:
        story.append(Paragraph(
            f"Pie chart unavailable ({e}). Run: pip install kaleido", body_s))

    # ── Monthly trend chart ────────────────────────────────────────────────────
    if "date" in df.columns:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Sentiment Over Time", h2_s))
        try:
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"])
            td = (d.groupby([pd.Grouper(key="date", freq="ME"), "predicted_sentiment"])
                   .size().reset_index(name="count"))
            if len(td) > 1:
                fig_trend = px.line(td, x="date", y="count",
                                    color="predicted_sentiment",
                                    color_discrete_map=COLOUR_MAP)
                fig_trend.update_layout(
                    margin=dict(l=60, r=20, t=20, b=50),
                    paper_bgcolor="white",
                    xaxis_title="Date",
                    yaxis_title="Number of Reviews",
                    yaxis=dict(
                        rangemode="tozero",
                        dtick=1,
                        tickformat="d",
                    ),
                    legend_title_text="Sentiment",
                )
                png_trend = fig_trend.to_image(
                    format="png", width=700, height=320, scale=2)
                story.append(RLImage(io.BytesIO(png_trend),
                                     width=6.0*inch, height=2.7*inch))
            else:
                story.append(Paragraph(
                    "Not enough date range for a trend chart.", body_s))
        except Exception as e:
            story.append(Paragraph(f"Trend chart unavailable: {e}", body_s))

    # ── Theme summary table (if themes exist) ─────────────────────────────────
    if "themes" in df.columns:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Top Themes", h2_s))
        exp_all = df["themes"].str.split(r",\s*").explode().str.strip()
        exp_all = exp_all[~exp_all.isin(["FAILED", "", "NOT PROCESSED"])]
        tc = exp_all.value_counts().head(8).reset_index()
        tc.columns = ["Theme", "Mentions"]
        theme_rows = [["Theme", "Mentions"]] + tc.values.tolist()
        theme_rows = [[str(c) for c in row] for row in theme_rows]
        theme_tbl = Table(theme_rows, colWidths=[4.0*inch, 2.0*inch])
        theme_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), BG),
            ("TEXTCOLOR",     (0, 0), (-1, 0), HDR),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 9),
            ("FONTNAME",      (0, 1), (-1,-1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1,-1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1,-1),
             [rl_colors.white, rl_colors.HexColor("#f9fafb")]),
            ("TEXTCOLOR",     (0, 1), (-1,-1), rl_colors.HexColor("#374151")),
            ("GRID",          (0, 0), (-1,-1), 0.4, RULE),
            ("TOPPADDING",    (0, 0), (-1,-1), 5),
            ("BOTTOMPADDING", (0, 0), (-1,-1), 5),
            ("LEFTPADDING",   (0, 0), (-1,-1), 8),
        ]))
        story.append(theme_tbl)

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(df):
    st.markdown(METRIC_CSS, unsafe_allow_html=True)

    total    = len(df)
    pos      = int((df["predicted_sentiment"] == "positive").sum())
    neg      = int((df["predicted_sentiment"] == "negative").sum())
    mixed    = int(df["predicted_sentiment"].isin(["neutral","neutral/mixed"]).sum())
    avg_conf = df["confidence"].mean() if "confidence" in df.columns else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews",   f"{total:,}",                      "100% of reviews")
    c2.metric("Positive",        f"{pos:,}",                        f"{pos/total*100:.1f}% of reviews")
    c3.metric("Negative",        f"{neg:,}",                        f"{neg/total*100:.1f}% of reviews")
    c4.metric("Neutral / Mixed", f"{mixed:,}",                      f"{mixed/total*100:.1f}% of reviews")
    c5.metric("Avg Confidence",  f"{avg_conf*100:.1f}%" if avg_conf else "N/A", "model certainty")

    st.markdown("---")

    # Pie chart — full width
    counts = df["predicted_sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig_pie = px.pie(counts, names="sentiment", values="count",
                     title="Predicted Sentiment Distribution",
                     color="sentiment", color_discrete_map=COLOUR_MAP)
    st.plotly_chart(fig_pie, use_container_width=True)
    _chart_download(fig_pie, "sentiment_distribution.png", "Download sentiment pie chart")

    # Time trend — full width below pie
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            td = (df.groupby([pd.Grouper(key="date", freq="ME"), "predicted_sentiment"])
                    .size().reset_index(name="count"))
            if len(td) > 1:
                fig_trend = px.line(td, x="date", y="count",
                                    color="predicted_sentiment",
                                    title="Sentiment Over Time",
                                    color_discrete_map=COLOUR_MAP)
                fig_trend.update_layout(**_base_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Reviews",
                    legend_title_text="Sentiment",
                    yaxis=dict(
                        rangemode="tozero",
                        range=[0, None],
                        dtick=1,
                        tickformat="d",
                    ),
                ))
                st.plotly_chart(fig_trend, use_container_width=True)
                _chart_download(fig_trend, "sentiment_trend.png", "Download trend chart")
            else:
                st.info("Not enough date range for a trend chart.")
        except Exception:
            st.info("Date column could not be parsed.")
    else:
        st.info("No 'date' column found — add one to your CSV to see trends.")

    st.markdown("---")
    st.markdown("**Data preview — first 10 rows**")
    preview_cols = [c for c in ["predicted_sentiment","confidence","is_mixed","themes"]
                    if c in df.columns]
    st.dataframe(df[preview_cols].head(10), use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Download**")
    col_csv, col_pdf = st.columns(2)

    with col_csv:
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="dl_csv_overview",
        )

    with col_pdf:
        if st.button("Generate PDF Report", key="gen_pdf_overview"):
            with st.spinner("Building PDF report..."):
                try:
                    pdf_bytes = _build_overview_pdf(df)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"overview_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        key="dl_pdf_overview",
                    )
                    st.success("PDF ready — click Download PDF Report above.")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                    st.info(
                        "Make sure reportlab is installed:\n"
                        "pip install reportlab\n\n"
                        "For charts inside the PDF also run:\n"
                        "pip install kaleido"
                    )


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
                               xbins=dict(start=0.0, end=1.0, size=0.10))
        fig_hist.update_layout(**_base_layout(
            xaxis_title="Model Confidence Score (higher = more certain)",
            yaxis_title="Number of Reviews",
            xaxis=dict(
                tickformat=".0%",
                range=[-0.05, 1.05],
                tickvals=[i/10 for i in range(0, 11)],
                ticktext=[f"{i*10}%" for i in range(0, 11)],
            ),
            yaxis=_hist_yaxis(pos_df),
        ))
        st.plotly_chart(fig_hist, use_container_width=True)
        _chart_download(fig_hist, "positive_confidence.png", "Download confidence chart")

    if "themes" in pos_df.columns:
        st.markdown("**Most mentioned themes in positive reviews**")
        exp = pos_df["themes"].str.split(r",\s*").explode().str.strip()
        exp = exp[~exp.isin(["FAILED","","NOT PROCESSED"])]
        tc  = exp.value_counts().head(8).reset_index()
        tc.columns = ["Theme","Count"]
        fig_bar = px.bar(tc, x="Count", y="Theme", orientation="h",
                         color_discrete_sequence=["#16a34a"])
        fig_bar.update_traces(marker_line_color="white", marker_line_width=1)
        fig_bar.update_layout(**_base_layout(
            xaxis_title="Number of reviews mentioning this theme",
            **_horizontal_bar_axis(tc["Count"].max())))
        st.plotly_chart(fig_bar, use_container_width=True)
        _chart_download(fig_bar, "positive_themes.png", "Download themes chart")

    st.markdown("**Sample positive reviews**")
    tc_name = _text_col(pos_df)
    hc, bc  = st.columns([3, 1])
    with bc:
        st.button("Refresh", key="pos_refresh")
    for _, row in pos_df.sample(min(10, len(pos_df))).iterrows():
        conf = f" *(confidence: {row['confidence']*100:.0f}%)*" if "confidence" in row else ""
        st.success(f'"{row[tc_name]}"{conf}')

    _report_download(df, key="rpt_positive")


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
                               xbins=dict(start=0.0, end=1.0, size=0.10))
        fig_hist.update_layout(**_base_layout(
            xaxis_title="Model Confidence Score (higher = more certain)",
            yaxis_title="Number of Reviews",
            xaxis=dict(
                tickformat=".0%",
                range=[-0.05, 1.05],
                tickvals=[i/10 for i in range(0, 11)],
                ticktext=[f"{i*10}%" for i in range(0, 11)],
            ),
            yaxis=_hist_yaxis(neg_df),
        ))
        st.plotly_chart(fig_hist, use_container_width=True)
        _chart_download(fig_hist, "negative_confidence.png", "Download confidence chart")

    if "themes" in neg_df.columns:
        st.markdown("**Most mentioned themes in negative reviews**")
        exp = neg_df["themes"].str.split(r",\s*").explode().str.strip()
        exp = exp[~exp.isin(["FAILED","","NOT PROCESSED"])]
        tc  = exp.value_counts().head(8).reset_index()
        tc.columns = ["Theme","Count"]
        fig_bar = px.bar(tc, x="Count", y="Theme", orientation="h",
                         color_discrete_sequence=["#dc2626"])
        fig_bar.update_traces(marker_line_color="white", marker_line_width=1)
        fig_bar.update_layout(**_base_layout(
            xaxis_title="Number of reviews mentioning this theme",
            **_horizontal_bar_axis(tc["Count"].max())))
        st.plotly_chart(fig_bar, use_container_width=True)
        _chart_download(fig_bar, "negative_themes.png", "Download themes chart")

    st.markdown("**Sample negative reviews**")
    tc_name = _text_col(neg_df)
    hc, bc  = st.columns([3, 1])
    with bc:
        st.button("Refresh", key="neg_refresh")
    for _, row in neg_df.sample(min(10, len(neg_df))).iterrows():
        conf = f" *(confidence: {row['confidence']*100:.0f}%)*" if "confidence" in row else ""
        st.error(f'"{row[tc_name]}"{conf}')

    _report_download(df, key="rpt_negative")


def page_neutral(df):
    neu_df = df[df["predicted_sentiment"].isin(["neutral","neutral/mixed"])].copy()
    st.markdown(f"### Neutral / Mixed Reviews — {len(neu_df):,} total")
    st.write("These reviews sit in the middle — the model did not detect a clearly "
             "positive or negative tone.")
    if neu_df.empty:
        st.info("No neutral or mixed reviews found in this dataset.")
        return

    if "confidence" in neu_df.columns:
        fig_hist = px.histogram(neu_df, x="confidence",
                                title="Confidence Score Distribution — Neutral / Mixed Reviews",
                                color_discrete_sequence=["#6b7280"])
        fig_hist.update_traces(marker_line_color="white", marker_line_width=2,
                               xbins=dict(start=0.0, end=1.0, size=0.10))
        fig_hist.update_layout(**_base_layout(
            xaxis_title="Model Confidence Score (higher = more certain it is neutral)",
            yaxis_title="Number of Reviews",
            xaxis=dict(
                tickformat=".0%",
                range=[-0.05, 1.05],
                tickvals=[i/10 for i in range(0, 11)],
                ticktext=[f"{i*10}%" for i in range(0, 11)],
            ),
            yaxis=_hist_yaxis(neu_df),
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
        tc  = exp.value_counts().head(8).reset_index()
        tc.columns = ["Theme","Count"]
        fig_bar = px.bar(tc, x="Count", y="Theme", orientation="h",
                         color_discrete_sequence=["#6b7280"])
        fig_bar.update_traces(marker_line_color="white", marker_line_width=1)
        fig_bar.update_layout(**_base_layout(
            xaxis_title="Number of reviews mentioning this theme",
            **_horizontal_bar_axis(tc["Count"].max())))
        st.plotly_chart(fig_bar, use_container_width=True)
        _chart_download(fig_bar, "neutral_themes.png", "Download themes chart")

    st.markdown("**Sample neutral / mixed reviews**")
    tc_name = _text_col(neu_df)
    hc, bc  = st.columns([3, 1])
    with bc:
        st.button("Refresh", key="neu_refresh")
    for _, row in neu_df.sample(min(10, len(neu_df))).iterrows():
        conf  = f" *(confidence: {row['confidence']*100:.0f}%)*" if "confidence" in row else ""
        label = "Mixed" if ("is_mixed" in row and row["is_mixed"]) else "Neutral"
        st.warning(f'**[{label}]** "{row[tc_name]}"{conf}')

    _report_download(df, key="rpt_neutral")


def page_themes(df):
    if "themes" not in df.columns:
        st.warning("No themes column found. Run the analysis pipeline first.")
        return

    exp_all = df["themes"].str.split(r",\s*").explode().str.strip()
    exp_all = exp_all[~exp_all.isin(["FAILED","","NOT PROCESSED"])]
    theme_counts = exp_all.value_counts().head(20).reset_index()
    theme_counts.columns = ["Theme","Count"]

    fig_theme_bar = px.bar(theme_counts, x="Count", y="Theme", orientation="h",
                           title="Most Common Themes", color="Count",
                           color_continuous_scale="Viridis")
    fig_theme_bar.update_traces(marker_line_color="white", marker_line_width=1)
    fig_theme_bar.update_layout(**_base_layout(
        xaxis_title="Number of mentions", coloraxis_showscale=False,
        **_horizontal_bar_axis(theme_counts["Count"].max())))
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
        scols = [c for c in ["positive","negative","neutral","neutral/mixed"]
                 if c in pivot.columns]
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
                extra_stop = {"review","user","star","stars","https","http",
                               "amp","just","like","im"}
                sw  = list(set(sk_text.ENGLISH_STOP_WORDS) | extra_stop)
                tv  = TfidfVectorizer(ngram_range=(2,3), stop_words=sw)
                mat = tv.fit_transform(list(sent_docs.values()))
                idx = list(sent_docs.keys()).index(selected)
                dense  = mat.toarray()
                scores = dense[idx] * (dense.argmax(axis=0) == idx)
                top_i  = scores.argsort()[-10:][::-1]
                phrases = [tv.get_feature_names_out()[i] for i in top_i if scores[i] > 0]
                pcounts = [sent_docs[selected].count(p) for p in phrases]
                pf = pd.DataFrame({"Phrase":phrases,"Count":pcounts})\
                       .sort_values("Count", ascending=True)
                fig3 = px.bar(pf, x="Count", y="Phrase", orientation="h")
                fig3.update_traces(marker_line_color="white", marker_line_width=1)
                fig3.update_layout(**_base_layout(
                    xaxis_title="Mentions", yaxis_title="",
                    **_horizontal_bar_axis(pf["Count"].max())))
                st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.info(f"Could not extract phrases: {e}")

        if "date" in df.columns:
            st.markdown("---")
            st.subheader("Time-Based & Emergent Trends")
            try:
                theme_time = (
                    df_exp.groupby([pd.Grouper(key="date", freq="ME"), "Theme"])
                    .size().reset_index(name="count")
                )
                if len(theme_time["date"].unique()) > 1:
                    st.markdown(f"**Monthly volume for '{selected}'**")
                    sel_tt  = theme_time[theme_time["Theme"] == selected]
                    n_bars  = len(sel_tt)
                    chart_w = max(900, n_bars * 55)

                    fig_t = px.bar(sel_tt, x="date", y="count",
                                   labels={"date":"Month","count":"Mentions"})
                    fig_t.update_traces(
                        marker_color="#065a82",
                        marker_line_color="white",
                        marker_line_width=2,
                        width=1000 * 60 * 60 * 24 * 20,
                    )
                    fig_t.update_layout(
                        width=chart_w, height=450, bargap=0.4,
                        paper_bgcolor="white", plot_bgcolor="white",
                        margin=dict(l=50, r=20, t=45, b=130),
                        xaxis=dict(
                            tickformat="%b %Y", tickmode="array",
                            tickvals=sel_tt["date"].tolist(),
                            tickangle=-45, tickfont=dict(size=11),
                            title="Month", showgrid=False,
                        ),
                        yaxis=dict(
                            rangemode="tozero", range=[0, 20], dtick=2,
                            tickformat="d", title="Mentions",
                            showgrid=True, gridcolor="#e5e7eb",
                        ),
                    )
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
                    emer["Change"]     = (emer["count_curr"] - emer["count_prev"]).astype(int)
                    emer["count_curr"] = emer["count_curr"].astype(int)
                    emer["count_prev"] = emer["count_prev"].astype(int)
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

        st.markdown("---")
        st.markdown("**Read actual reviews by theme + sentiment**")
        dd_col1, dd_col2 = st.columns(2)
        with dd_col1:
            dd_theme = st.selectbox("Theme:", unique_themes, key="dd_theme")
        with dd_col2:
            dd_sent  = st.selectbox(
                "Sentiment:", ["negative","positive","neutral","neutral/mixed"],
                key="dd_sent")
        dd_data = df_exp[(df_exp["Theme"]==dd_theme) &
                         (df_exp["predicted_sentiment"]==dd_sent)]
        if dd_data.empty:
            st.info(f"No {dd_sent} reviews found for '{dd_theme}'.")
        else:
            hc, bc = st.columns([3, 1])
            with bc:
                st.button("Refresh", key="dd_refresh")
            tc_name = _text_col(dd_data)
            for rev in dd_data.sample(min(5, len(dd_data)))[tc_name].tolist():
                st.info(f'"{rev}"')

    _report_download(df, key="rpt_themes")


def page_outliers(df):
    st.markdown("### Outlier & Edge-Case Reviews")
    st.write(
        "**Low-confidence predictions** are reviews the model was unsure about. "
        "**Mixed-signal reviews** contain both positive and negative language.")

    if "confidence" in df.columns:
        st.markdown("---")
        st.markdown("#### Low-Confidence Predictions")
        threshold = st.slider(
            "Show reviews below this confidence level:", 0.40, 0.90, 0.60, 0.05,
            help="60% means the model was only 60% sure of its prediction.")
        low_conf = df[df["confidence"] < threshold].copy()
        st.markdown(
            f"**{len(low_conf):,} reviews** predicted with less than "
            f"**{threshold*100:.0f}% confidence** — may be worth checking manually.")

        if not low_conf.empty:
            fig_out = px.histogram(
                low_conf, x="confidence", color="predicted_sentiment",
                title="How uncertain was the model? (grouped by predicted sentiment)",
                color_discrete_map=COLOUR_MAP, barmode="group")
            fig_out.update_traces(marker_line_color="white", marker_line_width=2,
                                  xbins=dict(start=0.0, end=1.0, size=0.05))
            fig_out.update_layout(**_base_layout(
                xaxis_title="Model Confidence Score",
                yaxis_title="Number of Reviews",
                legend_title_text="Predicted Sentiment",
                xaxis=dict(
                    tickformat=".0%",
                    range=[-0.05, 1.05],
                    tickvals=[i/10 for i in range(0, 11)],
                    ticktext=[f"{i*10}%" for i in range(0, 11)],
                ),
                yaxis=_hist_yaxis(low_conf),
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
            st.download_button(
                "Download low-confidence reviews",
                data=low_conf.to_csv(index=False),
                file_name="low_confidence_reviews.csv",
                mime="text/csv",
                key="dl_low_conf",
            )

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
            st.download_button(
                "Download mixed-signal reviews",
                data=mixed_df.to_csv(index=False),
                file_name="mixed_signal_reviews.csv",
                mime="text/csv",
                key="dl_mixed",
            )
        else:
            st.info("No mixed-signal reviews detected in this dataset.")

    _report_download(df, key="rpt_outliers")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("Feedback Intelligence Platform")
    st.markdown(
        "Upload a review CSV in the sidebar, click **Run Analysis**, "
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
            df = extract_themes(df, THEMES)
            st.session_state.analyzed_df = df
        except Exception as exc:
            st.error(f"Error during analysis: {exc}")

    if "analyzed_df" in st.session_state:
        df = st.session_state.analyzed_df
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview",
            "Positive Reviews",
            "Neutral Reviews",
            "Negative Reviews",
            "Theme Extraction",
            "Outliers",
        ])
        with tab1: page_overview(df)
        with tab2: page_positive(df)
        with tab3: page_neutral(df)
        with tab4: page_negative(df)
        with tab5: page_themes(df)
        with tab6: page_outliers(df)

    elif not uploaded_file:
        st.info("Upload a CSV file in the sidebar and click Run Analysis to get started.")


if __name__ == "__main__":
    main()

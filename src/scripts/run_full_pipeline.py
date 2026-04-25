"""
Customer Feedback Intelligence Platform - Dashboard
"""

import argparse
import subprocess
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
from collections import Counter

# Optional dependency, falls back gracefully if not installed
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "sample_data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_FILE = OUTPUT_DIR / "sentiment_model.pkl"
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


COLOR_POSITIVE = "#10B981"
COLOR_NEGATIVE = "#EF4444"
COLOR_NEUTRAL = "#9CA3AF"
COLOR_PRIMARY = "#2563EB"
COLOR_WARNING = "#F59E0B"

SENTIMENT_COLOR_MAP = {
    "positive": COLOR_POSITIVE,
    "negative": COLOR_NEGATIVE,
    "neutral": COLOR_NEUTRAL,
}


st.set_page_config(
    page_title="Customer Feedback Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .alert-spike {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .lifecycle-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def call_llm(prompt, model="gemma3:4b"):
    try:
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise e


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
        return "neutral"
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


def build_prompt(batch, themes):
    """
    Build the LLM classification prompt for a batch of reviews.
    """
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
    """
    Send one batch to the local LLM and return themes.
    Retries up to max_retries times if something goes wrong or if hallucinated themes are detected.
    """
    batch_idx, batch = batch_info
    prompt = build_prompt(batch, themes_list)

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(prompt)

            # Extract JSON block (handles both [] or {} bounds)
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

                # Flatten list of dicts to a single dict if LLM wrapped them in an array
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
                    themes.append(idx_themes)

            if len(themes) != len(batch):
                raise ValueError(f"Batch mismatch: LLM returned {len(themes)} arrays, batch has {len(batch)} reviews.")

            # Validate that every review was assigned at least one valid theme
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
                    raise ValueError(f"Hallucination detected: '{theme_list}' not in allowed list")

                validated_themes.append(valid_for_review)

            return batch_idx, batch, validated_themes, "success"

        except Exception as e:
            print(f"Batch {batch_idx} attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)

    return batch_idx, batch, None, "failed"


def extract_themes(df, themes_list, batch_size=30, max_workers=2):
    """
    Extract themes for the reviews in the DataFrame using the LLM.
    """
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
            except Exception as e:
                b = futures[future]
                batch_idx, batch = b
                status = "failed"
                batch_themes = [["FAILED (ERROR)"]] * len(batch)

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
                status_text.text(f"Processing batch {completed_batches}/{total_batches}...")

    themes_lookup = {r["original_idx"]: r["themes"] for r in successful_results + failed_results}
    df["themes"] = [themes_lookup.get(i, "FAILED") for i in range(len(df))]

    initial_len = len(df)
    df = df[~df["themes"].str.contains("FAILED", na=False)]

    progress_bar.progress(1.0)
    status_text.text("Theme extraction complete!")

    return df


def normalize_sentiment_label(s):
    """Map any 'neutral/mixed' label to 'neutral' for clean display."""
    if isinstance(s, str) and s.lower() in ("neutral/mixed", "neutral"):
        return "neutral"
    return s


def get_exploded_themes(df):
    """Explode the comma-separated themes column into one row per (review, theme) pair."""
    df_exp = df.assign(Theme=df['themes'].str.split(r",\s*")).explode('Theme')
    df_exp['Theme'] = df_exp['Theme'].str.strip()
    df_exp = df_exp[~df_exp['Theme'].isin(["FAILED", "", "NOT PROCESSED"])]
    df_exp = df_exp[df_exp['Theme'].notna()]
    return df_exp.reset_index(drop=True)


def kpi_card(label, value, color="#1F2937"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(df):
    total = len(df)
    sentiment_counts = df["predicted_sentiment"].value_counts()
    pos_pct = sentiment_counts.get("positive", 0) / total * 100 if total else 0
    neg_pct = sentiment_counts.get("negative", 0) / total * 100 if total else 0
    neu_pct = (sentiment_counts.get("neutral", 0) + sentiment_counts.get("neutral/mixed", 0)) / total * 100 if total else 0

    avg_stars = df["stars"].mean() if "stars" in df.columns else None

    cols = st.columns(5)
    with cols[0]:
        kpi_card("Total Reviews", f"{total:,}", "#1F2937")
    with cols[1]:
        kpi_card("Positive", f"{pos_pct:.1f}%", COLOR_POSITIVE)
    with cols[2]:
        kpi_card("Negative", f"{neg_pct:.1f}%", COLOR_NEGATIVE)
    with cols[3]:
        kpi_card("Neutral", f"{neu_pct:.1f}%", COLOR_NEUTRAL)
    with cols[4]:
        if avg_stars is not None and not pd.isna(avg_stars):
            kpi_card("Avg Rating", f"{avg_stars:.2f} ", "#1F2937")
        else:
            kpi_card("Themes Found", f"{len(THEMES)}", "#1F2937")


def render_sentiment_pie(df):
    counts = df["predicted_sentiment"].apply(normalize_sentiment_label).value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    fig = px.pie(
        counts, names="sentiment", values="count",
        color="sentiment", color_discrete_map=SENTIMENT_COLOR_MAP,
        hole=0.4,
    )
    fig.update_layout(
        showlegend=True,
        height=350,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


def render_top_compliments_concerns(df_exploded):
    pivot = pd.crosstab(df_exploded['Theme'], df_exploded['predicted_sentiment'].apply(normalize_sentiment_label))

    if pivot.empty:
        st.info("Not enough data for compliments/concerns analysis.")
        return

    pivot['Total'] = pivot.sum(axis=1)
    # Only consider themes with enough reviews to be meaningful
    pivot = pivot[pivot['Total'] >= 5]

    if 'positive' in pivot.columns:
        pivot['positive_pct'] = (pivot['positive'] / pivot['Total'] * 100).round(1)
    else:
        pivot['positive_pct'] = 0

    if 'negative' in pivot.columns:
        pivot['negative_pct'] = (pivot['negative'] / pivot['Total'] * 100).round(1)
    else:
        pivot['negative_pct'] = 0

    top_compliments = pivot.sort_values('positive_pct', ascending=False).head(5)
    top_concerns = pivot.sort_values('negative_pct', ascending=False).head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Compliments")
        st.caption("Themes customers love most")
        comp_df = top_compliments.reset_index()[['Theme', 'positive_pct', 'Total']]
        comp_df.columns = ['Theme', 'Positive %', 'Reviews']

        fig = px.bar(
            comp_df, x='Positive %', y='Theme', orientation='h',
            color='Positive %', color_continuous_scale=['#D1FAE5', '#10B981', '#047857'],
            text='Positive %',
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False,
            xaxis_title="", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top Concerns")
        st.caption("Themes customers complain about most")
        conc_df = top_concerns.reset_index()[['Theme', 'negative_pct', 'Total']]
        conc_df.columns = ['Theme', 'Negative %', 'Reviews']

        fig = px.bar(
            conc_df, x='Negative %', y='Theme', orientation='h',
            color='Negative %', color_continuous_scale=['#FEE2E2', '#EF4444', '#991B1B'],
            text='Negative %',
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False,
            xaxis_title="", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_heatmap(df_exploded):
    pivot = pd.crosstab(
        df_exploded['Theme'],
        df_exploded['predicted_sentiment'].apply(normalize_sentiment_label),
        normalize='index'
    ) * 100

    for col in ['positive', 'neutral', 'negative']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[['positive', 'neutral', 'negative']]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=['Positive', 'Neutral', 'Negative'],
        y=pivot.index,
        colorscale=[
            [0, '#FFFFFF'],
            [0.5, '#FED7AA'],
            [1, '#DC2626']
        ],
        text=pivot.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 12, "color": "black"},
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="% of reviews"),
    ))
    fig.update_layout(
        height=400,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_title="", yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Darker red cells indicate themes with higher concentration of that sentiment.")


def render_sentiment_over_time(df):
    if "date" not in df.columns:
        st.info("No date column found — time trend analysis unavailable.")
        return

    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df = df[df["date"].notna()]

        if df.empty:
            st.info("Date column could not be parsed.")
            return

        df["sentiment_clean"] = df["predicted_sentiment"].apply(normalize_sentiment_label)
        time_df = (
            df.groupby([pd.Grouper(key="date", freq="ME"), "sentiment_clean"])
            .size().reset_index(name="count")
        )

        if len(time_df) <= 1:
            st.info("Not enough date variance for a meaningful time trend.")
            return

        fig = px.line(
            time_df, x="date", y="count", color="sentiment_clean",
            color_discrete_map=SENTIMENT_COLOR_MAP,
            labels={"sentiment_clean": "Sentiment", "date": "Month", "count": "Reviews"},
        )
        fig.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render time trend: {e}")


def render_spike_detection(df):
    if "date" not in df.columns:
        return

    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df = df[df["date"].notna()]
        if df.empty:
            return

        df["sentiment_clean"] = df["predicted_sentiment"].apply(normalize_sentiment_label)
        neg_df = df[df["sentiment_clean"] == "negative"]
        monthly = neg_df.groupby(pd.Grouper(key="date", freq="ME")).size().reset_index(name="count")

        if len(monthly) < 3:
            st.info("Not enough monthly data to detect spikes (need at least 3 months).")
            return

        mean = monthly["count"].mean()
        std = monthly["count"].std()
        # 1.5 standard deviations above the mean is the spike cutoff
        threshold = mean + 1.5 * std

        spikes = monthly[monthly["count"] > threshold].sort_values("count", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["date"], y=monthly["count"],
            marker_color=[COLOR_WARNING if c > threshold else COLOR_NEGATIVE for c in monthly["count"]],
            name="Negative reviews",
            hovertemplate='<b>%{x|%b %Y}</b><br>Negative reviews: %{y}<extra></extra>',
        ))
        fig.add_hline(
            y=threshold, line_dash="dash", line_color=COLOR_WARNING,
            annotation_text=f"Spike threshold ({threshold:.0f})",
            annotation_position="top right",
        )
        fig.add_hline(
            y=mean, line_dash="dot", line_color="#9CA3AF",
            annotation_text=f"Average ({mean:.0f})",
            annotation_position="bottom right",
        )
        fig.update_layout(
            height=400,
            margin=dict(t=30, b=20, l=20, r=20),
            xaxis_title="Month", yaxis_title="Negative review count",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        if not spikes.empty:
            st.markdown("#### ⚠️ Detected Spikes")
            for _, row in spikes.head(5).iterrows():
                pct_above = (row["count"] - mean) / mean * 100 if mean > 0 else 0
                st.markdown(f"""
                <div class="alert-spike">
                    <b>{row['date'].strftime('%B %Y')}</b> —
                    {int(row['count'])} negative reviews
                    ({pct_above:+.0f}% above average of {mean:.0f})
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No significant spikes detected. Negative review volume is stable.")
    except Exception as e:
        st.warning(f"Could not run spike detection: {e}")


def classify_theme_lifecycle(df_exploded):
    """Classify each theme as Emerging / Growing / Stable / Declining based on monthly trend."""
    if "date" not in df_exploded.columns:
        return None

    try:
        df_exploded = df_exploded.copy()
        df_exploded["date"] = pd.to_datetime(df_exploded["date"], errors='coerce')
        df_exploded = df_exploded[df_exploded["date"].notna()]
        if df_exploded.empty:
            return None

        theme_time = (
            df_exploded.groupby([pd.Grouper(key="date", freq="ME"), "Theme"])
            .size().reset_index(name="count")
        )

        months = sorted(theme_time["date"].unique())
        if len(months) < 4:
            return None

        # Compare the average mention volume of the first half of the period against the second half
        midpoint = len(months) // 2
        first_half_months = months[:midpoint]
        second_half_months = months[midpoint:]

        results = []
        for theme in theme_time["Theme"].unique():
            theme_data = theme_time[theme_time["Theme"] == theme]
            first_half_avg = theme_data[theme_data["date"].isin(first_half_months)]["count"].mean() or 0
            second_half_avg = theme_data[theme_data["date"].isin(second_half_months)]["count"].mean() or 0
            first_half_presence = (theme_data[theme_data["date"].isin(first_half_months)]["count"] > 0).sum()

            if first_half_presence <= 1 and second_half_avg > 0:
                lifecycle = "🚀 Emerging"
                color = "#3B82F6"
            elif second_half_avg > first_half_avg * 1.3:
                lifecycle = "📈 Growing"
                color = "#10B981"
            elif second_half_avg < first_half_avg * 0.7:
                lifecycle = "📉 Declining"
                color = "#F59E0B"
            else:
                lifecycle = "➡️ Stable"
                color = "#6B7280"

            change_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 100

            results.append({
                "Theme": theme,
                "Lifecycle": lifecycle,
                "Color": color,
                "First Half Avg": round(first_half_avg, 1),
                "Second Half Avg": round(second_half_avg, 1),
                "Change %": round(change_pct, 1),
            })

        return pd.DataFrame(results).sort_values("Change %", ascending=False)
    except Exception:
        return None


def render_theme_lifecycle(df_exploded):
    lifecycle_df = classify_theme_lifecycle(df_exploded)
    if lifecycle_df is None or lifecycle_df.empty:
        st.info("Not enough time-spread data for lifecycle analysis (need at least 4 distinct months).")
        return

    st.markdown("#### Theme Lifecycle Classification")
    st.caption("How each theme's mention volume has changed from the first half to the second half of the analyzed period.")

    for _, row in lifecycle_df.iterrows():
        cols = st.columns([2, 1.5, 2, 2])
        with cols[0]:
            st.markdown(f"**{row['Theme']}**")
        with cols[1]:
            st.markdown(
                f"<span class='lifecycle-badge' style='background-color:{row['Color']}20; color:{row['Color']};'>{row['Lifecycle']}</span>",
                unsafe_allow_html=True
            )
        with cols[2]:
            st.markdown(f"Avg mentions/mo: **{row['First Half Avg']} → {row['Second Half Avg']}**")
        with cols[3]:
            change_color = "#10B981" if row['Change %'] > 0 else "#EF4444" if row['Change %'] < 0 else "#6B7280"
            st.markdown(f"<span style='color:{change_color}; font-weight:600;'>{row['Change %']:+.1f}%</span>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:0.3rem 0; border-color:#F3F4F6;'>", unsafe_allow_html=True)


def render_emergent_themes(df_exploded):
    if "date" not in df_exploded.columns:
        return

    try:
        df_exploded = df_exploded.copy()
        df_exploded["date"] = pd.to_datetime(df_exploded["date"], errors='coerce')
        df_exploded = df_exploded[df_exploded["date"].notna()]
        if df_exploded.empty:
            return

        theme_time = (
            df_exploded.groupby([pd.Grouper(key="date", freq="ME"), "Theme"])
            .size().reset_index(name="count")
        )
        months = sorted(theme_time["date"].unique())

        if len(months) < 2:
            st.info("Need at least 2 months of data for momentum analysis.")
            return

        curr_month = months[-1]
        prev_month = months[-2]

        curr = theme_time[theme_time["date"] == curr_month].set_index("Theme")
        prev = theme_time[theme_time["date"] == prev_month].set_index("Theme")

        emergent = curr[['count']].join(prev[['count']], lsuffix='_curr', rsuffix='_prev', how='outer').fillna(0)
        emergent['Change'] = emergent['count_curr'] - emergent['count_prev']
        emergent = emergent.sort_values('Change', ascending=False)

        st.markdown(f"**Momentum: {prev_month.strftime('%b %Y')} → {curr_month.strftime('%b %Y')}**")
        emergent_display = emergent.rename(columns={
            'count_prev': 'Previous Month',
            'count_curr': 'Current Month',
            'Change': 'Momentum',
        })[['Previous Month', 'Current Month', 'Momentum']]

        st.dataframe(
            emergent_display.style.format({
                'Previous Month': '{:.0f}',
                'Current Month': '{:.0f}',
                'Momentum': '{:+.0f}',
            }).background_gradient(subset=['Momentum'], cmap='RdYlGn'),
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Could not compute emergent themes: {e}")


def render_word_cloud(df, sentiment):
    if not WORDCLOUD_AVAILABLE:
        st.info("Word cloud feature requires the 'wordcloud' package. Install it with: `pip install wordcloud`")
        return

    df_filtered = df[df["predicted_sentiment"].apply(normalize_sentiment_label) == sentiment]
    if df_filtered.empty:
        st.info(f"No {sentiment} reviews to build word cloud.")
        return

    text = " ".join(df_filtered["clean_text"].fillna("").astype(str).tolist())
    if not text.strip():
        st.info(f"No text content for {sentiment} word cloud.")
        return

    extra_stop = {"review", "user", "star", "stars", "https", "http", "amp", "just", "like", "im", "ive",
                  "got", "get", "go", "going", "went", "told", "said", "would", "could", "still"}
    stop_words = set(sk_text.ENGLISH_STOP_WORDS) | extra_stop

    color_map = {
        "positive": "Greens",
        "negative": "Reds",
        "neutral": "Greys",
    }

    try:
        wc = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap=color_map.get(sentiment, "viridis"),
            stopwords=stop_words,
            max_words=80,
            relative_scaling=0.5,
            min_font_size=10,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not generate word cloud: {e}")


def render_deep_dive(df, df_exploded):
    unique_themes = sorted(df_exploded['Theme'].unique().tolist())
    if not unique_themes:
        st.info("No themes available for deep dive.")
        return

    col1, col2 = st.columns(2)
    with col1:
        dd_theme = st.selectbox("Select Theme", unique_themes, key="dd_theme_v2")
    with col2:
        dd_sentiment = st.selectbox(
            "Select Sentiment",
            ["negative", "positive", "neutral"],
            key="dd_sentiment_v2"
        )

    df_exploded_clean = df_exploded.copy()
    df_exploded_clean["sentiment_clean"] = df_exploded_clean["predicted_sentiment"].apply(normalize_sentiment_label)

    dd_data = df_exploded_clean[
        (df_exploded_clean['Theme'] == dd_theme) &
        (df_exploded_clean['sentiment_clean'] == dd_sentiment)
    ]

    if dd_data.empty:
        st.info(f"No **{dd_sentiment}** reviews found for **{dd_theme}**.")
        return

    st.markdown(f"<p style='color:#6B7280;'>Found <b>{len(dd_data)}</b> {dd_sentiment} reviews for <b>{dd_theme}</b></p>",
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"##### Top Phrases — {dd_sentiment.title()} reviews of {dd_theme}")
        try:
            sent_data = df_exploded_clean[df_exploded_clean['sentiment_clean'] == dd_sentiment]
            theme_docs = sent_data.groupby('Theme')['clean_text'].apply(
                lambda texts: ' '.join(texts.dropna())
            ).to_dict()

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

                # Only keep phrases that are most distinctive to this theme to prevent bleed-over
                is_primary_theme = (tfidf_dense.argmax(axis=0) == theme_idx)
                exclusive_scores = theme_scores * is_primary_theme

                top_indices = exclusive_scores.argsort()[-10:][::-1]
                top_phrases = [feature_names[i] for i in top_indices if exclusive_scores[i] > 0]

                phrase_counts = []
                for p in top_phrases:
                    count = theme_docs[dd_theme].count(p)
                    phrase_counts.append(count)

                if top_phrases:
                    phrase_df = pd.DataFrame({"Phrase": top_phrases, "Count": phrase_counts})
                    phrase_df = phrase_df.sort_values(by="Count", ascending=True)

                    fig = px.bar(phrase_df, x="Count", y="Phrase", orientation='h')
                    fig.update_traces(marker_color=SENTIMENT_COLOR_MAP.get(dd_sentiment, "#6B7280"))
                    fig.update_layout(
                        height=400,
                        margin=dict(t=10, b=10, l=10, r=10),
                        xaxis_title="Mentions", yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough distinctive phrases for this combination.")
            else:
                st.info("Not enough text to extract phrases.")
        except ValueError:
            st.info("Not enough words to extract meaningful phrases.")
        except Exception as e:
            st.warning(f"Could not extract phrases: {e}")

    with col_b:
        header_col, btn_col = st.columns([2, 1])
        with header_col:
            st.markdown(f"##### Sample {dd_sentiment.title()} Reviews")
        with btn_col:
            # Pressing this triggers a Streamlit rerun, which re-samples 5 fresh reviews
            st.button("🔄 Refresh", help="Load 5 different random reviews",
                     use_container_width=True, key="refresh_reviews_v2")

        sample_reviews = dd_data.sample(min(5, len(dd_data)))
        text_col = "raw_text" if "raw_text" in dd_data.columns else (
            "text" if "text" in dd_data.columns else "clean_text"
        )

        for review_text in sample_reviews[text_col].tolist():
            display_text = str(review_text)
            if len(display_text) > 600:
                display_text = display_text[:600] + "..."
            st.info(f'"{display_text}"')


def render_summary_table(df):
    cols_to_show = ["predicted_sentiment", "confidence", "is_mixed", "themes"]
    available_cols = [c for c in cols_to_show if c in df.columns]
    display_df = df[available_cols].head(10).copy()
    if "predicted_sentiment" in display_df.columns:
        display_df["predicted_sentiment"] = display_df["predicted_sentiment"].apply(normalize_sentiment_label)
    if "confidence" in display_df.columns:
        display_df["confidence"] = display_df["confidence"].round(3)
    st.dataframe(display_df, use_container_width=True)


def render_dashboard(df):
    df_exploded = get_exploded_themes(df)

    render_kpi_cards(df)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "📈 Trends & Spikes",
        "🔍 Deep Dive",
        "📥 Export"
    ])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Sentiment Distribution")
            render_sentiment_pie(df)
        with col2:
            st.markdown("### Theme Frequency")
            exploded = df_exploded["Theme"].value_counts().head(10).reset_index()
            exploded.columns = ["Theme", "Count"]
            fig = px.bar(
                exploded, x="Count", y="Theme", orientation='h',
                color="Count", color_continuous_scale="Blues",
            )
            fig.update_layout(
                height=350,
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_showscale=False,
                xaxis_title="", yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        render_top_compliments_concerns(df_exploded)

        st.markdown("---")
        st.markdown("### Theme × Sentiment Heatmap")
        st.caption("Visual breakdown of how each theme is perceived by customers. Spot problem areas at a glance.")
        render_heatmap(df_exploded)

        st.markdown("---")
        st.markdown("###  Data preview - frist 10 rows")
        render_summary_table(df)

    with tab2:
        st.markdown("### Sentiment Over Time")
        st.caption("Monthly volume of each sentiment class across the analyzed period.")
        render_sentiment_over_time(df)

        st.markdown("---")
        st.markdown("### ⚡ Spike Detection")
        st.caption("Months where negative review volume exceeded 1.5 standard deviations above the average — likely indicates an incident or operational issue.")
        render_spike_detection(df)

        st.markdown("---")
        st.markdown("### Theme Lifecycle")
        render_theme_lifecycle(df_exploded)

        st.markdown("---")
        st.markdown("### Most Recent Month Momentum")
        render_emergent_themes(df_exploded)

    with tab3:
        st.markdown("### What are customers actually saying?")
        st.caption("Filter by theme + sentiment to see distinctive phrases and read actual review verbatims.")
        render_deep_dive(df, df_exploded)

        st.markdown("---")
        st.markdown("### Word Cloud by Sentiment")
        st.caption("Visual representation of the most frequent words used in each sentiment category. Bigger = more frequent.")

        wc_sentiment = st.radio(
            "Choose sentiment for word cloud:",
            ["positive", "negative", "neutral"],
            horizontal=True,
            key="wc_sentiment",
        )
        render_word_cloud(df, wc_sentiment)

    with tab4:
        st.markdown("### Export Analyzed Data")
        st.caption("Download the cleaned analysis results as CSV")

        # Only keep columns that are actually present in the dataframe
        display_cols = []
        for c in ["review_id", "stars", "date", "raw_text", "predicted_sentiment",
                  "confidence", "is_mixed", "themes"]:
            if c in df.columns:
                display_cols.append(c)

        export_df = df[display_cols].copy()
        if "predicted_sentiment" in export_df.columns:
            export_df["predicted_sentiment"] = export_df["predicted_sentiment"].apply(normalize_sentiment_label)
        if "confidence" in export_df.columns:
            export_df["confidence"] = export_df["confidence"].round(3)

        st.markdown("#### Preview of export")
        st.dataframe(export_df.head(10), use_container_width=True)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="Download CSV",
                data=export_df.to_csv(index=False),
                file_name="analysis_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.markdown(f"<p style='color:#6B7280; padding-top:0.5rem;'>{len(export_df):,} reviews × {len(display_cols)} columns</p>",
                       unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Summary Statistics")

        sum_col1, sum_col2, sum_col3 = st.columns(3)

        with sum_col1:
            st.markdown("**Sentiment Counts**")
            sent_counts = df["predicted_sentiment"].apply(normalize_sentiment_label).value_counts()
            for sent, count in sent_counts.items():
                pct = count / len(df) * 100
                st.markdown(f"- **{sent.title()}**: {count:,} ({pct:.1f}%)")

        with sum_col2:
            st.markdown("**Top 3 Themes**")
            top_themes = df_exploded["Theme"].value_counts().head(3)
            for theme, count in top_themes.items():
                st.markdown(f"- **{theme}**: {count:,}")

        with sum_col3:
            st.markdown("**Model Confidence**")
            avg_conf = df["confidence"].mean() if "confidence" in df.columns else 0
            mixed_count = df["is_mixed"].sum() if "is_mixed" in df.columns else 0
            st.markdown(f"- **Avg Confidence**: {avg_conf:.2%}")
            st.markdown(f"- **Mixed Reviews Flagged**: {int(mixed_count):,}")


def main():
    st.markdown('<div class="main-header">Customer Feedback Intelligence Platform</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML-powered sentiment analysis + LLM theme extraction for actionable customer insights</div>',
                unsafe_allow_html=True)

    model, vectorizer, train_accuracy = load_or_train_model()
    if model is None or vectorizer is None:
        st.error("Model could not be loaded or trained.")
        return

    if train_accuracy is not None:
        st.success(f"✅ Model trained successfully with accuracy {train_accuracy:.4f}")

    st.sidebar.markdown("### ⚙️ Controls")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV for analysis",
        type="csv",
        help="CSV should include 'text', 'clean_text', or 'raw_text' column."
    )

    run_button = st.sidebar.button("▶️ Run Analysis", use_container_width=True)

    if st.sidebar.button("🗑️ Reset / Clear", use_container_width=True):
        st.session_state.clear()
        st.success("App cache cleared.")
        st.rerun()

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
    elif not uploaded_file:
        st.info("Upload a CSV file in the sidebar and click **Run Analysis** to begin.")


if __name__ == "__main__":
    main()
"""
pipeline_ui.py
Customer Feedback Intelligence Platform — Dashboard UI

Builds the entire visual dashboard including all charts, tabs, filters,
the Trends & Insights section, the About/Methodology page, and PDF export.

Color palette uses a colorblind-friendly scheme (IBM Color Blind Safe palette):
  Positive  — #0077BB (blue)
  Negative  — #CC3311 (red-orange, distinct from green)
  Neutral   — #BBBBBB (gray)
  Spike     — #EE7733 (orange)
  Highlight — #009988 (teal)

Author: Christian East; February 22 2026
Collaborators: Birajman Tamang, Kelsang Yonjan
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
import io
from fpdf import FPDF

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]

### Colorblind-friendly palette (IBM Color Blind Safe)
CB_POSITIVE  = "#0077BB"   # blue
CB_NEGATIVE  = "#CC3311"   # red-orange
CB_NEUTRAL   = "#BBBBBB"   # gray
CB_MIXED     = "#999999"   # dark gray
CB_ORANGE    = "#EE7733"   # orange (spikes/warnings)
CB_TEAL      = "#009988"   # teal (highlights)

COLOUR_MAP = {
    "positive":      CB_POSITIVE,
    "negative":      CB_NEGATIVE,
    "neutral":       CB_NEUTRAL,
    "neutral/mixed": CB_MIXED,
}

### Global CSS — larger fonts + colorblind-friendly styling
DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 20px !important;
}
.stMarkdown p, .stMarkdown li {
    font-size: 20px !important;
    line-height: 1.8 !important;
    color: #1f2937 !important;
}
.stMarkdown h1 { font-size: 2.4rem !important; font-weight: 700 !important; color: #1e3a5f !important; }
.stMarkdown h2 { font-size: 2.0rem !important; font-weight: 700 !important; color: #1e3a5f !important; }
.stMarkdown h3 { font-size: 1.6rem !important; font-weight: 600 !important; color: #1e3a5f !important; }
.stMarkdown h4 { font-size: 1.3rem !important; font-weight: 600 !important; color: #1e3a5f !important; }

label, .stSelectbox label, .stTextInput label,
.stSlider label, .stCheckbox label, .stTextArea label,
.stRadio label { font-size: 19px !important; font-weight: 600 !important; color: #374151 !important; }

.stButton > button {
    font-size: 18px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 2px solid #1e3a5f !important;
    color: #1e3a5f !important;
    background-color: #ffffff !important;
    padding: 8px 20px !important;
}
.stButton > button:hover {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
}

.stTextInput input, .stTextArea textarea {
    font-size: 18px !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #f0f4ff !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 17px !important;
    font-weight: 600 !important;
    color: #374151 !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #065a82 100%) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: #ffffff !important;
    color: #1e3a5f !important;
    font-weight: 700 !important;
    font-size: 17px !important;
    border-radius: 8px !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #cfe8ff !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: rgba(255,255,255,0.18) !important;
    border: 2px dashed rgba(255,255,255,0.6) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small { color: #ffffff !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {
    fill: #ffffff !important; stroke: #ffffff !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
    border: 1.5px solid #c7d7f0;
    border-radius: 12px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 2px 8px rgba(30,58,95,0.08);
}
[data-testid="stMetricLabel"] { font-size: 16px !important; font-weight: 600 !important; color: #6b7280 !important; }
[data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 700 !important; color: #1e3a5f !important; }
[data-testid="stMetricDelta"] { font-size: 15px !important; color: #6b7280 !important; }
[data-testid="stMetricDelta"] svg { display: none; }

.stDataFrame { font-size: 17px !important; border-radius: 8px !important; }
.stCaption, [data-testid="stCaptionContainer"] { font-size: 16px !important; color: #6b7280 !important; }
.stInfo, .stSuccess, .stWarning, .stError { font-size: 18px !important; border-radius: 8px !important; }
.stExpander summary { font-size: 18px !important; font-weight: 600 !important; }

[data-baseweb="select"] > div { border: none !important; box-shadow: none !important; }
[data-baseweb="select"] { border: 1.5px solid #d1d5db !important; border-radius: 8px !important; }
</style>
"""

AXIS_FONT  = dict(family="Inter, sans-serif", size=16, color="#1e3a5f")
TICK_FONT  = dict(family="Inter, sans-serif", size=14, color="#374151")
CHART_FONT = dict(family="Inter, sans-serif", size=15, color="#1f2937")


### Layout helpers

def _horizontal_bar_axis(max_val, title=""):
    nice_max = max(1, int(max_val) + 1)
    step     = max(1, int(nice_max / 8))
    return dict(
        xaxis=dict(
            range=[0, nice_max + step * 0.5],
            tick0=0, dtick=step, tickformat="d",
            rangemode="tozero", tickfont=TICK_FONT,
            title=dict(text=title, font=AXIS_FONT) if title else {},
        ),
        yaxis=dict(categoryorder="total ascending", tickfont=TICK_FONT),
    )


def _base_layout(**extra):
    layout = dict(
        paper_bgcolor="white", plot_bgcolor="#fafbff",
        margin=dict(l=10, r=20, t=55, b=80),
        bargap=0.2, bargroupgap=0.05,
        font=CHART_FONT,
        title_font=dict(family="Inter, sans-serif", size=19, color="#1e3a5f"),
        legend=dict(
            font=dict(family="Inter, sans-serif", size=15, color="#374151"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb", borderwidth=1,
        ),
    )
    layout.update(extra)
    return layout


def _hist_yaxis(df):
    n       = len(df)
    est_max = max(20, n // 4)
    ceiling = ((est_max // 20) + 1) * 20
    return dict(rangemode="tozero", dtick=20, tickformat="d", range=[0, ceiling], tickfont=TICK_FONT)


def _chart_download(fig, filename, label="Download chart as PNG"):
    try:
        png_bytes = fig.to_image(format="png", scale=2)
        st.download_button(label=label, data=png_bytes, file_name=filename, mime="image/png")
    except Exception:
        st.caption("Install kaleido for chart downloads: pip install kaleido")


### Saved layout helpers

def load_configs():
    CONFIG_FILE = PROJECT_ROOT / "src" / "resources" / "saved_tabs.json"
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(name, modules):
    CONFIG_FILE = PROJECT_ROOT / "src" / "resources" / "saved_tabs.json"
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    configs = load_configs()
    configs[name] = modules
    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f, indent=4)


### Trends & Insights helpers

def normalize_sentiment_label(s):
    """Map neutral/mixed to neutral for simplified analysis."""
    if isinstance(s, str) and s.lower() in ("neutral/mixed", "neutral"):
        return "neutral"
    return s


def get_exploded_themes(df):
    """
    Split the themes column so each review-theme pair gets its own row.
    e.g. a review with 'Customer Service, Speed of Service' becomes two rows.
    """
    if "themes" not in df.columns:
        return pd.DataFrame(columns=["Theme", "predicted_sentiment", "clean_text"])
    df_exp = df.assign(Theme=df["themes"].str.split(r",\s*")).explode("Theme")
    df_exp["Theme"] = df_exp["Theme"].str.strip()
    df_exp = df_exp[~df_exp["Theme"].isin(["FAILED", "", "NOT PROCESSED"])]
    return df_exp[df_exp["Theme"].notna()].reset_index(drop=True)


def render_top_compliments_concerns(df_exploded):
    """Show which themes have the highest positive % and highest negative %."""
    pivot = pd.crosstab(
        df_exploded["Theme"],
        df_exploded["predicted_sentiment"].apply(normalize_sentiment_label)
    )
    if pivot.empty:
        st.info("Not enough data for compliments/concerns analysis.")
        return
    pivot["Total"]        = pivot.sum(axis=1)
    pivot                 = pivot[pivot["Total"] >= 5]
    pivot["positive_pct"] = (pivot.get("positive", 0) / pivot["Total"] * 100).round(1)
    pivot["negative_pct"] = (pivot.get("negative", 0) / pivot["Total"] * 100).round(1)

    top_compliments = pivot.sort_values("positive_pct", ascending=False).head(5)
    top_concerns    = pivot.sort_values("negative_pct",  ascending=False).head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Compliments — themes customers love most**")
        comp_df = top_compliments.reset_index()[["Theme","positive_pct","Total"]]
        comp_df.columns = ["Theme","Positive %","Reviews"]
        fig = px.bar(comp_df, x="Positive %", y="Theme", orientation="h",
                     color="Positive %",
                     color_continuous_scale=["#cce5ff", "#0077BB", "#003f6b"],
                     text="Positive %")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          textfont=dict(size=15))
        fig.update_layout(height=340, margin=dict(t=10,b=10,l=0,r=70),
                          yaxis=dict(categoryorder="total ascending", automargin=True,
                                     tickfont=TICK_FONT),
                          coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top Concerns — themes customers complain about most**")
        conc_df = top_concerns.reset_index()[["Theme","negative_pct","Total"]]
        conc_df.columns = ["Theme","Negative %","Reviews"]
        fig = px.bar(conc_df, x="Negative %", y="Theme", orientation="h",
                     color="Negative %",
                     color_continuous_scale=["#ffd5cc", "#CC3311", "#7a1f0a"],
                     text="Negative %")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          textfont=dict(size=15))
        fig.update_layout(height=340, margin=dict(t=10,b=10,l=0,r=70),
                          yaxis=dict(categoryorder="total ascending", automargin=True,
                                     tickfont=TICK_FONT),
                          coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


def render_heatmap(df_exploded):
    """
    Show a grid of themes vs sentiments colored by percentage.
    Darker = higher concentration of that sentiment for that theme.
    Uses a colorblind-friendly white → teal → dark blue scale.
    """
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
        colorscale=[[0,"#FFFFFF"],[0.5,"#66b2d6"],[1,"#003f6b"]],
        text=pivot.values.round(1),
        texttemplate="%{text}%",
        textfont={"size":14,"color":"black"},
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="% of reviews", tickfont=dict(size=14)),
    ))
    fig.update_layout(height=420, margin=dict(t=20,b=20,l=20,r=20),
                      xaxis=dict(tickfont=TICK_FONT),
                      yaxis=dict(tickfont=TICK_FONT))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Darker blue = higher concentration of that sentiment for that theme.")


def render_spike_detection(df):
    """
    Detect months where negative reviews were unusually high.
    The threshold is set at 1.5 standard deviations above the average.
    """
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
            marker_color=[CB_ORANGE if c > threshold else CB_NEGATIVE for c in monthly["count"]],
            hovertemplate="<b>%{x|%b %Y}</b><br>Negative reviews: %{y}<extra></extra>",
        ))
        fig.add_hline(y=threshold, line_dash="dash", line_color=CB_ORANGE,
                      annotation_text=f"Spike threshold ({threshold:.0f})",
                      annotation_position="top right",
                      annotation_font=dict(size=14))
        fig.add_hline(y=mean, line_dash="dot", line_color="#9CA3AF",
                      annotation_text=f"Average ({mean:.0f})",
                      annotation_position="bottom right",
                      annotation_font=dict(size=14))
        fig.update_layout(height=420, margin=dict(t=30,b=20,l=20,r=20),
                          xaxis=dict(title="Month", tickfont=TICK_FONT,
                                     title_font=AXIS_FONT),
                          yaxis=dict(title="Negative review count", tickfont=TICK_FONT,
                                     title_font=AXIS_FONT),
                          showlegend=False, font=CHART_FONT)
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
    """
    Compare the first half vs second half of the time period for each theme.
    Classifies themes as Emerging, Growing, Stable, or Declining.
    """
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
                lifecycle, color = "Emerging",  CB_TEAL
            elif sh_avg > fh_avg * 1.3:
                lifecycle, color = "Growing",   CB_POSITIVE
            elif sh_avg < fh_avg * 0.7:
                lifecycle, color = "Declining", CB_ORANGE
            else:
                lifecycle, color = "Stable",    CB_NEUTRAL

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
    """Display each theme's lifecycle stage with color-coded labels."""
    st.markdown("**Theme Lifecycle Classification**")
    st.caption(
        "How each theme's mention volume changed from the first half to "
        "the second half of the period."
    )
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
                f'padding:4px 12px; border-radius:10px; font-weight:700; font-size:15px;">'
                f'{row["Lifecycle"]}</span>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(f"Avg/mo: **{row['First Half Avg']} → {row['Second Half Avg']}**")
        with cols[3]:
            clr = CB_TEAL if row["Change %"] > 0 else CB_NEGATIVE if row["Change %"] < 0 else CB_NEUTRAL
            st.markdown(
                f'<span style="color:{clr}; font-weight:700; font-size:16px;">'
                f'{row["Change %"]:+.1f}%</span>',
                unsafe_allow_html=True)
        st.markdown("<hr style='margin:0.3rem 0; border-color:#F3F4F6;'>", unsafe_allow_html=True)


def render_emergent_themes(df_exploded):
    """Show how theme mention volume changed between the two most recent months."""
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
        emer["Change"]     = (emer["count_curr"] - emer["count_prev"]).astype(int)
        emer["count_curr"] = emer["count_curr"].astype(int)
        emer["count_prev"] = emer["count_prev"].astype(int)
        st.markdown(
            f"**Momentum: {months[-2].strftime('%b %Y')} → {months[-1].strftime('%b %Y')}**"
        )
        st.dataframe(
            emer.sort_values("Change", ascending=False)
            [["count_prev","count_curr","Change"]]
            .rename(columns={"count_prev":"Prev Month",
                              "count_curr":"Current Month",
                              "Change":"Momentum"}),
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Could not compute emergent themes: {e}")


def render_word_cloud(df, sentiment):
    """Generate a word cloud for the selected sentiment class."""
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
    extra_stop = {
        "review","user","star","stars","https","http","amp",
        "just","like","im","ive","got","get","go","going",
        "went","told","said","would","could","still"
    }
    stop_words = set(sk_text.ENGLISH_STOP_WORDS) | extra_stop
    # colorblind-friendly colormaps
    color_map = {"positive":"Blues","negative":"Oranges","neutral":"Greys"}
    try:
        wc = WordCloud(
            width=800, height=400, background_color="white",
            colormap=color_map.get(sentiment,"viridis"),
            stopwords=stop_words, max_words=80,
            relative_scaling=0.5, min_font_size=12
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not generate word cloud: {e}")


### About / Methodology page

def render_about():
    """
    Display the About and Methodology page — Group 5 credits,
    platform overview, model details, theme descriptions, and tech stack.
    """
    st.markdown("## About & Methodology")

    # Credits card
    st.markdown(
        '<div style="background:#f0f4ff; border-left:5px solid #1e3a5f; '
        'padding:20px 26px; border-radius:10px; margin-bottom:18px;">'
        '<p style="font-size:20px; font-weight:700; color:#1e3a5f; margin:0 0 8px 0;">'
        'Project By Group 5</p>'
        '<p style="font-size:18px; color:#374151; margin:0 0 12px 0;">'
        'Christian East &nbsp;·&nbsp; Birajman Tamang &nbsp;·&nbsp; Kelsang Yonjan</p>'
        '<p style="font-size:17px; color:#6b7280; margin:0 0 4px 0;">'
        '<strong>CSCI 491</strong></p>'
        '<p style="font-size:17px; color:#6b7280; margin:0;">'
        'Special thanks to <strong>Dr. Jennifer Lavergne</strong> and '
        '<strong>Dr. Lasang Tamang</strong></p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### What This Platform Does")
    st.markdown(
        """
        The Customer Feedback Intelligence Platform analyses customer reviews using two AI systems:

        - **Sentiment Classification** — predicts whether a review is positive, negative, or neutral
        - **Theme Extraction** — identifies which business topics each review is about

        Everything runs locally on your Mac — no data is sent to the internet.
        """
    )

    st.markdown("---")
    st.markdown("### Sentiment Model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Algorithm:** Logistic Regression")
        st.markdown("**Vectorizer:** TF-IDF (5,000 features, 1–2 word phrases)")
        st.markdown("**Training split:** 80% train / 20% test (also tested 75/25)")
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
        | Below 60% | Low — review flagged in Outliers section |
        """
    )

    st.markdown("---")
    st.markdown("### Theme Extraction")
    st.markdown(
        """
        Reviews are sent in batches to a locally-running LLM (Gemma 3 4B via Ollama).
        The model assigns 1–3 themes per review from the approved list only —
        any invented themes are rejected and retried up to 5 times.
        Results are cached so the same CSV skips LLM processing on re-upload.

        **The 8 business themes:**
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
            f'<div style="background:#f8f9fa; border-left:4px solid #1e3a5f; '
            f'padding:10px 16px; border-radius:6px; margin-bottom:6px;">'
            f'<strong style="color:#1e3a5f; font-size:17px;">{theme}</strong> — '
            f'<span style="color:#374151; font-size:16px;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Technology Stack")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("**ML**")
        st.markdown("scikit-learn · Logistic Regression · TF-IDF")
    with tc2:
        st.markdown("**LLM**")
        st.markdown("Ollama · Gemma 3 4B")
    with tc3:
        st.markdown("**Dashboard**")
        st.markdown("Streamlit · Plotly · ReportLab · pandas")

    st.markdown("---")
    st.caption("Customer Feedback Intelligence Platform — CSCI 491 · Group 5")


### Main dashboard renderer

def render_dashboard(df, THEMES):
    """
    Build the full dashboard with all tabs.
    Called from main.py after the CSV has been processed.

    Tabs:
    1. Prediction Summary
    2. Sentiment Over Time
    3. Top Themes
    4. Theme Sentiment
    5. Time-Oriented Trends
    6. Phrases & Reviews
    7. Trends & Insights (spike detection, lifecycle, word cloud, etc.)
    8. About
    """
    # inject CSS
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    ### Sidebar date filter
    st.sidebar.divider()
    st.sidebar.subheader("Date Range Filter")

    has_date = "date" in df.columns
    original_count = len(df)

    if has_date:
        try:
            df["date"] = pd.to_datetime(df["date"])
            min_date   = df["date"].min().date()
            max_date   = df["date"].max().date()
            c1, c2     = st.sidebar.columns(2)
            with c1:
                start_date = st.date_input("From:", value=min_date,
                                            min_value=min_date, max_value=max_date,
                                            key="filter_start_date")
            with c2:
                end_date = st.date_input("To:", value=max_date,
                                          min_value=min_date, max_value=max_date,
                                          key="filter_end_date")
            if start_date > end_date:
                st.sidebar.error("Start date must be before end date.")
            else:
                df = df[(df["date"].dt.date >= start_date) &
                        (df["date"].dt.date <= end_date)]
                filtered_count = len(df)
                if filtered_count < original_count:
                    st.sidebar.success(f"Filtered: {filtered_count:,} of {original_count:,} reviews")
                else:
                    st.sidebar.info("All reviews included")
        except Exception as e:
            st.sidebar.warning(f"Could not parse date column: {e}")
            has_date = False
    else:
        st.sidebar.info("No date column found in your CSV.")

    st.sidebar.divider()

    ### Layout customization
    st.sidebar.subheader("Dashboard Layout")
    available_modules = [
        "Prediction Summary",
        "Overall Sentiment Over Time",
        "Top Extracted Themes",
        "Theme Sentiment Distribution",
        "Time-Oriented Trends",
        "Phrases and Reviews",
    ]

    configs = load_configs()
    if "active_layout" not in st.session_state:
        st.session_state.active_layout = available_modules
    else:
        st.session_state.active_layout = [
            m for m in st.session_state.active_layout if m in available_modules
        ]

    def load_selected_config():
        selected = st.session_state.config_selector
        if selected != "Custom" and selected in configs:
            st.session_state.active_layout = configs[selected]

    st.sidebar.selectbox(
        "Load a Saved Layout:",
        ["Custom"] + list(configs.keys()),
        key="config_selector",
        on_change=load_selected_config,
    )

    selected_modules = st.sidebar.multiselect(
        "Select visualizations:",
        available_modules,
        default=st.session_state.active_layout,
        key="active_layout",
        help="Select modules in the order you want them to appear.",
    )

    grid_layout = st.sidebar.selectbox(
        "Grid Layout:", ["1 Column","2 Columns","3 Columns"], index=0
    )
    col_count = 1 if "1" in grid_layout else (2 if "2" in grid_layout else 3)

    with st.sidebar.expander("Save Current Layout"):
        new_name = st.text_input("Preset Name:", placeholder="e.g. Theme Overview")
        if st.button("Save Layout", use_container_width=True):
            if new_name.strip():
                save_config(new_name.strip(), selected_modules)
                st.success("Saved!")
            else:
                st.warning("Please enter a name.")

    st.sidebar.divider()

    ### Pre-compute exploded themes for any module that needs them
    df_exploded    = None
    unique_themes  = []
    has_themes     = "themes" in df.columns
    if has_themes and any(m in selected_modules for m in available_modules[2:]):
        df_exploded = df.assign(Theme=df["themes"].str.split(r",\s*")).explode("Theme")
        df_exploded["Theme"] = df_exploded["Theme"].str.strip()
        df_exploded = df_exploded[~df_exploded["Theme"].isin(["FAILED","","NOT PROCESSED"])]
        df_exploded = df_exploded.dropna(subset=["Theme","predicted_sentiment"])
        unique_themes = sorted(df_exploded["Theme"].unique().tolist())

    ### Tabs — modular section + fixed Trends & Insights + About
    tab_labels = selected_modules + ["Trends & Insights", "About"]
    tabs       = st.tabs(tab_labels)

    charts_for_report  = {}
    texts_for_report   = {}
    reviews_for_report = {}

    # render selected modules
    if col_count > 1:
        grid_containers = st.columns(col_count)
    else:
        grid_containers = [st.container()]

    module_index = 0

    for tab_i, module in enumerate(selected_modules):
        with tabs[tab_i]:
            time_oriented  = module in ["Overall Sentiment Over Time","Time-Oriented Trends"]
            if time_oriented and col_count > 1:
                active_container = st.container()
            else:
                active_container = grid_containers[module_index % col_count]
                module_index += 1

            with active_container:

                if module == "Prediction Summary":
                    st.subheader("Prediction Summary")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Reviews Analyzed", f"{len(df):,}")
                    if "confidence" in df.columns:
                        m2.metric("Avg Model Confidence", f"{df['confidence'].mean():.1%}")
                    if "is_mixed" in df.columns:
                        mixed_pct = df["is_mixed"].sum() / len(df) * 100
                        m3.metric("Mixed Sentiment Rate", f"{mixed_pct:.1f}%")

                    sentiment_counts = df["predicted_sentiment"].value_counts().reset_index()
                    sentiment_counts.columns = ["sentiment","count"]
                    fig_pie = px.pie(
                        sentiment_counts, names="sentiment", values="count",
                        title="Predicted Sentiment Distribution",
                        color="sentiment", color_discrete_map=COLOUR_MAP, hole=0.4
                    )
                    fig_pie.update_layout(**_base_layout())
                    st.plotly_chart(fig_pie, use_container_width=True)
                    charts_for_report["Predicted Sentiment Distribution"] = fig_pie
                    _chart_download(fig_pie, "sentiment_distribution.png",
                                    "Download pie chart")

                    if "confidence" in df.columns:
                        fig_box = px.box(
                            df, x="confidence", y="predicted_sentiment",
                            color="predicted_sentiment",
                            title="Prediction Confidence by Sentiment Class",
                            labels={"confidence":"Model Confidence Score",
                                    "predicted_sentiment":"Sentiment"},
                            color_discrete_map=COLOUR_MAP,
                        )
                        fig_box.update_layout(**_base_layout(
                            xaxis=dict(title=dict(text="Model Confidence Score",
                                                  font=AXIS_FONT), tickfont=TICK_FONT),
                            yaxis=dict(title=dict(text="Sentiment", font=AXIS_FONT),
                                       tickfont=TICK_FONT),
                        ))
                        st.plotly_chart(fig_box, use_container_width=True)
                        charts_for_report["Prediction Confidence"] = fig_box

                    with st.expander("View Raw Prediction Output"):
                        cols_to_show = (
                            ["date","predicted_sentiment","confidence","is_mixed"]
                            if "date" in df.columns
                            else ["predicted_sentiment","confidence","is_mixed"]
                        )
                        if has_themes:
                            cols_to_show.append("themes")
                        cols_to_show.append("clean_text")
                        cols_to_show = [c for c in cols_to_show if c in df.columns]
                        st.dataframe(df[cols_to_show].head(50), use_container_width=True)

                elif module == "Overall Sentiment Over Time":
                    st.subheader("Overall Sentiment Over Time")
                    if "date" in df.columns:
                        try:
                            df["date"] = pd.to_datetime(df["date"])
                            time_df = (
                                df.groupby([pd.Grouper(key="date", freq="ME"),
                                            "predicted_sentiment"])
                                .size().reset_index(name="count")
                            )
                            if len(time_df) > 1:
                                fig = px.line(
                                    time_df, x="date", y="count",
                                    color="predicted_sentiment",
                                    title="Monthly Sentiment Progression",
                                    color_discrete_map=COLOUR_MAP,
                                    markers=True,
                                )
                                fig.update_layout(**_base_layout(
                                    xaxis=dict(title=dict(text="Date", font=AXIS_FONT),
                                               tickfont=TICK_FONT),
                                    yaxis=dict(title=dict(text="Number of Reviews",
                                                          font=AXIS_FONT),
                                               tickfont=TICK_FONT,
                                               rangemode="tozero", range=[0,None],
                                               dtick=1, tickformat="d"),
                                    legend=dict(title=dict(text="Sentiment"),
                                                font=TICK_FONT),
                                ))
                                st.plotly_chart(fig, use_container_width=True)
                                charts_for_report["Monthly Sentiment Progression"] = fig
                                _chart_download(fig, "sentiment_over_time.png",
                                                "Download trend chart")
                            else:
                                st.info("Not enough date variance for a time series.")
                        except Exception:
                            st.info("Date column found but could not be parsed.")
                    else:
                        st.warning("No date column found.")

                elif module == "Top Extracted Themes":
                    st.subheader("Top Extracted Themes")
                    if has_themes and df_exploded is not None and not df_exploded.empty:
                        theme_summary = (df_exploded["Theme"].value_counts()
                                         .head(20).reset_index())
                        theme_summary.columns = ["Theme","Count"]
                        c1, c2 = st.columns([1,2])
                        with c1:
                            st.dataframe(theme_summary, use_container_width=True)
                        with c2:
                            fig = px.bar(
                                theme_summary, x="Count", y="Theme", orientation="h",
                                title="Most Common Review Themes",
                                color="Count",
                                color_continuous_scale=[
                                    "#cce5ff","#66b2d6","#0077BB","#005a8e"
                                ],
                            )
                            fig.update_layout(**_base_layout(
                                coloraxis_showscale=False,
                                **_horizontal_bar_axis(
                                    theme_summary["Count"].max(),
                                    title="Number of Mentions")
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                            charts_for_report["Top Themes"] = fig
                            _chart_download(fig, "top_themes.png", "Download chart")
                    else:
                        st.warning("Run LLM theme extraction first.")

                elif module == "Theme Sentiment Distribution":
                    st.subheader("Theme Sentiment Distribution")
                    if has_themes and df_exploded is not None and not df_exploded.empty:
                        selected_theme = st.selectbox(
                            "Select a Theme:", unique_themes, key="dist_theme"
                        )
                        theme_data = df_exploded[df_exploded["Theme"] == selected_theme]
                        tc = theme_data["predicted_sentiment"].value_counts().reset_index()
                        tc.columns = ["sentiment","count"]
                        fig_dist = px.pie(
                            tc, names="sentiment", values="count",
                            title=f"Sentiment for '{selected_theme}'",
                            color="sentiment", color_discrete_map=COLOUR_MAP,
                        )
                        fig_dist.update_layout(**_base_layout())
                        st.plotly_chart(fig_dist, use_container_width=True)
                        charts_for_report["Theme Sentiment Distribution"] = fig_dist
                        _chart_download(fig_dist,
                                        f"theme_{selected_theme.replace(' ','_')}.png",
                                        "Download chart")

                        st.markdown(f"**Detailed breakdown — {selected_theme}**")
                        pivot_df = pd.crosstab(
                            df_exploded["Theme"].values,
                            df_exploded["predicted_sentiment"].values
                        )
                        pivot_df.index.name    = "Theme"
                        pivot_df.columns.name = "Predicted Sentiment"
                        pivot_df["Total"] = pivot_df.sum(axis=1)
                        raw_counts = df["predicted_sentiment"].value_counts()
                        total_row  = {
                            col: raw_counts.get(col, 0)
                            for col in pivot_df.columns if col != "Total"
                        }
                        total_row["Total"] = sum(total_row.values())
                        pivot_df.loc["Total Raw Reviews"] = pd.Series(total_row)
                        cols_to_pct = [
                            c for c in ["positive","negative","neutral","neutral/mixed"]
                            if c in pivot_df.columns
                        ]
                        for col in cols_to_pct:
                            pivot_df[col+" (%)"] = (
                                pivot_df[col] / pivot_df["Total"] * 100
                            ).round(1)
                        ordered_cols = []
                        for c in cols_to_pct:
                            ordered_cols.extend([c, c+" (%)"])
                        if "Total" in pivot_df.columns:
                            ordered_cols.append("Total")
                        rows_to_keep = [
                            t for t in [selected_theme,"Total Raw Reviews"]
                            if t in pivot_df.index
                        ]
                        st.dataframe(
                            pivot_df.loc[rows_to_keep][ordered_cols],
                            use_container_width=True,
                        )
                    else:
                        st.warning("Run LLM theme extraction first.")

                elif module == "Time-Oriented Trends":
                    st.subheader("Time-Oriented Trends")
                    if has_themes and df_exploded is not None and not df_exploded.empty and "date" in df.columns:
                        try:
                            df_exploded["date"] = pd.to_datetime(df_exploded["date"])
                            trend_df = (
                                df_exploded.groupby(
                                    [pd.Grouper(key="date", freq="ME"), "Theme"]
                                ).size().reset_index(name="count")
                            )
                            fig_trend = px.line(
                                trend_df, x="date", y="count", color="Theme",
                                title="Theme Volume Over Time", markers=True,
                            )
                            fig_trend.update_layout(**_base_layout(
                                xaxis=dict(title=dict(text="Month", font=AXIS_FONT),
                                           tickfont=TICK_FONT),
                                yaxis=dict(title=dict(text="Mentions", font=AXIS_FONT),
                                           tickfont=TICK_FONT,
                                           rangemode="tozero"),
                            ))
                            st.plotly_chart(fig_trend, use_container_width=True)
                            charts_for_report["Time-Oriented Trends"] = fig_trend
                            _chart_download(fig_trend, "theme_trends.png",
                                            "Download chart")
                        except Exception:
                            st.warning("Could not parse dates for trend chart.")
                    else:
                        st.warning("Date column and theme extraction required.")

                elif module == "Phrases and Reviews":
                    st.subheader("Phrases and Review Data")
                    if has_themes and df_exploded is not None and not df_exploded.empty:
                        deep_theme = st.selectbox(
                            "Select Theme:", unique_themes, key="deep_theme"
                        )
                        deep_sentiment = st.selectbox(
                            "Filter by Sentiment:",
                            ["All","positive","negative","neutral","neutral/mixed"],
                            key="deep_sentiment",
                        )
                        deep_df = df_exploded[df_exploded["Theme"] == deep_theme]
                        if deep_sentiment != "All":
                            deep_df = deep_df[deep_df["predicted_sentiment"] == deep_sentiment]
                        review_list = df.loc[deep_df.index.unique()]

                        state_key = f"insights_{deep_theme}_{deep_sentiment}"
                        if state_key not in st.session_state:
                            st.session_state[state_key] = {
                                "insights": None, "top_indices": [], "sampled_df": None
                            }

                        col1, col2 = st.columns([1,2])
                        with col1:
                            st.markdown("**Actionable Information**")
                            if len(review_list) > 0:
                                st.info(f"{len(review_list)} reviews match criteria.")
                                if st.button("Generate LLM Analysis", use_container_width=True):
                                    with st.spinner("Analysing reviews with AI..."):
                                        try:
                                            from pipeline_llm import call_llm
                                            import re
                                            sampled = review_list.dropna(
                                                subset=["clean_text"]
                                            ).sample(min(50, len(review_list)))
                                            texts = sampled["clean_text"].astype(str).tolist()
                                            numbered = "\n".join([
                                                f"Review {i}: {t}"
                                                for i, t in enumerate(texts)
                                            ])
                                            sentiment_ctx = (
                                                f"{deep_sentiment} "
                                                if deep_sentiment != "All" else ""
                                            )
                                            prompt = (
                                                f"Analyse the following {sentiment_ctx}"
                                                f"customer reviews about '{deep_theme}'.\n\n"
                                                f"Task 1: Extract 3-5 specific actionable reasons "
                                                f"why customers feel this way. Use specific language "
                                                f"(not 'bad service' but 'staff ignored customers "
                                                f"at the counter'). Short bullet points.\n"
                                                f"Task 2: Identify the 3 most relevant Review Numbers.\n\n"
                                                f"Format:\nINSIGHTS:\n[bullets]\n"
                                                f"RELEVANT_IDS:\n[number, number, number]\n\n"
                                                f"Reviews:\n{numbered}"
                                            )
                                            raw_response = call_llm(prompt)
                                            insights_text = raw_response
                                            top_indices   = []
                                            if "RELEVANT_IDS:" in raw_response:
                                                parts = raw_response.split("RELEVANT_IDS:")
                                                insights_text = parts[0].replace("INSIGHTS:","").strip()
                                                found = re.findall(r"\d+", parts[1].strip())
                                                for num_str in found:
                                                    idx = int(num_str)
                                                    if 0 <= idx < len(sampled):
                                                        top_indices.append(idx)
                                                top_indices = list(dict.fromkeys(top_indices))[:5]
                                            st.session_state[state_key] = {
                                                "insights": insights_text,
                                                "top_indices": top_indices,
                                                "sampled_df": sampled,
                                            }
                                        except Exception as e:
                                            st.error(f"Could not generate insights: {e}")

                                cached = st.session_state.get(state_key, {})
                                if cached.get("insights"):
                                    st.success("Analysis Complete")
                                    st.markdown(cached["insights"])
                                    insight_key = f"Insights: {deep_theme} ({deep_sentiment})"
                                    texts_for_report[insight_key] = cached["insights"]
                                    top_ids = cached.get("top_indices", [])
                                    llm_df  = cached.get("sampled_df", None)
                                    if llm_df is not None and len(top_ids) > 0:
                                        reviews_for_report[insight_key] = llm_df.iloc[top_ids]
                            else:
                                st.info("No reviews match criteria.")

                        with col2:
                            st.markdown("**Review Explorer**")
                            tab_ai, tab_rand = st.tabs(["Top AI Matches","Random Feed"])

                            def render_feed(feed_df):
                                with st.container(height=500):
                                    for _, row in feed_df.iterrows():
                                        sent = str(row.get("predicted_sentiment","")).upper()
                                        header = f"**{sent}**"
                                        if "stars" in row and pd.notna(row["stars"]):
                                            header += f" | {'★' * int(row['stars'])}"
                                        if "date" in row and pd.notna(row["date"]):
                                            header += f" | {row['date']}"
                                        st.markdown(header)
                                        st.markdown(
                                            f'> *"{row.get("clean_text","")}"*'
                                        )
                                        tags = []
                                        if "themes" in row and pd.notna(row["themes"]):
                                            tags.append(f"**{row['themes']}**")
                                        if "confidence" in row and pd.notna(row["confidence"]):
                                            tags.append(f"Conf: {row['confidence']:.1%}")
                                        if "is_mixed" in row and row["is_mixed"]:
                                            tags.append("Mixed")
                                        if tags:
                                            st.caption(" · ".join(tags))
                                        st.divider()

                            with tab_ai:
                                cached = st.session_state.get(state_key, {})
                                top_ids = cached.get("top_indices", [])
                                llm_df  = cached.get("sampled_df", None)
                                if llm_df is not None and len(top_ids) > 0:
                                    render_feed(llm_df.iloc[top_ids])
                                else:
                                    st.info("Generate LLM analysis first.")

                            with tab_rand:
                                seed_key = f"seed_{state_key}"
                                if seed_key not in st.session_state:
                                    st.session_state[seed_key] = 0
                                if st.button("Refresh Random Feed", key=f"ref_{state_key}"):
                                    st.session_state[seed_key] += 1
                                if len(review_list) > 0:
                                    sample_size = min(10, len(review_list))
                                    rand_df = review_list.sample(
                                        sample_size,
                                        random_state=st.session_state[seed_key]
                                    )
                                    render_feed(rand_df)
                                else:
                                    st.info("No reviews match criteria.")
                    else:
                        st.warning("Run LLM theme extraction first.")

    ### Trends & Insights tab (always shown)
    with tabs[len(selected_modules)]:
        st.markdown("### Trends & Insights")
        st.write(
            "Advanced analytics — spike detection, theme lifecycle, "
            "compliments vs concerns, word clouds and emergent themes."
        )

        trends_exploded = get_exploded_themes(df)

        st.markdown("---")
        st.markdown("#### Top Compliments vs Concerns")
        st.caption(
            "Themes with the highest positive % vs highest negative % — "
            "at least 5 reviews required."
        )
        if not trends_exploded.empty:
            render_top_compliments_concerns(trends_exploded)
        else:
            st.info("Run LLM theme extraction to see this section.")

        st.markdown("---")
        st.markdown("#### Theme × Sentiment Heatmap")
        st.caption(
            "Visual breakdown of how each theme is perceived. "
            "Darker blue = higher concentration of that sentiment."
        )
        if not trends_exploded.empty:
            render_heatmap(trends_exploded)
        else:
            st.info("Run LLM theme extraction to see this section.")

        st.markdown("---")
        st.markdown("#### Negative Review Spike Detection")
        st.caption(
            "Months where negative reviews exceeded 1.5 standard deviations "
            "above average — likely indicates an incident."
        )
        render_spike_detection(df)

        st.markdown("---")
        render_theme_lifecycle(trends_exploded)

        st.markdown("---")
        st.markdown("#### Month-over-Month Theme Momentum")
        render_emergent_themes(trends_exploded)

        st.markdown("---")
        st.markdown("#### Word Cloud by Sentiment")
        st.caption("Most frequent words per sentiment class. Larger = more frequent.")
        wc_sent = st.radio(
            "Choose sentiment:",
            ["positive","negative","neutral"],
            horizontal=True, key="wc_sentiment_trends",
        )
        render_word_cloud(df, wc_sent)

    ### About tab (always shown)
    with tabs[len(selected_modules) + 1]:
        render_about()

    ### PDF report section
    st.markdown("---")
    st.subheader("Download Report")
    st.write("Generate a PDF report with all charts and AI insights from this session.")
    report_title = st.text_input(
        "Report Title:", value="Customer Feedback Intelligence Report",
        key="report_title"
    )
    if st.button("Generate PDF Report", use_container_width=False):
        with st.spinner("Building report..."):
            try:
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("times","B",20)
                safe_title = report_title.encode("latin-1","replace").decode("latin-1")
                pdf.cell(0, 12, safe_title, align="C", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("times","",12)
                pdf.cell(0, 8,
                         f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
                         align="C", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(6)

                for chart_name, fig in charts_for_report.items():
                    pdf.add_page()
                    pdf.set_font("times","B",14)
                    safe_name = chart_name.encode("latin-1","replace").decode("latin-1")
                    pdf.cell(0, 10, safe_name, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(3)
                    try:
                        img_bytes = fig.to_image(format="png", width=900, scale=1.5)
                        img_io    = io.BytesIO(img_bytes)
                        pdf.image(img_io, w=pdf.epw)
                    except Exception:
                        pdf.set_font("times","",11)
                        pdf.cell(0, 8, "(Chart image unavailable — install kaleido)",
                                 new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)

                for title, insight in texts_for_report.items():
                    pdf.add_page()
                    pdf.set_font("times","B",14)
                    safe_t = title.encode("latin-1","replace").decode("latin-1")
                    pdf.cell(0, 10, safe_t, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(3)
                    pdf.set_font("times","",11)
                    safe_i = insight.encode("latin-1","replace").decode("latin-1")
                    pdf.multi_cell(0, 6, safe_i)
                    pdf.ln(5)

                    if title in reviews_for_report:
                        rev_df = reviews_for_report[title]
                        pdf.set_font("times","B",12)
                        pdf.cell(0, 8, "Supporting Reviews:", new_x="LMARGIN", new_y="NEXT")
                        pdf.ln(2)
                        for idx, (_, row) in enumerate(rev_df.iterrows(), 1):
                            if pdf.get_y() > 220:
                                pdf.add_page()
                            pdf.set_font("times","B",11)
                            sent   = str(row.get("predicted_sentiment","")).upper()
                            header = f"Review {idx}: {sent}"
                            if "stars" in row and pd.notna(row["stars"]):
                                header += f" | {int(row['stars'])} stars"
                            if "date" in row and pd.notna(row["date"]):
                                header += f" | {row['date']}"
                            pdf.multi_cell(0, 6,
                                           header.encode("latin-1","replace").decode("latin-1"))
                            pdf.set_font("times","",10)
                            review_text = str(row.get("clean_text",""))
                            pdf.multi_cell(0, 5,
                                           f'"{review_text}"'.encode("latin-1","replace")
                                           .decode("latin-1"))
                            meta = []
                            if "themes" in row and pd.notna(row["themes"]):
                                meta.append(f"Theme: {row['themes']}")
                            if "confidence" in row and pd.notna(row["confidence"]):
                                meta.append(f"Confidence: {row['confidence']:.1%}")
                            if meta:
                                pdf.set_font("times","I",9)
                                pdf.multi_cell(0, 4,
                                               " | ".join(meta).encode("latin-1","replace")
                                               .decode("latin-1"))
                            pdf.ln(3)

                pdf_output = pdf.output()
                st.success("Report ready!")
                st.download_button(
                    label="Download PDF Report",
                    data=bytes(pdf_output),
                    file_name=f"{report_title.replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=False,
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.info("Make sure fpdf is installed: pip install fpdf2")

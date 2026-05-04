import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import io
from fpdf import FPDF
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# loads saved dashboard configurations
def load_configs():
    CONFIG_FILE = PROJECT_ROOT / "src" / "resources" / "saved_tabs.json"
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# saves current dashboard configuration for future use
def save_config(name, modules):
    CONFIG_FILE = PROJECT_ROOT / "src" / "resources" / "saved_tabs.json"
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    configs = load_configs()
    configs[name] = modules
    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f, indent=4)

def render_dashboard(df, THEMES):

    # Global font size increase (+20) and color overrides
    st.markdown("""
        <style>
        /* Base text */
        html, body, [class*="css"] {
            font-size: 135% !important;
        }
        /* Streamlit metric labels and values */
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
            font-weight: 700 !important;
        }
        /* Dataframe / table text */
        .stDataFrame, .stDataFrame table, .stDataFrame td, .stDataFrame th {
            font-size: 1.2rem !important;
            font-weight: 600 !important;
        }
        .stDataFrame th {
            font-weight: 700 !important;
        }
        /* Sidebar text */
        section[data-testid="stSidebar"] * {
            font-size: 20px !important;
        }
        /* Expander headers */
        .streamlit-expanderHeader {
            font-size: 20px !important;
            font-weight: 700 !important;
        }
        /* Tab labels */
        button[data-baseweb="tab"] {
            font-size: 20px !important;
            font-weight: 700 !important;
        }
        /* Caption text */
        .stCaption {
            font-size: 18px !important;
        }
        /* Chart axis labels and titles rendered in DOM */
        .js-plotly-plot .plotly .xtick text,
        .js-plotly-plot .plotly .ytick text {
            font-size: 20px !important;
            font-weight: 700 !important;
        }
        .js-plotly-plot .plotly .g-gtitle text {
            font-size: 24px !important;
            font-weight: 700 !important;
        }
        /* Multiselect tags */
        span[data-baseweb="tag"] {
            background-color: #0072B2 !important;
        }
        /* Multiselect tags */
        span[data-baseweb="tag"] {
            background-color: #0072B2 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Plotly global font template applied to every new figure
    import plotly.io as pio
    pio.templates["custom_large"] = pio.templates["plotly"]
    pio.templates["custom_large"].layout.font = dict(size=18, color="#000000")
    pio.templates["custom_large"].layout.title.font = dict(size=30, color="#1e3a5f")
    pio.templates["custom_large"].layout.xaxis.tickfont = dict(size=20, color="#000000")
    pio.templates["custom_large"].layout.xaxis.title.font = dict(size=30, color="#000000")
    pio.templates["custom_large"].layout.yaxis.tickfont = dict(size=20, color="#000000")
    pio.templates["custom_large"].layout.yaxis.title.font = dict(size=30, color="#000000")
    pio.templates["custom_large"].layout.legend.font = dict(size=30, color="#000000")
    pio.templates["custom_large"].layout.legend.title.font = dict(size=30, color="#000000")
    pio.templates.default = "custom_large"
    # end global overrides

    # date range filter for time-based analysis
    st.sidebar.divider()
    st.sidebar.subheader("Date Range Filter")
    
    has_date_column = "date" in df.columns
    original_count = len(df)
    
    if has_date_column:
        try:

            # ensure date column is datetime
            df["date"] = pd.to_datetime(df["date"])
            
            # get min and max dates from the data
            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            
            # create date range selector (always active)
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input(
                    "From:",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="filter_start_date",
                    help="Select start date:"
                )
            with col2:
                end_date = st.date_input(
                    "To:",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="filter_end_date",
                    help="Select end date:"
                )
            
            # validate date range
            if start_date > end_date:
                st.sidebar.error("The start date must be before end date.")
            else:

                # apply date filter
                df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]
                
                # show filter summary
                filtered_count = len(df)
                if filtered_count < original_count:
                    st.sidebar.success(
                        f"Filtered: {filtered_count:,} of {original_count:,} reviews"
                    )
                else:
                    st.sidebar.info("All reviews are included")
        except Exception as e:
            st.sidebar.warning(f"Could not parse date column: {e}")
            has_date_column = False
    else:
        st.sidebar.info("Your CSV does not contain a date column, or lacks date information.")
    
    st.sidebar.divider()

    st.sidebar.subheader("Dashboard Layout Setup")
    available_modules = [
        "Prediction Summary",
        "Overall Sentiment Over Time",
        "Top Extracted Themes",
        "Theme Sentiment Distribution",
        "Time-Oriented Trends",
        "Phrases and Reviews",
        "Negative Spike Detection",
        "Theme Sentiment Heatmap",
    ]

    charts_for_report = {}
    texts_for_report = {}
    reviews_for_report = {}
    
    configs = load_configs()
    
    if "active_layout" not in st.session_state:
        st.session_state.active_layout = available_modules
    else:

        # cleans session state to prevent errors
        st.session_state.active_layout = [m for m in st.session_state.active_layout if m in available_modules]

    # loads selected config from selectbox, renders wanted visualizations
    def load_selected_config():
        selected = st.session_state.config_selector
        if selected != "Custom" and selected in configs:
            st.session_state.active_layout = configs[selected]

    # selecbox for loading configurations
    st.sidebar.selectbox(
        "Load a Saved Layout:", 
        ["Custom"] + list(configs.keys()), 
        key="config_selector", 
        on_change=load_selected_config
    )

    # multiselect for customizing configurations and layout
    selected_modules = st.sidebar.multiselect(
        "Select and organize visualizations:",
        available_modules,
        default=st.session_state.active_layout,
        key="active_layout",
        help="Remove modules you don't need, or select them in the order you want them to appear on the page."
    )
    
        # allows swapping between 1, 2, and 3 column configurations
    grid_layout = st.sidebar.selectbox(
        "Grid Layout:",
        ["1 Column", "2 Columns", "3 Columns"],
        index=0,
        help="Choose how many columns to display visualizations across"
    )
    
    if "1 Column" in grid_layout:
        col_count = 1
    elif "2 Columns" in grid_layout:
        col_count = 2
    else:
        col_count = 3

    # saves current configuration once named
    with st.sidebar.expander("Save Current Layout"):
        new_layout_name = st.text_input("Preset Name:", placeholder="e.g., Theme Overview")
        if st.button("Save", use_container_width=True):
            if new_layout_name.strip():
                save_config(new_layout_name.strip(), selected_modules)
                st.success(f"Saved!")
            else:
                st.markdown('<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">Please enter a name.</div>', unsafe_allow_html=True)
                


    # calculates exploded dataframe for any modules requiring theme extraction
    df_exploded = None
    unique_themes = []
    has_themes = "themes" in df.columns
    if has_themes and any(m in selected_modules for m in available_modules[2:]):
        df_exploded = df.assign(Theme=df['themes'].str.split(",\\s*")).explode('Theme')
        df_exploded['Theme'] = df_exploded['Theme'].str.strip()
        df_exploded = df_exploded[~df_exploded['Theme'].isin(["FAILED", "", "NOT PROCESSED"])]
        df_exploded = df_exploded.dropna(subset=['Theme', 'predicted_sentiment'])
        unique_themes = sorted(df_exploded['Theme'].unique().tolist())

    if col_count > 1:
        grid_containers = st.columns(col_count)
    else:
        grid_containers = [st.container()]
    
    # index tracking for visualization placement on grid
    module_index = 0  
    
    for module in selected_modules:

        # gives time-oriented visualizations multiple columns of leg-room
        time_oriented = module in ["Overall Sentiment Over Time", "Time-Oriented Trends"]
        
        
        if time_oriented and col_count > 1:
            active_container = st.container()
        else:

            # if not time-oriented, use one column per visualization
            active_container = grid_containers[module_index % col_count]
            module_index += 1

        with active_container:
            if module == "Prediction Summary":
                
                st.subheader("Prediction Summary")

                # basic kpis as styled boxes
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f'<div style="background:#f0f4ff; border:2px solid #1e3a5f; border-radius:12px; '
                        f'padding:20px 24px; text-align:center;">'
                        f'<div style="font-size:30px; font-weight:700; color:#1e3a5f; margin-bottom:8px;">Total Reviews Analyzed</div>'
                        f'<div style="font-size:36px; font-weight:800; color:#000000;">{len(df):,}</div>'
                        f'</div>', unsafe_allow_html=True)
                with m2:
                    if "confidence" in df.columns:
                        st.markdown(
                            f'<div style="background:#f0f4ff; border:2px solid #1e3a5f; border-radius:12px; '
                            f'padding:20px 24px; text-align:center;">'
                            f'<div style="font-size:30px; font-weight:700; color:#1e3a5f; margin-bottom:8px;">Avg Model Confidence</div>'
                            f'<div style="font-size:36px; font-weight:800; color:#000000;">{df["confidence"].mean():.1%}</div>'
                            f'</div>', unsafe_allow_html=True)
                with m3:
                    if "is_mixed" in df.columns:
                        mixed_pct = (df['is_mixed'].sum() / len(df)) * 100
                        st.markdown(
                            f'<div style="background:#f0f4ff; border:2px solid #1e3a5f; border-radius:12px; '
                            f'padding:20px 24px; text-align:center;">'
                            f'<div style="font-size:30px; font-weight:700; color:#1e3a5f; margin-bottom:8px;">Mixed Sentiment Rate</div>'
                            f'<div style="font-size:36px; font-weight:800; color:#000000;">{mixed_pct:.1f}%</div>'
                            f'</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                sentiment_counts = df["predicted_sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["sentiment", "count"]
                fig_pie = px.pie(
                    sentiment_counts,
                    names="sentiment",
                    values="count",
                    title="Predicted Sentiment Distribution",
                    color="sentiment",
                    color_discrete_map={"positive": "#5a9e6f", "neutral": "#8a8a8a", "neutral/mixed": "#8a8a8a", "negative": "#4a6fa5"},
                    hole=0.4
                )
                st.plotly_chart(_apply_font(fig_pie), use_container_width=True)
                charts_for_report["Predicted Sentiment Distribution"] = fig_pie
                
                
                # model intensity boxplot
                if "confidence" in df.columns:
                    fig_box = px.box(
                        df,
                        x="confidence",
                        y="predicted_sentiment",
                        color="predicted_sentiment",
                        title="Prediction Intensity and Uncertainty",
                        labels={"confidence": "Model Confidence Score", "predicted_sentiment": ""},
                        color_discrete_map={"positive": "#5a9e6f", "neutral": "#8a8a8a", "neutral/mixed": "#8a8a8a", "negative": "#4a6fa5"}
                    )
                    st.plotly_chart(_apply_font(fig_box), use_container_width=True)
                    charts_for_report["Prediction Intensity and Uncertainty"] = fig_box
                
                with st.expander("View Raw Prediction Output"):
                    cols_to_show = ["date", "predicted_sentiment", "confidence", "is_mixed"] if "date" in df.columns else ["predicted_sentiment", "confidence", "is_mixed"]
                    if has_themes: cols_to_show.append("themes")
                    cols_to_show.append("clean_text")

                    # filter only to available columns
                    cols_to_show = [c for c in cols_to_show if c in df.columns]
                    st.dataframe(df[cols_to_show].head(50), use_container_width=True)

            elif module == "Overall Sentiment Over Time":

                st.subheader("Overall Sentiment Over Time")
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
                                title="Monthly Sentiment Progression",
                                color_discrete_map={"positive": "#5a9e6f", "neutral": "#8a8a8a", "neutral/mixed": "#8a8a8a", "negative": "#4a6fa5"}
                            )
                            st.plotly_chart(_apply_font(fig), use_container_width=True)
                            charts_for_report["Monthly Sentiment Progression"] = fig
                        else:
                            st.info("Not enough date variance to plot time series.")
                    except Exception:
                        st.info("Date column found but could not be parsed as datetime.")
                else:
                    st.markdown("<div style='background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;'>No date column found for time-series visualization.</div>", unsafe_allow_html=True)

            # barchart for theme density
            elif module == "Top Extracted Themes":

                st.subheader("Top Extracted Themes")
                if has_themes and not df_exploded.empty:
                    theme_summary = df_exploded['Theme'].value_counts().head(20).reset_index()
                    theme_summary.columns = ["Theme", "Count"]

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(theme_summary, use_container_width=True)
                    with col2:
                        # Colorblind-friendly palette (Wong 2011): no red, yellow, pink, orange
                        CB_PALETTE = [
                            "#0072B2", "#56B4E9", "#009E73", "#648FFF",
                            "#785EF0", "#029E73", "#4C72B0", "#64B5F6",
                        ]
                        fig = px.bar(
                            theme_summary,
                            x="Count",
                            y="Theme",
                            orientation='h',
                            title="Most Common Review Themes",
                            color="Theme",
                            color_discrete_sequence=CB_PALETTE,
                        )
                        fig.update_traces(showlegend=False)
                        fig.update_layout(height=400, yaxis={"categoryorder":"total ascending", "title": ""}, margin=dict(t=40, b=40, l=180, r=20))
                        st.plotly_chart(_apply_font(fig), use_container_width=True)
                        charts_for_report["Top Extracted Themes"] = fig
                else:
                    st.markdown('<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">No themes available.</div>', unsafe_allow_html=True)

            # pie chart for sentiments filtered by theme
            elif module == "Theme Sentiment Distribution":

                st.subheader("Theme Sentiment Distribution")
                if has_themes and not df_exploded.empty:
                    st.markdown("Theme Sentiment Distribution")
                    selected_theme = st.selectbox("Select a Theme:", unique_themes, key="dist_theme")
                
                    theme_data = df_exploded[df_exploded['Theme'] == selected_theme]
                    theme_sentiment_counts = theme_data['predicted_sentiment'].value_counts().reset_index()
                    theme_sentiment_counts.columns = ["sentiment", "count"]
                
                    fig_dist = px.pie(
                        theme_sentiment_counts, 
                        names="sentiment", 
                        values="count", 
                        title=f"Sentiment Distribution for '{selected_theme}'",
                        color="sentiment", 
                        color_discrete_map={"positive": "#5a9e6f", "neutral": "#8a8a8a", "neutral/mixed": "#8a8a8a", "negative": "#4a6fa5"}
                    )
                    st.plotly_chart(_apply_font(fig_dist), use_container_width=True)
                    charts_for_report["Theme Sentiment Distribution"] = fig_dist
                
                    st.markdown(f"**Detailed Data: {selected_theme} vs. Total (Counts and Percentages)**")
                    pivot_df = pd.crosstab(df_exploded['Theme'].values, df_exploded['predicted_sentiment'].values)
                    pivot_df.index.name = "Theme"
                    pivot_df.columns.name = "Predicted Sentiment"
                    
                    # row totals
                    pivot_df["Total"] = pivot_df.sum(axis=1)
                    
                    # shows statistics from total raw reviews for comparison
                    raw_counts = df['predicted_sentiment'].value_counts()
                    total_row = {col: raw_counts.get(col, 0) for col in pivot_df.columns if col != "Total"}
                    total_row["Total"] = sum(total_row.values())
                    pivot_df.loc["Total Raw Reviews"] = pd.Series(total_row)
                
                    cols_to_percent = [col for col in ["positive", "negative", "neutral", "neutral/mixed"] if col in pivot_df.columns]
                    for col in cols_to_percent:
                        pivot_df[col + " (%)"] = (pivot_df[col] / pivot_df["Total"] * 100).round(1)
                
                    ordered_cols = []
                    for col in cols_to_percent:
                        ordered_cols.extend([col, col + " (%)"])
                    if "Total" in pivot_df.columns:
                        ordered_cols.append("Total")
                        
                    rows_to_keep = [t for t in [selected_theme, "Total Raw Reviews"] if t in pivot_df.index]
                    pivot_df = pivot_df.loc[rows_to_keep]
                
                    st.dataframe(pivot_df[ordered_cols], use_container_width=True)
                else:
                    st.markdown('<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">No themes available.</div>', unsafe_allow_html=True)

            # time-series graph for review counts filtered on theme
            elif module == "Time-Oriented Trends":
                st.subheader("Time-Oriented Trends")
                if has_themes and not df_exploded.empty and "date" in df.columns:
                    try:
                        df_exploded["date"] = pd.to_datetime(df_exploded["date"])
                        trend_df = (
                            df_exploded.groupby([pd.Grouper(key="date", freq="ME"), "Theme"])
                            .size()
                            .reset_index(name="count")
                        )
                    
                        CB_PALETTE = [
                            "#0072B2", "#56B4E9", "#009E73", "#648FFF",
                            "#785EF0", "#029E73", "#4C72B0", "#64B5F6",
                        ]
                        fig_trend = px.line(
                            trend_df, x="date", y="count", color="Theme",
                            title="Theme Volume Over Time",
                            markers=True,
                            color_discrete_sequence=CB_PALETTE,
                        )
                        st.plotly_chart(_apply_font(fig_trend), use_container_width=True)
                        charts_for_report["Time-Oriented Trends"] = fig_trend
                    except Exception as e:
                        st.markdown('<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">Could not parse dates for trend chart.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">Date column or theme extraction required for time-based trends.</div>', unsafe_allow_html=True)

            # LLM interpretation of review content   
            elif module == "Phrases and Reviews":

                st.subheader("Phrases and Review Data")
                if has_themes and not df_exploded.empty:
                    deep_theme = st.selectbox("Select Theme to Analyze", unique_themes, key="deep_theme")
                    deep_sentiment = st.selectbox(
                        "Filter by Sentiment:", 
                        ["All", "positive", "negative", "neutral", "neutral/mixed"],
                        key="deep_sentiment"
                    )
                
                    deep_df = df_exploded[df_exploded["Theme"] == deep_theme]
                    if deep_sentiment != "All":
                        deep_df = deep_df[deep_df["predicted_sentiment"] == deep_sentiment]
                
                    # finding indexed items, excluding duplicates
                    review_list = df.loc[deep_df.index.unique()]
                    
                    # session states gives state persistence
                    state_key = f"insights_{deep_theme}_{deep_sentiment}"
                    if state_key not in st.session_state:
                        st.session_state[state_key] = {"insights": None, "top_indices": [], "sampled_df": None}
                    if f"random_seed_{state_key}" not in st.session_state:
                        st.session_state[f"random_seed_{state_key}"] = 0
                
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("**Actionable Information**")
                        if len(review_list) > 0:
                            st.info(f"{len(review_list)} reviews match criteria.")
                            if st.button("Generate LLM Analysis", use_container_width=True):
                                with st.spinner("Analyzing reviews with AI..."):
                                    try:
                                        from pipeline_llm import call_llm
                                        import re
                                    
                                        # samples up to 50 reviews for LLM inferences
                                        sampled_df_for_llm = review_list.dropna(subset=['clean_text']).sample(min(50, len(review_list)))
                                        sample_texts = sampled_df_for_llm["clean_text"].astype(str).tolist()
                                        
                                        # review indexing for LLM recall
                                        numbered_reviews = "\n".join([f"Review {i}: {txt}" for i, txt in enumerate(sample_texts)])
                                    
                                        sentiment_context = f"{deep_sentiment} " if deep_sentiment != "All" else ""
                                        prompt = (
                                            f"Analyze the following {sentiment_context}customer reviews focusing specifically on the theme '{deep_theme}'.\n\n"
                                            f"Task 1: Extract the top 3-5 specific, actionable reasons why customers feel this way. "
                                            f"Do not use generic phrases (e.g. bypass 'bad customer service' for 'staff ignored customers at the counter'). "
                                            f"Provide the insights as a clean, short bulleted list.\n"
                                            f"Task 2: Identify the 3 most relevant Review Numbers that best demonstrate these reasons.\n\n"
                                            f"IMPORTANT: You must format your final response with exactly these two sections:\n"
                                            f"INSIGHTS:\n[your bullet points here]\n"
                                            f"RELEVANT_IDS:\n[number, number, number]\n\n"
                                            f"Reviews:\n{numbered_reviews}"
                                        )
                                    
                                        raw_response = call_llm(prompt, model="gemma2:9b")
                                        
                                        # response parsing
                                        insights_text = raw_response
                                        top_indices = []
                                        
                                        if "RELEVANT_IDS:" in raw_response:
                                            parts = raw_response.split("RELEVANT_IDS:")
                                            insights_text = parts[0].replace("INSIGHTS:", "").strip()
                                            ids_text = parts[1].strip()
                                            
                                            # extract the indices
                                            found_nums = re.findall(r'\d+', ids_text)
                                            # index validation
                                            for num_str in found_nums:
                                                idx = int(num_str)
                                                if 0 <= idx < len(sampled_df_for_llm):
                                                    top_indices.append(idx)
                                                    
                                            # no duplicates, 5 max reviews pulled for evidence
                                            top_indices = list(dict.fromkeys(top_indices))[:5]
                                            
                                        st.session_state[state_key] = {
                                            "insights": insights_text,
                                            "top_indices": top_indices,
                                            "sampled_df": sampled_df_for_llm
                                        }
                                    except Exception as e:
                                        st.error(f"Could not generate insights: {e}")
                                        
                            # displays any cached insights
                            cached_data = st.session_state.get(state_key, {})
                            if cached_data.get("insights"):
                                st.success("Analysis Complete")
                                st.markdown(cached_data["insights"])
                                
                                # store insights for PDF export
                                insight_key = f"Insights: {deep_theme} ({deep_sentiment})"
                                texts_for_report[insight_key] = cached_data["insights"]
                                
                                # store associated reviews for PDF export
                                top_ids = cached_data.get("top_indices", [])
                                llm_df = cached_data.get("sampled_df", None)
                                if llm_df is not None and len(top_ids) > 0:
                                    ai_picks = llm_df.iloc[top_ids]
                                    reviews_for_report[insight_key] = ai_picks
                                
                        else:
                            st.info("No reviews match criteria.")
                    with col2:
                        st.markdown("**Review Explorer**")
                        tab1, tab2 = st.tabs(["Top AI Matches", "Random Feed"])
                        
                        def render_feed(feed_df):
                            with st.container(height=500):
                                for _, row in feed_df.iterrows():
                                    sent = str(row.get('predicted_sentiment', 'Unknown')).upper()
                                    
                                    # structure for review display
                                    header_str = f"**{sent}**"
                                    if 'stars' in row and pd.notna(row['stars']):
                                        header_str += f" | {'★' * int(row['stars'])}"
                                    if 'date' in row and pd.notna(row['date']):
                                        header_str += f" | Date: {row['date']}"
                                    
                                    st.markdown(header_str)
                                    st.markdown(f"> *\"{row.get('clean_text', '')}\"*")
                                    
                                    # mapping tags and confidence level
                                    tags = []
                                    if 'themes' in row and pd.notna(row['themes']):
                                        tags.append(f"**{row['themes']}**")
                                    if 'confidence' in row and pd.notna(row['confidence']):
                                        tags.append(f"Conf: {row['confidence']:.1%}")
                                    if 'is_mixed' in row and row['is_mixed']:
                                        tags.append("Mixed")
                                        
                                    if tags:
                                        st.caption(" • ".join(tags))
                                    st.divider()
                                    
                        with tab1:
                            cached_data = st.session_state.get(state_key, {})
                            top_ids = cached_data.get("top_indices", [])
                            llm_df = cached_data.get("sampled_df", None)
                            
                            if cached_data.get("insights") and llm_df is not None and len(top_ids) > 0:

                                # pulls reviews indexed by the LLM
                                st.info("Reviews highlighted by AI as most influential:")
                                ai_picks = llm_df.iloc[top_ids]
                                render_feed(ai_picks)
                            elif cached_data.get("insights"):
                                st.info("AI completed analysis but could not cite specific valid review IDs.")
                            else:
                                st.info("Run LLM analysis to show driving customer data.")
                        
                        with tab2:
                            cols = st.columns([2, 1])
                            with cols[0]:
                                st.markdown(f"(Showing randomly selected reviews)")
                            with cols[1]:
                                if st.button("Refresh Feed", use_container_width=True, key=f"btn_refresh_{state_key}"):
                                    st.session_state[f"random_seed_{state_key}"] += 1
                                    
                            if len(review_list) > 0:
                                cur_seed = st.session_state[f"random_seed_{state_key}"]
                                feed_df = review_list.sample(min(20, len(review_list)), random_state=cur_seed) if len(review_list) > 20 else review_list
                                render_feed(feed_df)
                                
                                # store random feed reviews for optional PDF export
                                random_key = f"Random Reviews: {deep_theme} ({deep_sentiment})"
                                if st.checkbox("Include these reviews in PDF export", key=f"export_random_{state_key}"):
                                    reviews_for_report[random_key] = feed_df
                                elif random_key in reviews_for_report:
                                    del reviews_for_report[random_key]
                            else:
                                st.info("No reviews available.")
                else:
                    st.markdown('<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">Themes required for analysis.</div>', unsafe_allow_html=True)

            elif module == "Negative Spike Detection":
                st.subheader("Negative Review Spike Detection")
                st.markdown('<p style="font-size:18px; color:#000000;">Months where negative reviews exceeded 1.5 standard deviations above average — likely indicates an incident.</p>', unsafe_allow_html=True)
                render_spike_detection(df)

            elif module == "Theme Sentiment Heatmap":
                st.subheader("Theme × Sentiment Heatmap")
                st.caption("Visual breakdown of how each theme is perceived across sentiment classes.")
                if has_themes and df_exploded is not None and not df_exploded.empty:
                    try:
                        render_theme_heatmap(df_exploded)
                    except Exception as e:
                        st.markdown(f'<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">Could not render heatmap: {e}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">Theme extraction required for heatmap.</div>', unsafe_allow_html=True)

    # About & Methodology always visible as a foldable expander at the bottom
    st.markdown("---")
    with st.expander("About & Methodology", expanded=False):
        render_about_page()

    # report builder: allows client to export dashboard information in PDF format
    with st.sidebar:
        st.subheader("Export to PDF Report")
        # allows client to choose a title for their report
        report_title = st.text_input("Title", value="Customer Feedback Report")
        
        # allows client to upload their company logo to personalize report
        logo_file = st.file_uploader(
            "Upload Company Logo (Optional)", 
            type=["png", "jpg", "jpeg"],
            help="Add your company logo to personalize the report"
        )

        if st.button("Export to PDF", use_container_width=True):
            if not charts_for_report and not texts_for_report:
                st.markdown('<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">Please have at least one chart or insight active to export.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Creating PDF..."):
                    pdf = FPDF()
                    pdf.add_page()

                    # sets title to value given by user above
                    pdf.set_font("times", "B", 18)
                    safe_title = report_title.encode('latin-1', 'replace').decode('latin-1')
                    pdf.cell(0, 15, safe_title, align="C", new_x="LMARGIN", new_y="NEXT")
                    
                    # provides the date of report creation
                    pdf.set_font("times", "", 12)
                    current_date = datetime.now().strftime("%B %d, %Y")
                    pdf.cell(0, 8, current_date, align="C", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)
                    
                    # renders company logo
                    if logo_file is not None:
                        try:
                            logo_bytes = io.BytesIO(logo_file.getvalue())

                            # centers the logo in the title page
                            pdf.image(logo_bytes, x=80, y=pdf.get_y(), w=50)
                            pdf.ln(35) 
                        except Exception:

                            # skips logo render if any errors arise
                            pass 
                    else:
                        pdf.ln(10)

                    # renders charts in report, one per page
                    for title, fig in charts_for_report.items():

                        # margins for all charts, preventing visualization cutoff
                        fig.update_layout(

                            # generous margins for all chart types
                            margin=dict(l=200, r=60, t=100, b=80), 
                            font=dict(size=13),
                            title_font_size=19,
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=1.02
                            ),

                            # final margin adjustment
                            yaxis=dict(automargin=True),  
                            xaxis=dict(automargin=True)
                        )
                        
                        # checks for time-oriented visualizations
                        time_oriented = any(keyword in title for keyword in ["Time", "Progression", "Trends", "Over Time"])
                        
                        if time_oriented:
                            
                            # if time-series, renders visualization in landscape orientation
                            pdf.add_page(orientation='L')
                            
                            # sets a larger canvas for landscape visualization
                            img_data = fig.to_image(format="png", width=1800, height=900, scale=2)
                            img_relay = io.BytesIO(img_data)
                            
                            pdf.set_font("times", "B", 14)
                            safe_chart_title = title.encode('latin-1', 'replace').decode('latin-1')
                            pdf.cell(0, 10, safe_chart_title, align="L", new_x="LMARGIN", new_y="NEXT")
                            pdf.ln(5)
                            
                            # renders image
                            pdf.image(img_relay, x=10, y=pdf.get_y(), w=277, h=138)
                        else:

                            # other visualizations are oriented in portrait
                            pdf.add_page()

                            # sets smaller canvas for portrait visualization orientation
                            img_data = fig.to_image(format="png", width=1400, height=1050, scale=2)
                            img_relay = io.BytesIO(img_data)

                            pdf.set_font("times", "B", 14)
                            safe_chart_title = title.encode('latin-1', 'replace').decode('latin-1')
                            pdf.cell(0, 10, safe_chart_title, align="L", new_x="LMARGIN", new_y="NEXT")
                            pdf.ln(5)
                            
                            # renders image
                            pdf.image(img_relay, x=10, y=pdf.get_y(), w=190, h=143)
                    
                    # exports ai review insights to pdf with supporting evidence
                    for title, text_content in texts_for_report.items():
                        pdf.add_page()
                        pdf.set_font("times", "B", 14)
                        safe_text_title = title.encode('latin-1', 'replace').decode('latin-1')
                        pdf.cell(0, 10, safe_text_title, align="L", new_x="LMARGIN", new_y="NEXT")
                        pdf.ln(5)
                        
                        pdf.set_font("times", "", 12)
                        safe_content = text_content.replace('**', '').replace('*', '-').encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 7, safe_content)
                        
                        # include associated reviews as supporting evidence
                        if title in reviews_for_report:
                            pdf.ln(5)
                            pdf.set_font("times", "B", 12)
                            pdf.cell(0, 8, "Relevant Selected Reviews:", new_x="LMARGIN", new_y="NEXT")
                            pdf.ln(3)
                            
                            review_df = reviews_for_report[title]
                            pdf.set_font("times", "", 10)
                            
                            for idx, (_, row) in enumerate(review_df.iterrows(), 1):
                                # check space and add page if needed
                                if pdf.get_y() > 220:
                                    pdf.add_page()
                                
                                # sets back to left margin before each review
                                pdf.set_x(pdf.l_margin)
                                pdf.set_font("times", "B", 10)
                                
                                # review header
                                sent = str(row.get('predicted_sentiment', 'Unknown')).upper()
                                header_parts = [f"Review {idx}: {sent}"]
                                
                                if 'stars' in row and pd.notna(row['stars']):
                                    stars = int(row['stars'])
                                    header_parts.append(f"{stars} stars")
                                if 'date' in row and pd.notna(row['date']):
                                    header_parts.append(f"Date: {row['date']}")
                                
                                header_text = " | ".join(header_parts)
                                safe_header = header_text.encode('latin-1', 'replace').decode('latin-1')
                                pdf.multi_cell(0, 5, safe_header)
                                
                                # check space after header
                                if pdf.get_y() > 220:
                                    pdf.add_page()
                                
                                # back to left margin again
                                pdf.set_x(pdf.l_margin)
                                pdf.set_font("times", "", 10)
                                review_text = row.get('clean_text', '')
                                safe_review = str(review_text).encode('latin-1', 'replace').decode('latin-1')
                                
                                # gets rid of additional spaces
                                pdf.multi_cell(0, 5, f'"{safe_review}"')

                                # check space before metadata
                                if pdf.get_y() > 240:
                                    pdf.add_page()
                                
                                # metadata
                                meta_parts = []
                                if 'themes' in row and pd.notna(row['themes']):
                                    theme_text = str(row['themes']).replace('**', '')
                                    meta_parts.append(f"Theme: {theme_text}")
                                if 'confidence' in row and pd.notna(row['confidence']):
                                    meta_parts.append(f"Confidence: {row['confidence']:.1%}")
                                if 'is_mixed' in row and row['is_mixed']:
                                    meta_parts.append("Mixed Sentiment")
                                
                                if meta_parts:
                                    pdf.set_x(pdf.l_margin)
                                    pdf.set_font("times", "I", 9)
                                    meta_text = " | ".join(meta_parts)
                                    safe_meta = meta_text.encode('latin-1', 'replace').decode('latin-1')
                                    pdf.multi_cell(0, 4, safe_meta)
                                
                                pdf.ln(3)
                    
                    # export any additional random reviews selected for export
                    for title in reviews_for_report:
                        if title not in texts_for_report: 
                            pdf.add_page()
                            pdf.set_font("times", "B", 14)
                            safe_title = title.encode('latin-1', 'replace').decode('latin-1')
                            pdf.cell(0, 10, safe_title, align="L", new_x="LMARGIN", new_y="NEXT")
                            pdf.ln(5)
                            
                            review_df = reviews_for_report[title]
                            pdf.set_font("times", "", 10)
                            
                            for idx, (_, row) in enumerate(review_df.iterrows(), 1):

                                # check space before starting new review
                                if pdf.get_y() > 220:
                                    pdf.add_page()
                                    pdf.set_font("times", "", 10)
                                
                                
                                sent = str(row.get('predicted_sentiment', 'Unknown')).upper()
                                header_parts = [f"Review {idx}: {sent}"]
                                
                                if 'stars' in row and pd.notna(row['stars']):
                                    stars = int(row['stars'])
                                    header_parts.append(f"{stars} stars")
                                if 'date' in row and pd.notna(row['date']):
                                    header_parts.append(f"Date: {row['date']}")
                                
                                header_text = " | ".join(header_parts)
                                safe_header = header_text.encode('latin-1', 'replace').decode('latin-1')
                                pdf.set_font("times", "B", 10)
                                pdf.multi_cell(0, 5, safe_header)
                                
                                # check end of page after end of review
                                if pdf.get_y() > 220:
                                    pdf.add_page()
                                    pdf.set_font("times", "", 10)
                                
                                # review text
                                review_text = row.get('clean_text', '')
                                safe_review = str(review_text).encode('latin-1', 'replace').decode('latin-1')
                                pdf.set_font("times", "", 10)
                                pdf.multi_cell(0, 5, f'  "{safe_review}"')
                                
                                # check space before metadata
                                if pdf.get_y() > 240:
                                    pdf.add_page()
                                    pdf.set_font("times", "", 10)
                                
                                # metadata
                                meta_parts = []
                                if 'themes' in row and pd.notna(row['themes']):
                                    theme_text = str(row['themes']).replace('**', '')
                                    meta_parts.append(f"Theme: {theme_text}")
                                if 'confidence' in row and pd.notna(row['confidence']):
                                    meta_parts.append(f"Confidence: {row['confidence']:.1%}")
                                if 'is_mixed' in row and row['is_mixed']:
                                    meta_parts.append("Mixed Sentiment")
                                
                                if meta_parts:
                                    pdf.set_x(pdf.l_margin)
                                    pdf.set_font("times", "I", 9)
                                    meta_text = " | ".join(meta_parts)
                                    safe_meta = meta_text.encode('latin-1', 'replace').decode('latin-1')
                                    pdf.multi_cell(0, 4, safe_meta)
                                
                                pdf.ln(3)  

                    pdf_output = pdf.output()

                    st.success("Report Complete")
                    st.download_button(
                        label="Download PDF Report",
                        data=bytes(pdf_output),
                        file_name=f"{report_title.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

def _apply_font(fig):
    """Force large black fonts on all chart axes."""
    fig.update_layout(
        font=dict(size=18, color="#000000"),
        title_font=dict(size=30, color="#000000"),
        legend=dict(font=dict(size=30, color="#000000")),
    )
    fig.update_xaxes(tickfont=dict(size=14, color="#000000"), title_font=dict(size=30, color="#000000"))
    fig.update_yaxes(tickfont=dict(size=14, color="#000000"), title_font=dict(size=30, color="#000000"))
    return fig


def render_spike_detection(df):
    """Detect months where negative reviews spiked above 1.5 std devs from average."""
    import numpy as np
    import plotly.graph_objects as go

    if "date" not in df.columns:
        st.info("No date column found — spike detection requires a date column.")
        return
    try:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[d["date"].notna()]
        if d.empty:
            st.info("No valid dates found in dataset.")
            return

        neg_df  = d[d["predicted_sentiment"] == "negative"]
        monthly = neg_df.groupby(pd.Grouper(key="date", freq="ME")).size().reset_index(name="count")

        if len(monthly) < 3:
            st.info("Need at least 3 months of data to detect spikes.")
            return

        mean      = monthly["count"].mean()
        std       = monthly["count"].std()
        threshold = mean + 1.5 * std
        spikes    = monthly[monthly["count"] > threshold].sort_values("count", ascending=False)

        bar_colors = ["#4a6fa5" if c > threshold else "#8a8a8a" for c in monthly["count"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["date"],
            y=monthly["count"],
            marker_color=bar_colors,
            hovertemplate="<b>%{x|%b %Y}</b><br>Negative reviews: %{y}<extra></extra>",
            name="Monthly negatives",
        ))
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="#4a6fa5",
            annotation_text=f"<b>Spike threshold ({threshold:.0f})</b>",
            annotation_position="top right",
            annotation_font=dict(size=13, color="#4a6fa5"),
        )
        fig.add_hline(
            y=mean, line_dash="dot", line_color="black",
            annotation_text=f"<b>Average ({mean:.0f})</b>",
            annotation_position="bottom right",
            annotation_font=dict(size=13, color="black"),
        )
        fig.update_layout(
            height=450,
            title=dict(text="<b>Monthly Negative Review Volume</b>", font=dict(size=24, color="#000000")),
            margin=dict(t=60, b=80, l=180, r=40),
            xaxis=dict(
                title=dict(text="Month", font=dict(size=14, color="#000000")),
                tickfont=dict(size=14, color="#000000"),
            ),
            yaxis=dict(
                title=dict(text="Count", font=dict(size=14, color="#000000")),
                tickfont=dict(size=14, color="#000000"),
                rangemode="tozero",
            ),
            showlegend=False,
        )
        st.plotly_chart(_apply_font(fig), use_container_width=True)

        if not spikes.empty:
            st.markdown("**Detected Spikes:**")
            for _, row in spikes.head(5).iterrows():
                pct_above = (row["count"] - mean) / mean * 100 if mean > 0 else 0
                st.markdown(
                    f'<div style="background:#dbeafe; border-left:4px solid #1e3a5f; '
                    f'padding:12px 18px; border-radius:6px; margin-bottom:8px;">'
                    f'<strong style="color:#1e3a5f;">{row["date"].strftime("%B %Y")}</strong>'
                    f' — {int(row["count"])} negative reviews '
                    f'({pct_above:+.0f}% above average of {mean:.0f})'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No significant spikes detected. Negative review volume is stable.")

    except Exception as e:
        st.markdown(f'<div style="background:#dbeafe; border-left:4px solid #1e3a5f; padding:10px 16px; border-radius:6px; color:#1e3a5f;">Could not run spike detection: {e}</div>', unsafe_allow_html=True)


def render_theme_heatmap(df_exploded):
    """Render a theme × sentiment heatmap using a blue-scale palette."""
    import plotly.graph_objects as go

    def _normalize(s):
        if isinstance(s, str) and s.lower() in ("neutral/mixed", "neutral"):
            return "neutral"
        return s

    # Normalize sentiment and build pivot manually to avoid duplicate label errors
    df_heatmap = df_exploded[["Theme", "predicted_sentiment"]].copy()
    df_heatmap["sentiment_norm"] = df_heatmap["predicted_sentiment"].apply(_normalize)

    # Count occurrences per theme+sentiment, then compute row percentages manually
    counts = (
        df_heatmap.groupby(["Theme", "sentiment_norm"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = counts.groupby("Theme")["count"].transform("sum")
    counts["pct"] = counts["count"] / totals * 100

    # Pivot to wide format — this avoids crosstab duplicate label issues entirely
    pivot = counts.pivot_table(
        index="Theme", columns="sentiment_norm", values="pct", aggfunc="sum", fill_value=0
    )
    pivot.columns.name = None
    pivot.index.name = None

    for col in ["positive", "neutral", "negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["positive", "neutral", "negative"]]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=["<b>Positive</b>", "<b>Neutral</b>", "<b>Negative</b>"],
        y=pivot.index,
        colorscale=[[0, "#FFFFFF"], [0.5, "#93C5FD"], [1, "#1e3a5f"]],
        text=pivot.values.round(1),
        texttemplate="<b>%{text}%</b>",
        textfont=dict(size=13, color="black"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(
            title=dict(text="<b>% of reviews</b>", font=dict(size=14, color="#000000")),
            tickfont=dict(size=14, color="#000000"),
        ),
    ))
    fig.update_layout(
        height=450,
        title=dict(text="<b>Theme × Sentiment Heatmap</b>", font=dict(size=14, color="#000000")),
        margin=dict(t=60, b=20, l=160, r=20),
        xaxis=dict(
            tickfont=dict(size=16, color="#000000"),
            title="",
            side="bottom",
        ),
        yaxis=dict(
            tickfont=dict(size=16, color="#000000"),
            title="",
            automargin=True,
        ),
    )
    st.plotly_chart(_apply_font(fig), use_container_width=True)
    st.caption("Darker blue cells = higher concentration of that sentiment for that theme.")


def render_about_page():
    """Render the About & Methodology page."""

    st.markdown(
        '<div style="background:#f0f4ff; border-left:4px solid #1e3a5f; '
        'padding:18px 22px; border-radius:8px; margin-bottom:16px;">'
        '<p style="font-size:30px; font-weight:700; color:#000000; margin:0 0 8px 0;">'
        'Project By Group 5</p>'
        '<p style="font-size:28px; font-weight:600; color:#000000; margin:0 0 10px 0;">'
        'Christian East &nbsp;·&nbsp; Birajman Tamang &nbsp;·&nbsp; Kelsang Yonjan</p>'
        '<p style="font-size:27px; font-weight:700; color:#000000; margin:0 0 4px 0;">'
        'CSCI 491</p>'
        '<p style="font-size:27px; font-weight:600; color:#000000; margin:0;">'
        'Special thanks to <strong>Dr. Jennifer Lavergne</strong> and '
        '<strong>Dr. Lasang Tamang</strong></p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### What This Platform Does")
    st.markdown(
        '<p style="font-size:28px; color:#000000; font-weight:500;">'
        'The Customer Feedback Intelligence Platform analyses customer reviews using two AI systems:</p>'
        '<ul style="font-size:28px; color:#000000; font-weight:500;">'
        '<li><strong>Sentiment Classification</strong> — predicts whether a review is positive, negative, or neutral</li>'
        '<li><strong>Theme Extraction</strong> — identifies which business topics each review is about</li>'
        '</ul>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Sentiment Model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p style="font-size:27px; color:#000000;"><strong>Algorithm:</strong> Logistic Regression</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:27px; color:#000000;"><strong>Vectorizer:</strong> TF-IDF (5,000 features, 1–2 word phrases)</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:27px; color:#000000;"><strong>Training split:</strong> 80% train / 20% test</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p style="font-size:27px; color:#000000;"><strong>Classes:</strong> Positive · Negative · Neutral/Mixed</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:27px; color:#000000;"><strong>Mixed-signal detection:</strong> contrast words + dual polarity vocabulary</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:27px; color:#000000;"><strong>Confidence score:</strong> model\'s probability for its predicted class</p>', unsafe_allow_html=True)

    st.markdown(
        '<table style="font-size:27px; color:#000000; width:100%; border-collapse:collapse;">'
        '<tr><th style="text-align:left; padding:8px; border:1px solid #ccc; background:#f0f4ff;">Score</th>'
        '<th style="text-align:left; padding:8px; border:1px solid #ccc; background:#f0f4ff;">Meaning</th></tr>'
        '<tr><td style="padding:8px; border:1px solid #ccc;">90–100%</td><td style="padding:8px; border:1px solid #ccc;">Very certain</td></tr>'
        '<tr><td style="padding:8px; border:1px solid #ccc;">70–89%</td><td style="padding:8px; border:1px solid #ccc;">Confident</td></tr>'
        '<tr><td style="padding:8px; border:1px solid #ccc;">60–69%</td><td style="padding:8px; border:1px solid #ccc;">Moderate</td></tr>'
        '<tr><td style="padding:8px; border:1px solid #ccc;">Below 60%</td><td style="padding:8px; border:1px solid #ccc;">Low — review manually</td></tr>'
        '</table>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Theme Extraction")
    st.markdown(
        '<p style="font-size:27px; color:#000000; font-weight:500;">'
        'Reviews are sent in batches to a locally-running LLM (Gemma 2 9B via Ollama). '
        'The model assigns 1–3 themes per review from the approved list only — any invented '
        'themes are rejected and retried up to 5 times.</p>'
        '<p style="font-size:27px; color:#000000; font-weight:700;">The 8 themes:</p>',
        unsafe_allow_html=True,
    )
    theme_descriptions = {
        "Product Quality":      "Food, drink, or item quality and standards",
        "Product Availability": "Out of stock items or limited menu",
        "Customer Service":     "Staff attitude, helpfulness, complaint handling",
        "Speed of Service":     "Wait times, queue length, order delays",
        "Store Environment":    "Cleanliness, atmosphere, seating, parking",
        "Price and Value":      "Cost, affordability, value for money",
        "Digital and Rewards":  "App, online ordering, loyalty points",
        "Policies and Safety":  "Return policies, hygiene, health precautions",
    }
    for theme, desc in theme_descriptions.items():
        st.markdown(
            f'<div style="background:#f0f4ff; border-left:3px solid #1e3a5f; '
            f'padding:10px 16px; border-radius:4px; margin-bottom:6px;">'
            f'<strong style="color:#000000; font-size:27px;">{theme}</strong>'
            f'<span style="color:#000000; font-size:26px;"> — {desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Technology Stack")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown('<p style="font-size:27px; color:#000000;"><strong>ML</strong><br>scikit-learn · Logistic Regression · TF-IDF</p>', unsafe_allow_html=True)
    with tc2:
        st.markdown('<p style="font-size:27px; color:#000000;"><strong>LLM</strong><br>Ollama · Gemma 2 9B</p>', unsafe_allow_html=True)
    with tc3:
        st.markdown('<p style="font-size:27px; color:#000000;"><strong>Dashboard</strong><br>Streamlit · Plotly · fpdf2 · pandas</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Contributions")

    members = [
        {
            "name": "Christian East",
            "role": "Student - Computer Science",
            "contributions": [
                "Researched and selected Ollama for local LLM usage",
                "Designed system architecture flowchart",
                "Built LLM pipeline with retry logic and prompt engineering",
                'Developed "Generate LLM Analysis" feature in the dashboard',
                "Implemented LLM hallucination prevention and batch processing",
                "Created demo video",
            ]
        },
        {
            "name": "Birajman Tamang",
            "role": "Student - Computer Science",
            "contributions": [
                "Researched and created predicted themes",
                "Handled all data cleaning (duplicates, missing values, language filtering)",
                "Developed ML model using Logistic Regression + TF-IDF",
                "Wrote core pipeline integration code",
                "Built mixed sentiment detection logic",
            ]
        },
        {
            "name": "Kelsang Yonjan",
            "role": "Student - Computer Science",
            "contributions": [
                "Extracted Starbucks reviews from Yelp Open Dataset",
                "Recommended and set up Streamlit + Plotly for the dashboard",
                "Built text preprocessing pipeline",
                "Evaluated Logistic Regression model",
                "Designed and built the dashboard UI",
            ]
        },
    ]

    for member in members:
        items = "".join([f'<li style="font-size:27px; color:#000000; margin-bottom:6px;">{c}</li>' for c in member["contributions"]])
        st.markdown(
            f'<div style="background:#f0f4ff; border-left:4px solid #1e3a5f; '
            f'padding:16px 20px; border-radius:8px; margin-bottom:12px;">'
            f'<p style="font-size:30px; font-weight:700; color:#1e3a5f; margin:0 0 4px 0;">{member["name"]}</p>'
            f'<p style="font-size:20px; font-weight:500; color:#000000; margin:0 0 10px 0;">{member["role"]}</p>'
            f'<ul style="margin:0; padding-left:20px;">{items}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<p style="font-size:25px; color:#000000;">Customer Feedback Intelligence Platform — CSCI 491 · Group 5</p>', unsafe_allow_html=True)

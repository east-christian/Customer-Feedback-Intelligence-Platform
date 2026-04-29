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
        "Phrases and Reviews"
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
                st.warning("Please enter a name.")
                


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
                
                # basic kpis
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Reviews Analyzed", f"{len(df):,}")
                if "confidence" in df.columns:
                    m2.metric("Avg Model Confidence", f"{df['confidence'].mean():.1%}")
                if "is_mixed" in df.columns:
                    mixed_pct = (df['is_mixed'].sum() / len(df)) * 100
                    m3.metric("Mixed Sentiment Rate", f"{mixed_pct:.1f}%")

                sentiment_counts = df["predicted_sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["sentiment", "count"]
                fig_pie = px.pie(
                    sentiment_counts,
                    names="sentiment",
                    values="count",
                    title="Predicted Sentiment Distribution",
                    color="sentiment",
                    color_discrete_map={"positive": "green", "neutral": "gray", "neutral/mixed": "gray", "negative": "red"},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                charts_for_report["Predicted Sentiment Distribution"] = fig_pie
                
                
                # model intensity boxplot
                if "confidence" in df.columns:
                    fig_box = px.box(
                        df,
                        x="confidence",
                        y="predicted_sentiment",
                        color="predicted_sentiment",
                        title="Prediction Intensity and Uncertainty",
                        labels={"confidence": "Model Confidence Score", "predicted_sentiment": "Sentiment"},
                        color_discrete_map={"positive": "green", "neutral": "gray", "neutral/mixed": "gray", "negative": "red"}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
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
                                color_discrete_map={"positive": "green", "neutral": "gray", "neutral/mixed": "gray", "negative": "red"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            charts_for_report["Monthly Sentiment Progression"] = fig
                        else:
                            st.info("Not enough date variance to plot time series.")
                    except Exception:
                        st.info("Date column found but could not be parsed as datetime.")
                else:
                    st.warning("No 'date' column found for time-series visualization.")

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
                        fig = px.bar(
                            theme_summary,
                            x="Count",
                            y="Theme",
                            orientation='h',
                            title="Most Common Review Themes",
                            color="Count",
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        charts_for_report["Top Extracted Themes"] = fig
                else:
                    st.warning("No themes available.")

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
                        color_discrete_map={"positive": "green", "neutral": "gray", "neutral/mixed": "gray", "negative": "red"}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
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
                    st.warning("No themes available.")

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
                    
                        fig_trend = px.line(
                            trend_df, x="date", y="count", color="Theme",
                            title="Theme Volume Over Time",
                            markers=True
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                        charts_for_report["Time-Oriented Trends"] = fig_trend
                    except Exception as e:
                        st.warning("Could not parse dates for trend chart.")
                else:
                    st.warning("Date column or theme extraction required for time-based trends.")

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
                    st.warning("Themes required for analysis.")

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
                st.warning("Please have at least one chart or insight active to export.")
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
                            font=dict(size=11),
                            title_font_size=16,
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
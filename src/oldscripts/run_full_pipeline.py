import argparse
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Import our new modularized pipeline components
from pipeline_ml import load_or_train_model, preprocess_reviews, predict_reviews
from pipeline_llm import extract_themes
from pipeline_ui import render_dashboard

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "sample_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Define the THEMES list here
THEMES = [
    "Product Quality",        # Item quality, taste, order accuracy, 
    "Product Availability",   # Stock availability
    "Customer Service",       # Staff attitude, friendliness, support, issue resolution
    "Speed of Service",       # Wait times, drive-thru speed, delivery speed, queues
    "Store Environment",      # Cleanliness, atmosphere, lighting, parking, location 
    "Price & Value",          # Cost, affordability, value for money
    "Digital & Rewards",      # App functionality, website, online ordering, loyalty points
    "Policies & Safety",      # Return policies, health precautions, hygiene standards
]

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

def process_csv(input_path: Path):
    try:
        model, vectorizer, accuracy = load_or_train_model()
        if accuracy is not None:
            print(f"Model trained with new sample data. Accuracy: {accuracy:.4f}")
        else:
            print("Loaded existing model and vectorizer.")
            
        df = pd.read_csv(input_path)
        df = preprocess_reviews(df)
        df = predict_reviews(df, model, vectorizer)
        print(f"Sentiment prediction complete for {len(df)} rows.")

        return df

    except Exception as e:
        print(f"Pipeline error: {e}")
        return None

def run_streamlit_app():
    st.title("🍔 Fast Food Sentiment & Feedback Analysis")
    
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Customer Reviews (CSV)", type="csv")
    
    # Store processed df in session state so re-renders don't trigger re-extraction
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None

    if uploaded_file is not None:
        if st.session_state.processed_df is None or st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            # We have a new or un-processed file
            with st.spinner("Processing file & predicting sentiment..."):
                # Save temp file
                temp_path = OUTPUT_DIR / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                df = process_csv(temp_path)
                
            if df is not None:
                st.session_state.processed_df = df
                st.session_state.last_uploaded_filename = uploaded_file.name
            else:
                st.error("Failed to process the CSV file.")

        df = st.session_state.processed_df

        if df is not None:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Advanced Analysis")

            if st.sidebar.button("Extract Themes via LLM (Slow)", use_container_width=True):
                if "themes" in df.columns and len(df[~df["themes"].str.contains("FAILED", na=False)]) > 0:
                    st.sidebar.success("Themes already extracted!")
                else:
                    df = extract_themes(df, THEMES)
                    st.session_state.processed_df = df
                    st.rerun()

            # Render the modular dashboard
            render_dashboard(df, THEMES)

    else:
        st.info("Please upload a CSV file containing at least a 'text' or 'raw_text' column.")
        
        st.markdown("### Ready for analysis!")
        st.markdown("1. Upload your dataset in the sidebar.")
        st.markdown("2. The system checks models and automatically assigns `predicted_sentiment` + `is_mixed` confidences.")
        st.markdown("3. Run the LLM to generate targeted theme arrays.")
        st.markdown("4. Explore visual breakdowns, save stateful dashboard layouts, and interact with feedback analytics.")

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Pipeline Orchestrator")
    parser.add_argument("--input", type=str, help="Path to input CSV for headless processing")
    parser.add_argument("--no-ui", action="store_true", help="Run without launching Streamlit")
    
    args, unknown = parser.parse_known_args()

    if args.input and args.no_ui:
        print("Running in headless pipeline mode...")
        process_csv(Path(args.input))
    else:
        run_streamlit_app()

if __name__ == "__main__":
    main()
    total_batches = len(batches)
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_themes_with_retry, b, themes_list): b
            for b in batches
        }

        for future in as_completed(futures):
            try:
                # Add a timeout so the app doesn't hang forever if the LLM freezes
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
                    # Extract themes returns carefully validated arrays like: ["Store Environment", "Customer Service"]
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

            # Update progress bar less frequently to prevent Streamlit UI from hanging
            completed_batches += 1
            if completed_batches % max(1, (total_batches // 100)) == 0 or completed_batches == total_batches:
                progress_bar.progress(completed_batches / total_batches)
                status_text.text(f"Processed review batch {completed_batches}/{total_batches}. Please wait, local LLM parsing is intensive...")

    themes_lookup = {r["original_idx"]: r["themes"] for r in successful_results + failed_results}
    df["themes"] = [themes_lookup.get(i, "FAILED") for i in range(len(df))]

    # Drop any reviews that completely failed extraction to ensure data purity
    initial_len = len(df)
    df = df[~df["themes"].str.contains("FAILED", na=False)]
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} reviews because the LLM repeatedly hallucinated or timed out.")

    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("Theme extraction complete!")

    return df

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

def render_dashboard(df):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Layout Setup")
    available_modules = [
        "Prediction Summary",
        "Overall Sentiment Over Time",
        "Top Extracted Themes",
        "Theme Sentiment Breakdown & Distribution",
        "Time-Based & Emergent Trends",
        "Deep Dive Phrases & Reviews"
    ]
    
    configs = load_configs()
    
    if "active_layout" not in st.session_state:
        st.session_state.active_layout = available_modules

    def load_selected_config():
        selected = st.session_state.config_selector
        if selected != "Custom" and selected in configs:
            st.session_state.active_layout = configs[selected]

    st.sidebar.selectbox(
        "Load a Saved Layout:", 
        ["Custom"] + list(configs.keys()), 
        key="config_selector", 
        on_change=load_selected_config
    )

    selected_modules = st.sidebar.multiselect(
        "Select and organize visualizations:",
        available_modules,
        default=st.session_state.active_layout,
        key="active_layout",
        help="Remove modules you don't need, or select them in the order you want them to appear on the page."
    )
    
    with st.sidebar.expander("💾 Save Current Layout"):
        new_layout_name = st.text_input("Preset Name:", placeholder="e.g., Theme Overview")
        if st.button("Save", use_container_width=True):
            if new_layout_name.strip():
                save_config(new_layout_name.strip(), selected_modules)
                st.success(f"Saved!")
            else:
                st.warning("Please enter a name.")

    # Pre-calculate df_exploded if any module requiring themes is active
    df_exploded = None
    unique_themes = []
    has_themes = "themes" in df.columns
    if has_themes and any(m in selected_modules for m in available_modules[2:]):
        df_exploded = df.assign(Theme=df['themes'].str.split(",\\s*")).explode('Theme')
        df_exploded['Theme'] = df_exploded['Theme'].str.strip()
        df_exploded = df_exploded[~df_exploded['Theme'].isin(["FAILED", "", "NOT PROCESSED"])]
        df_exploded = df_exploded.reset_index(drop=True)
        unique_themes = sorted(df_exploded['Theme'].unique().tolist())

    for module in selected_modules:
        if module == "Prediction Summary":
            st.markdown("---")
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

        elif module == "Overall Sentiment Over Time":
            st.markdown("---")
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
                            title="Monthly Sentiment Evolution",
                            color_discrete_map={"positive": "green", "neutral": "gray", "neutral/mixed": "gray", "negative": "red"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough date variance to plot time series.")
                except Exception:
                    st.info("Date column found but could not be parsed as datetime.")
            else:
                st.warning("No 'date' column found for time-series visualization.")

        elif module == "Top Extracted Themes":
            st.markdown("---")
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
            else:
                st.warning("No themes available.")

        elif module == "Theme Sentiment Breakdown & Distribution":
            st.markdown("---")
            st.subheader("Theme Sentiment Breakdown & Distribution")
            if has_themes and not df_exploded.empty:
                st.markdown("**Interactive Theme Sentiment Distribution**")
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
                
                st.markdown("**Detailed Data: Sentiment Breakdown by Theme (Counts & Percentages)**")
                pivot_df = pd.crosstab(df_exploded['Theme'], df_exploded['predicted_sentiment'], margins=True, margins_name="Total")
                
                cols_to_percent = [col for col in ["positive", "negative", "neutral", "neutral/mixed"] if col in pivot_df.columns]
                for col in cols_to_percent:
                    pivot_df[col + " (%)"] = (pivot_df[col] / pivot_df["Total"] * 100).round(1)
                
                ordered_cols = []
                for col in cols_to_percent:
                    ordered_cols.extend([col, col + " (%)"])
                if "Total" in pivot_df.columns:
                    ordered_cols.append("Total")
                
                st.dataframe(pivot_df[ordered_cols], use_container_width=True)
            else:
                st.warning("No themes available.")

        elif module == "Time-Based & Emergent Trends":
            st.markdown("---")
            st.subheader("Time-Based & Emergent Trends")
            if "date" in df.columns and has_themes and not df_exploded.empty:
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
                            selected_theme_trend = st.selectbox("Select Theme to check volume:", unique_themes, key="trend_theme")
                            st.markdown(f"**Monthly Volume Trend for '{selected_theme_trend}'**")
                            sel_theme_time = theme_time[theme_time["Theme"] == selected_theme_trend]
                            
                            fig_trend = px.bar(
                                sel_theme_time,
                                x="date",
                                y="count",
                                title=f"Review mentions of '{selected_theme_trend}' alone",
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
                            
                            st.write(f"Change between **{prev_month.strftime('%b %Y')}** and **{curr_month.strftime('%b %Y')}**:")
                            st.dataframe(
                                rising_themes[['count_prev', 'count_curr', 'Change']]
                                .rename(columns={'count_prev': 'Prev. Mentions', 'count_curr': 'Current Mentions', 'Change': 'Momentum'}),
                                use_container_width=True
                            )
                    else:
                        st.info("The dataset spans less than a full month. Trend momentum cannot be established.")
                except Exception as e:
                    st.warning(f"Could not calculate emergent theme trends: {e}")
            else:
                st.warning("Both 'date' and 'themes' columns are required for this module.")

        elif module == "Deep Dive Phrases & Reviews":
            st.markdown("---")
            st.subheader("Deep Dive: What are customers actually saying?")
            if has_themes and not df_exploded.empty:
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
                            theme_docs = sent_data.groupby('Theme')['clean_text'].apply(lambda texts: ' '.join(texts.dropna())).to_dict()
                            
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
                                fig_phrases = px.bar(
                                    phrase_df, 
                                    x="Count", 
                                    y="Phrase", 
                                    orientation='h',
                                )
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
                            st.button("🔄 Refresh", help="Load different random reviews", use_container_width=True, key=f"btn_{module}")
                        
                        sample_reviews = dd_data.sample(min(5, len(dd_data)))
                        text_col = "text" if "text" in dd_data.columns else "raw_text" if "raw_text" in dd_data.columns else "clean_text"
                        for review_text in sample_reviews[text_col].tolist():
                            st.info(f'"{review_text}"')
            else:
                st.warning("No themes column found.")

    st.markdown("---")
    st.download_button(
        label="Download analyzed data",
        data=df.to_csv(index=False),
        file_name="analysis_results.csv",
        mime="text/csv",
    )

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

    if uploaded_file and run_button:
        try:
            df = pd.read_csv(uploaded_file)
            df = preprocess_reviews(df)
            df = predict_reviews(df, model, vectorizer)
            df = extract_themes(df, THEMES)  # Pass the THEMES list here
            st.session_state.analyzed_df = df
        except Exception as exc:
            st.error(f"Error processing uploaded file: {exc}")

    if "analyzed_df" in st.session_state:
        render_dashboard(st.session_state.analyzed_df)

    if not uploaded_file and "analyzed_df" not in st.session_state:
        st.info("Upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()
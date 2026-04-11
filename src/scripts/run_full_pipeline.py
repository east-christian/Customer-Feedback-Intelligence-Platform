import argparse
import subprocess
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, CountVectorizer
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "sample_data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_FILE = OUTPUT_DIR / "sentiment_model.pkl"
VECTORIZER_FILE = OUTPUT_DIR / "tfidf_vectorizer.pkl"

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

def call_llm(prompt, model="gemma2:9b"):
    """
    Call the Ollama LLM with the given prompt and model.
    """
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

def predict_reviews(df, model, vectorizer):
    tfidf = vectorizer.transform(df["clean_text"])
    preds = model.predict(tfidf)
    probs = model.predict_proba(tfidf)

    df["predicted_sentiment"] = preds
    df["confidence"] = probs.max(axis=1)
    for idx, cls in enumerate(model.classes_):
        df[f"prob_{cls}"] = probs[:, idx]
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
                # Try parsing
                try:
                    parsed_data = json.loads(clean)
                except json.JSONDecodeError:
                    try:
                        parsed_data = ast.literal_eval(clean)
                    except Exception:
                        try:
                            # If it spit out comma-separated dicts without outer brackets
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

                # Convert dict back to an ordered array according to the batch size
                themes = []
                for idx in range(1, len(batch) + 1):
                    key = str(idx)
                    idx_themes = parsed_dict.get(key, [])
                    if not isinstance(idx_themes, list):
                        idx_themes = [idx_themes]
                    themes.append(idx_themes)
            
            if len(themes) != len(batch):
                raise ValueError(f"Batch mismatch: LLM returned {len(themes)} exact theme arrays, but there are {len(batch)} reviews.")

            # Validate that every review was assigned at least one valid theme
            validated_themes = []
            for theme_list in themes:
                valid_for_review = []
                # Handle cases where LLM returns a dictionary inside the list
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

def extract_themes(df, themes_list, batch_size=5, max_workers=2):
    """
    Extract themes for the reviews in the DataFrame using the LLM.
    """
    reviews = df["clean_text"].fillna("").tolist()
    batches = [(i, reviews[i:i + batch_size]) for i in range(0, len(reviews), batch_size)]

    successful_results = []
    failed_results = []

    # Initialize Streamlit progress bar
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

def render_dashboard(df):
    st.subheader("Prediction Summary")
    st.write(df[["predicted_sentiment", "confidence", "themes"]].head(10))

    sentiment_counts = df["predicted_sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        title="Predicted Sentiment Distribution",
        color="sentiment",
        color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)

    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            time_df = (
                df.groupby([pd.Grouper(key="date", freq="M"), "predicted_sentiment"])
                .size()
                .reset_index(name="count")
            )
            fig = px.line(
                time_df,
                x="date",
                y="count",
                color="predicted_sentiment",
                title="Sentiment Over Time",
                color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"}
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Date column found but could not be parsed as datetime.")

    st.subheader("Top Extracted Themes")
    
    if "themes" in df.columns:
        # Explode the themes from comma-separated strings to calculate summary stats
        exploded = df["themes"].str.split(",\\s*").explode().str.strip()
        # Remove FAILED or empty themes
        exploded = exploded[~exploded.isin(["FAILED", "", "NOT PROCESSED"])]
        
        theme_summary_series = exploded.value_counts().head(20)
        theme_summary = theme_summary_series.reset_index()
        theme_summary.columns = ["Theme", "Count"]

        # Make columns layout
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
            
        # Optional: Show cross-tabulation of Themes vs Sentiment
        st.subheader("Theme Sentiment Breakdown & Distribution")
        
        # Explode the entire dataframe on themes to match sentiment to theme
        df_exploded = df.assign(Theme=df['themes'].str.split(",\\s*")).explode('Theme')
        df_exploded['Theme'] = df_exploded['Theme'].str.strip()
        # Remove empty or failed themes
        df_exploded = df_exploded[~df_exploded['Theme'].isin(["FAILED", "", "NOT PROCESSED"])]
        
        # Reset index to eliminate duplicate index labels caused by explode()
        # which breaks pd.crosstab when calculating sentiment distributions
        df_exploded = df_exploded.reset_index(drop=True)
        
        if not df_exploded.empty:
            # 1. Interactive Single-Theme Sentiment Distribution
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
                color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # 2. Detailed Data Table with Percentages
            st.markdown("**Detailed Data: Sentiment Breakdown by Theme (Counts & Percentages)**")
            pivot_df = pd.crosstab(df_exploded['Theme'], df_exploded['predicted_sentiment'], margins=True, margins_name="Total")
            
            # Calculate percentages for whatever sentiment classes exist natively
            cols_to_percent = [col for col in ["positive", "negative", "neutral"] if col in pivot_df.columns]
            for col in cols_to_percent:
                pivot_df[col + " (%)"] = (pivot_df[col] / pivot_df["Total"] * 100).round(1)
            
            # Reorder columns to put counts next to percentages
            ordered_cols = []
            for col in cols_to_percent:
                ordered_cols.extend([col, col + " (%)"])
            if "Total" in pivot_df.columns:
                ordered_cols.append("Total")
            
            st.dataframe(pivot_df[ordered_cols], use_container_width=True)

            # --- 3. Deep Dive: Review Verbatims and Subtopics ---
            st.markdown("---")
            st.subheader("Deep Dive: What are customers actually saying?")
            st.write("Understand the specific subtopics and read actual reviews driving the sentiment for a theme.")
            
            col_dd1, col_dd2 = st.columns(2)
            with col_dd1:
                dd_theme = st.selectbox("Select Theme for Deep Dive:", unique_themes, key="dd_theme")
            with col_dd2:
                dd_sentiment = st.selectbox("Select Sentiment:", ["negative", "positive", "neutral"], key="dd_sentiment")
                
            dd_data = df_exploded[(df_exploded['Theme'] == dd_theme) & (df_exploded['predicted_sentiment'] == dd_sentiment)]
            
            if dd_data.empty:
                st.info(f"No {dd_sentiment} reviews found for '{dd_theme}'.")
            else:
                dd_col1, dd_col2 = st.columns([1, 1])
                with dd_col1:
                    st.markdown(f"**Top Phrases driving {dd_sentiment} sentiment in {dd_theme}**")
                    try:
                        # Group by Theme to create one large document per theme for the selected sentiment
                        sent_data = df_exploded[df_exploded['predicted_sentiment'] == dd_sentiment]
                        theme_docs = sent_data.groupby('Theme')['clean_text'].apply(lambda texts: ' '.join(texts.dropna())).to_dict()
                        
                        if dd_theme in theme_docs and len(theme_docs[dd_theme].strip()) > 0:
                            extra_stop = {"review", "user", "star", "stars", "https", "http", "amp", "just", "like", "im"}
                            stop_words = list(set(sk_text.ENGLISH_STOP_WORDS) | extra_stop)
                            
                            corpus_themes = list(theme_docs.keys())
                            corpus_texts = list(theme_docs.values())
                            
                            # Use TfidfVectorizer to penalize phrases that appear across MULTIPLE themes
                            # This strictly isolates themes from recurring global subthemes like 'staff friendly'
                            tv = TfidfVectorizer(ngram_range=(2, 3), stop_words=stop_words)
                            tfidf_matrix = tv.fit_transform(corpus_texts)
                            
                            theme_idx = corpus_themes.index(dd_theme)
                            feature_names = tv.get_feature_names_out()
                            
                            # Extract the top 10 most distinctive phrases for this specific theme
                            theme_scores = tfidf_matrix[theme_idx].toarray()[0]
                            top_indices = theme_scores.argsort()[-10:][::-1]
                            top_phrases = [feature_names[i] for i in top_indices if theme_scores[i] > 0]
                            
                            # Count their actual mentions within the selected theme's reviews (for display purposes)
                            phrase_counts = []
                            for p in top_phrases:
                                count = theme_docs[dd_theme].count(p)
                                phrase_counts.append(count)
                            
                            phrase_df = pd.DataFrame({"Phrase": top_phrases, "Count": phrase_counts})
                            phrase_df = phrase_df.sort_values(by="Count", ascending=True)
                            
                            color_map = {"positive": "green", "neutral": "gray", "negative": "red"}
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
                        # Clicking this button forces Streamlit to rerun the script, 
                        # which will automatically draw a new `.sample()` of 5 reviews
                        st.button("🔄 Refresh", help="Load different random reviews", use_container_width=True)
                    
                    # Show up to 5 random reviews
                    sample_reviews = dd_data.sample(min(5, len(dd_data)))
                    text_col = "text" if "text" in dd_data.columns else "raw_text" if "raw_text" in dd_data.columns else "clean_text"
                    for review_text in sample_reviews[text_col].tolist():
                        st.info(f'"{review_text}"')

    else:
        st.warning("No themes column found to display statistics.")

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
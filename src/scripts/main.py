"""
main.py
Customer Feedback Intelligence Platform — Entry Point

Launches the Streamlit dashboard or runs in headless mode for batch processing.
Orchestrates the ML pipeline, LLM theme extraction, and dashboard rendering.

Author: Christian East; February 22 2026
Collaborators: Birajman Tamang, Kelsang Yonjan
"""

import argparse
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st

from pipeline_ml import load_or_train_model, preprocess_reviews, predict_reviews
from pipeline_llm import extract_themes
from pipeline_ui import render_dashboard

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "src" / "sample_data"
OUTPUT_DIR   = PROJECT_ROOT / "output"

# The 8 approved business themes used by the LLM
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

st.set_page_config(
    page_title="Customer Feedback Intelligence Platform",
    layout="wide"
)


def _csv_hash(df):
    """
    Create a unique fingerprint for a dataframe.
    Used to detect if the same CSV is uploaded twice so we can skip re-running the LLM.
    """
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


def process_csv(input_path: Path):
    """
    Run the full ML pipeline on a CSV file — preprocess, predict sentiment.
    Used both by the dashboard and by headless mode.
    Does NOT run theme extraction (that is a separate step).
    """
    try:
        model, vectorizer, accuracy = load_or_train_model()
        if accuracy is not None:
            print(f"Model trained. Accuracy: {accuracy:.4f}")
        else:
            print("Loaded existing model.")

        df = pd.read_csv(input_path)
        df = preprocess_reviews(df)
        df = predict_reviews(df, model, vectorizer)
        print(f"Sentiment prediction complete for {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Pipeline error: {e}")
        return None


def run_streamlit_app():
    """
    Launch the full Streamlit dashboard.

    Flow:
    *** Step 1 — Load or train the sentiment model (cached in memory)
    *** Step 2 — User uploads a CSV in the sidebar
    *** Step 3 — Sentiment prediction runs automatically on upload
    *** Step 4 — User can optionally run LLM theme extraction (separate button)
    *** Step 5 — Dashboard renders with all available data
    """
    st.markdown("# Customer Feedback Intelligence Platform")
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

    st.sidebar.header("Upload & Run")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV for analysis", type="csv",
        help="CSV must have a 'text', 'clean_text', or 'raw_text' column."
    )
    st.sidebar.markdown("---")

    run_button = st.sidebar.button("Run Analysis", use_container_width=True)

    if st.sidebar.button("Reset / Clear Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # store the df in session state so it survives Streamlit reruns
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None

    if uploaded_file is not None and run_button:
        # only reprocess if it is a new file or first upload
        if (st.session_state.processed_df is None or
                st.session_state.get("last_uploaded") != uploaded_file.name):
            with st.spinner("Processing file — predicting sentiment..."):
                temp_path = OUTPUT_DIR / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                df = process_csv(temp_path)

            if df is not None:
                st.session_state.processed_df    = df
                st.session_state.last_uploaded   = uploaded_file.name
            else:
                st.error("Failed to process the CSV file.")

    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df

        st.sidebar.markdown("---")
        st.sidebar.subheader("Theme Extraction (LLM)")

        if st.sidebar.button("Extract Themes via LLM", use_container_width=True,
                             help="This runs locally and may take several minutes for large datasets."):
            # use cached themes if same data was processed before
            cache_key = f"themes_{_csv_hash(df)}"
            if cache_key in st.session_state:
                df["themes"] = st.session_state[cache_key]
                st.sidebar.success("Themes loaded from cache.")
            elif "themes" in df.columns and df["themes"].notna().any():
                st.sidebar.success("Themes already extracted.")
            else:
                df = extract_themes(df, THEMES)
                st.session_state[cache_key]       = df["themes"].copy()
                st.session_state.processed_df     = df
                st.rerun()

        render_dashboard(df, THEMES)

    elif not uploaded_file:
        st.info("Upload a CSV file in the sidebar and click Run Analysis to get started.")
        st.markdown("---")
        st.markdown("**Your CSV file needs these columns:**")
        st.markdown("- **text** or **raw_text** — the review content")
        st.markdown("- **stars** — star rating from 1 to 5")
        st.markdown("- **date** — date of the review (for time-based charts)")
        st.markdown("- **review_id** — optional, will be generated if missing")


def main():
    parser = argparse.ArgumentParser(
        description="Customer Feedback Intelligence Platform"
    )
    parser.add_argument("--input",  type=str, help="Path to input CSV for headless processing")
    parser.add_argument("--no-ui",  action="store_true", help="Run without Streamlit UI")
    args, _ = parser.parse_known_args()

    if args.input and args.no_ui:
        print("Running in headless mode...")
        process_csv(Path(args.input))
    else:
        run_streamlit_app()


if __name__ == "__main__":
    main()

import argparse
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
    st.title("Customer Intelligence Feedback Platform")
    
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

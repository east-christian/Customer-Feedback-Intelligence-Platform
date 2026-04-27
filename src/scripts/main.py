import argparse
from pathlib import Path
import pandas as pd
import streamlit as st

# importing pipeline components
from pipeline_ml import load_or_train_model, preprocess_reviews, predict_reviews
from pipeline_llm import extract_themes
from pipeline_ui import render_dashboard

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "sample_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Define the THEMES list for LLM processing here
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
    
    # stores the df in a session state
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None

    if uploaded_file is not None:
        if st.session_state.processed_df is None or st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            # if there is a new or unprocessed file
            with st.spinner("Processing file, predicting sentiment"):
                # saves a temporary file in case of interruption
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
                    st.sidebar.success("Themes successfully extracted")
                else:
                    df = extract_themes(df, THEMES)
                    st.session_state.processed_df = df
                    st.rerun()

            # calls the dashboard
            render_dashboard(df, THEMES)

    else:
        st.info("Please upload a CSV file containing at least a 'text' or 'raw_text' column.")
        
        st.markdown("Ready for Analysis")
        st.markdown("1. Upload your dataset in the sidebar.")
        st.markdown("2. The system immediately makes predictions on basic customer sentiments after upload.")
        st.markdown("3. Run the LLM theme extraction tool to provide in-depth visualizations. (This will take a while to run.)")

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Pipeline Orchestrator")
    parser.add_argument("--input", type=str, help="Path to input CSV for processing")
    parser.add_argument("--no-ui", action="store_true", help="Run without Streamlit")
    
    args, unknown = parser.parse_known_args()

    if args.input and args.no_ui:
        print("Running in headless mode...")
        process_csv(Path(args.input))
    else:
        run_streamlit_app()

if __name__ == "__main__":
    main()

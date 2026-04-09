import argparse
import subprocess
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction import text as sk_text
import numpy as np
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent  # Adjusted to project root
DATA_DIR = PROJECT_ROOT / 'src' / 'sample_data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODEL_FILE = OUTPUT_DIR / "sentiment_model.pkl"
VECTORIZER_FILE = OUTPUT_DIR / "tfidf_vectorizer.pkl"

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

def sentiments_from_stars(stars, classification_type='three_class'):
    if classification_type == 'binary':
        if stars >= 4:
            return 'positive'
        elif stars <= 2:
            return 'negative'
        else:
            return None
    else:
        if stars >= 4:
            return 'positive'
        elif stars == 3:
            return 'neutral/mixed'
        else:
            return 'negative'

def add_sentiment_values_to_file():
    file = DATA_DIR / 'training_testing_data.csv'
    df = pd.read_csv(file)
    df['sentiment'] = df['stars'].apply(lambda x: sentiments_from_stars(x, 'three_class'))
    df.to_csv(file, index=False)

def train_model():
    file = DATA_DIR / 'training_testing_data.csv'
    if not file.exists():
        st.error("Training data not found. Ensure training_testing_data.csv exists in src/sample_data.")
        return None, None
    
    df = pd.read_csv(file)
    if 'sentiment' not in df.columns:
        add_sentiment_values_to_file()
        df = pd.read_csv(file)
    
    df_binary = df[df['sentiment'].notna()].copy()
    content = df_binary['clean_text']
    sent = df_binary['sentiment']
    
    content_train, content_test, sent_train, sent_test = train_test_split(
        content, sent, test_size=0.2, random_state=2016, stratify=sent
    )
    
    extra_stop = {'review','user','star','stars','https','http','amp'}
    stop_words = set(sk_text.ENGLISH_STOP_WORDS) | extra_stop
    
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8, stop_words=list(stop_words)
    )
    content_train_tfidf = vectorizer.fit_transform(content_train)
    content_test_tfidf = vectorizer.transform(content_test)
    
    model = LogisticRegression(max_iter=1000, random_state=2016, C=0.8)
    model.fit(content_train_tfidf, sent_train)
    
    # Quick evaluation (optional, can be removed for speed)
    sent_predict = model.predict(content_test_tfidf)
    accuracy = accuracy_score(sent_test, sent_predict)
    st.write(f"Model trained with accuracy: {accuracy:.4f}")
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    return model, vectorizer

def load_model():
    if MODEL_FILE.exists() and VECTORIZER_FILE.exists():
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    else:
        with st.spinner("Training model... This may take a few minutes."):
            return train_model()

def preprocess_and_predict(df, model, vectorizer):
    # Assume 'text' column exists; add cleaning if needed
    if 'text' not in df.columns:
        st.error("CSV must have a 'text' column.")
        return None
    df['clean_text'] = df['text'].fillna("").str.lower()  # Basic cleaning
    tfidf = vectorizer.transform(df['clean_text'])
    preds = model.predict(tfidf)
    probs = model.predict_proba(tfidf)
    df['predicted_sentiment'] = preds
    df['confidence'] = probs.max(axis=1)
    for i, cls in enumerate(model.classes_):
        df[f'prob_{cls}'] = probs[:, i]
    return df

def run_theme_extraction(input_csv):
    # Modify theme_extraction.py to accept input file via arg or env var if needed
    # For now, assume it processes a fixed file; copy user data to that path
    temp_input = OUTPUT_DIR / "temp_reviews.csv"
    input_csv.to_csv(temp_input, index=False)
    # Run theme script (update theme_extraction.py to read from temp_input)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "scripts" / "theme_extraction.py")], check=True)
    theme_output = pd.read_csv(OUTPUT_DIR / "theme_extracted_output.csv")
    return theme_output

def main():
    st.title("Sentiment Analysis & Theme Extraction Dashboard")
    
    model, vectorizer = load_model()
    if not model:
        return
    
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV with reviews (must have 'text' column)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Data uploaded!")
        
        if st.sidebar.button("Run Analysis"):
            with st.spinner("Running sentiment prediction..."):
                df_pred = preprocess_and_predict(df, model, vectorizer)
            
            with st.spinner("Running theme extraction..."):
                df_themes = run_theme_extraction(df_pred)
            
            # Merge (simple row-wise for now; improve with keys if available)
            final_df = pd.concat([df_themes.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)
            st.session_state['final_df'] = final_df
            st.success("Analysis complete!")
    
    if 'final_df' in st.session_state:
        df = st.session_state['final_df']
        
        # Visualization selector
        viz = st.selectbox("Choose Visualization", [
            "Sentiment Distribution",
            "Theme Counts",
            "Confidence Histogram",
            "Sentiment Over Time (if date available)",
            "Top Themes by Sentiment"
        ])
        
        if viz == "Sentiment Distribution":
            fig = px.bar(df['predicted_sentiment'].value_counts(), title="Sentiment Distribution")
            st.plotly_chart(fig)
        
        elif viz == "Theme Counts":
            if 'themes' in df.columns:
                themes = df['themes'].str.split(',').explode().str.strip().value_counts()
                fig = px.bar(themes, title="Theme Counts")
                st.plotly_chart(fig)
            else:
                st.write("Themes not available.")
        
        elif viz == "Confidence Histogram":
            fig, ax = plt.subplots()
            sns.histplot(df['confidence'], bins=20, ax=ax)
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)
        
        elif viz == "Sentiment Over Time (if date available)":
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                time_sent = df.groupby(df['date'].dt.to_period('M'))['predicted_sentiment'].value_counts().unstack()
                fig = px.line(time_sent, title="Sentiment Over Time")
                st.plotly_chart(fig)
            else:
                st.write("Date column not found.")
        
        elif viz == "Top Themes by Sentiment":
            if 'themes' in df.columns and 'predicted_sentiment' in df.columns:
                cross = df.groupby('predicted_sentiment')['themes'].apply(lambda x: x.str.split(',').explode().str.strip().value_counts().head(5))
                st.write(cross)
            else:
                st.write("Required columns not available.")
        
        st.download_button("Download Results", df.to_csv(index=False), "results.csv")

if __name__ == "__main__":
    main()
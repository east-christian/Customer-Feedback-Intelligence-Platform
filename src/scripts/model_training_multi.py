"""
Sentiment Analysis Model Training Script

This script trains a machine learning model to predict sentiment (positive/negative) 
from customer reviews based on their text content and star ratings.

Author: Christian East; February 22 2026
Collaborators: Birajman Tamang, Kelsang Yonjan
Model Type: Logistic Regression with TF-IDF Vectorization
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, log_loss
)
from sklearn.feature_extraction import text as sk_text
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from pathlib import Path

# Get the absolute path to the project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / 'src' / 'sample_data'
OUTPUT_DIR = PROJECT_ROOT / 'output'


# Sentiment classification functions

def sentiments_from_stars(stars, classification_type = 'three_class'):
    """
    Convert star ratings into sentiment labels.
    
    This function maps numerical star ratings (1-5) to sentiment categories.
    
    Parameters:
        stars (int): Star rating from 1 to 5
        classification_type (str): Either 'binary' or 'three_class'
            - 'binary': Only positive (4-5 stars) and negative (1-2 stars) (excludes neutral reviews from training)
            - 'three_class': Positive, neutral (3 stars), and negative
    
    Returns:
        str or None: Sentiment label ('positive', 'negative', 'neutral', or None)
    """
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
    """
    Data Preprocessing: Add sentiment labels to raw review data.
    
    This function reads the cleaned reviews dataset, applies sentiment 
    classification based on star ratings, and saves the processed data 
    with sentiment labels for model training.
    
    Input: training_testing_data.csv
    Output: training_testing_data.csv (overwrites original with sentiment labels)
    """
    # Load the cleaned source dataset
    file = DATA_DIR / 'training_testing_data.csv'
    df = pd.read_csv(file)

    # Extract star ratings for sentiment classification
    star_ratings = df['stars']
    print(star_ratings)

    # Apply binary sentiment classification (positive/negative only)
    df['sentiment'] = df['stars'].apply(lambda x: sentiments_from_stars(x, 'three_class'))

    print(df['sentiment'])
    
    # Save the labeled dataset back to the original file
    df.to_csv(DATA_DIR / 'training_testing_data.csv', index=False)

# Model Training and Evaluation

def main():
    """
    This function executes the model training process:
    1. Load preprocessed data with sentiment labels
    2. Split data into training and testing sets
    3. Vectorize text using TF-IDF (Term Frequency-Inverse Document Frequency)
    4. Train a Logistic Regression classifier
    5. Evaluate model performance and generate detailed metrics
    6. Save results to training.log for review
    """
    # Step 1: Data Loading
    # Load the sentiment-labeled dataset created by preprocessing
    file = DATA_DIR / 'training_testing_data.csv'
    df = pd.read_csv(file)
    
    # Filter out records with missing sentiment labels (e.g., 3-star reviews in binary classification)
    df_binary = df[df['sentiment'].notna()].copy()

    # Separate features (review text) and labels (sentiment)
    content = df_binary['clean_text']  # Review text content
    sent = df_binary['sentiment']      # Sentiment labels (positive/negative)

    # Step 2: Train-Test Split 
    # Split dataset: 80% training, 20% testing
    # Stratification ensures proportional distribution of sentiment classes in both sets
    content_train, content_test, sent_train, sent_test = train_test_split(
        content,
        sent,
        test_size=0.2,          # 20% reserved for model testing
        random_state=2016,      # Fixed seed for reproducible results
        stratify=sent           # Maintain sentiment distribution across splits
    )
    
    extra_stop = {'review','user','star','stars','https','http','amp'}
    stop_words = set(sk_text.ENGLISH_STOP_WORDS) | extra_stop

    # Step 3: Text Vectorization (TF-IDF)
    # Convert text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
    # This captures the importance of words relative to their frequency across all reviews
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Use top 5,000 most significant terms
        ngram_range=(1, 2),     # Include both individual words and two-word phrases
        min_df=2,               # Exclude rare terms (appearing in fewer than 2 reviews)
        max_df=0.8,           # Exclude common terms (appearing in more than 80% of reviews)
        stop_words=list(stop_words)
    )

    # Transform training data and learn vocabulary
    content_train_tfidf = vectorizer.fit_transform(content_train)
    
    # Transform test data using the same vocabulary (no additional learning)
    content_test_tfidf = vectorizer.transform(content_test)

    # Step 4: Model Training
    # Initialize Logistic Regression classifier
    # This algorithm is effective for binary text classification tasks
    model = LogisticRegression(
        max_iter=1000,          # Maximum iterations for model convergence
        random_state=2016,      # Fixed seed for reproducibility
        C=0.8                  # Regularization strength (lower = stronger regularization)
    )
    
    # Train the model on vectorized training data
    model.fit(content_train_tfidf, sent_train)
    
    # Extract feature importance for each sentiment class
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_  # Shape: (n_classes, n_features)
    
    # Create vocabulary dataframe with sentiment associations
    vocab_data = []
    for idx, feature in enumerate(feature_names):
        # Get TF-IDF average importance
        tfidf_value = content_train_tfidf.mean(axis=0).A1[idx]
        
        # Get coefficients for this feature across all sentiments
        feature_coefs = coefficients[:, idx]
        
        # Find which sentiment this feature most strongly predicts
        max_sentiment_idx = np.argmax(np.abs(feature_coefs))
        primary_sentiment = model.classes_[max_sentiment_idx]
        sentiment_coefficient = feature_coefs[max_sentiment_idx]
        
        # Store all sentiment coefficients for reference
        coef_dict = {f'{sent}_coef': feature_coefs[i] 
                    for i, sent in enumerate(model.classes_)}
        
        vocab_data.append({
            'Word/Phrase': feature,
            'TF-IDF_Average': tfidf_value,
            'Primary_Sentiment': primary_sentiment,
            'Sentiment_Coefficient': sentiment_coefficient,
            **coef_dict
        })
    
    vocab_dataframe = pd.DataFrame(vocab_data)
    vocab_dataframe = vocab_dataframe.sort_values(by=['TF-IDF_Average'], ascending=False)
    vocab_dataframe.to_csv(OUTPUT_DIR / 'vocab_data_multi.csv', index=False)
    print(f"\nVocabulary data with sentiment associations saved to vocab_data_multi.csv")

    # Step 5: Model Evaluation
    # Generate predictions on the test set
    sent_predict = model.predict(content_test_tfidf)
    sent_proba = model.predict_proba(content_test_tfidf)
    confidence = sent_proba.max(axis=1)

    accuracy = accuracy_score(sent_test, sent_predict)
    balanced_acc = balanced_accuracy_score(sent_test, sent_predict)
    macro_f1 = f1_score(sent_test, sent_predict, average='macro')
    weighted_f1 = f1_score(sent_test, sent_predict, average='weighted')
    multiclass_log_loss = log_loss(sent_test, sent_proba, labels=model.classes_)

    cm = confusion_matrix(sent_test, sent_predict, labels=model.classes_)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
    plt.title('Confusion Matrix - Logistic Regression; Multiclass Classification')
    plt.ylabel('Actual Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_multi.png', dpi=300)
    plt.close()

    # Create feature importance visualization by sentiment
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    for idx, sentiment in enumerate(model.classes_):
        # Get top 15 features for this sentiment based on coefficient strength
        sentiment_col = f'{sentiment}_coef'
        top_features = vocab_dataframe.nlargest(15, sentiment_col)

        # Determine base color per sentiment and create per-bar colors by ranking
        base_colors = {
            'positive': (0.0, 0.6, 0.0),
            'neutral/mixed': (0.5, 0.5, 0.5),
            'negative': (0.8, 0.0, 0.0)
        }

        n = len(top_features)
        if n == 0:
            # Nothing to plot for this sentiment
            axes[idx].set_visible(False)
            continue

        # Intensities from strongest (1.0) to lightest (0.4) across ranked features
        intensities = np.linspace(1.0, 0.4, n)

        # Choose base color (fallback to gray if sentiment label unexpected)
        base = base_colors.get(str(sentiment).lower(), (0.5, 0.5, 0.5))
        base = np.array(base)

        # Blend each bar color with white to produce lighter shades for lower-ranked features
        colors = [tuple(1 - inten * (1 - base)) for inten in intensities]

        # Create horizontal bar chart with per-bar colors
        axes[idx].barh(range(n), top_features[sentiment_col], color=colors)
        axes[idx].set_yticks(range(n))
        axes[idx].set_yticklabels(top_features['Word/Phrase'], fontsize=9)
        axes[idx].set_xlabel('Coefficient Value', fontsize=10)
        axes[idx].set_title(f'Top Features - {sentiment.capitalize()}', fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'features_by_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFeature importance by sentiment saved to features_by_sentiment.png")
    
    # Save predictions to CSV with original text and actual sentiment
    predictions_df = pd.DataFrame({
        'text': content_test.values,
        'actual_sentiment': sent_test.values,
        'predicted_sentiment': sent_predict,
        'confidence': confidence,
        'correct_prediction': (sent_test.values == sent_predict)
    })

    for i, cls in enumerate(model.classes_):
        predictions_df[f'prob_{cls}'] = sent_proba[:, i]

    predictions_df.to_csv(OUTPUT_DIR / 'predicted_data_multi.csv', index=False)
    print(f"\nPredictions saved to output/predicted_data_multi.csv")

    # Step 6: Results Logging
    # Export comprehensive training report to log file
    # Save original stdout to restore it later
    original_stdout = sys.stdout
    
    with open(OUTPUT_DIR / 'training_multi.log', 'w') as log_file:
        sys.stdout = log_file
        
        # Dataset Summary
        print("\nMODEL PERFORMANCE METRICS\n")
        print('\nClassification Report:')
        print(classification_report(sent_test, sent_predict))
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Balanced Accuracy: {balanced_acc:.4f}')
        print(f'Macro F1: {macro_f1:.4f}')
        print(f'Weighted F1: {weighted_f1:.4f}')
        print(f'Log Loss: {multiclass_log_loss:.4f}')

        print(f'\nConfusion Matrix (multiclass):')
        print(cm)
        print("\nRows = actual class, Columns = predicted class.")
    
    # Restore original stdout
    sys.stdout = original_stdout

# Execution

if __name__ == "__main__":
    """
    Main execution block.
    
    Process:
    1. Check if preprocessed sentiment-labeled data exists
    2. If not, generate it from raw cleaned reviews
    3. Execute model training pipeline
    4. Results are saved to output/training.log
    """
    # Verify or create sentiment-labeled dataset
    if not os.path.exists(DATA_DIR / 'training_testing_data.csv'):
        print("Error: training_testing_data.csv not found!")
        sys.exit(1)
    
    # Check if sentiment column exists, if not, add it
    test_df = pd.read_csv(DATA_DIR / 'training_testing_data.csv')
    if 'sentiment' not in test_df.columns:
        print("Preprocessing: Generating sentiment-labeled dataset..")
        add_sentiment_values_to_file()
        print("Dataset creation complete..\n")
    
    # Execute model training and evaluation
    print("Training start..")
    main()
    print("\n Training complete; Results saved to output/training.log")
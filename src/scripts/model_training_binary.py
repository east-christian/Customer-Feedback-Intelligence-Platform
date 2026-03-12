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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
            return 'neutral'
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
    df['sentiment'] = df['stars'].apply(lambda x: sentiments_from_stars(x, 'binary'))

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
    df_binary = df[df['sentiment'] != 'neutral'].copy()

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
    
    # Step 3: Text Vectorization (TF-IDF)
    # Convert text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
    # This captures the importance of words relative to their frequency across all reviews
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Use top 5,000 most significant terms
        ngram_range=(1, 2),     # Include both individual words and two-word phrases
        min_df=2,               # Exclude rare terms (appearing in fewer than 2 reviews)
        max_df=0.8              # Exclude common terms (appearing in more than 80% of reviews)
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
        C=0.5                   # Regularization strength (lower = stronger regularization)
    )
    
    # Train the model on vectorized training data
    model.fit(content_train_tfidf, sent_train)

    # Step 5: Model Evaluation
    # Generate predictions on the test set
    sent_predict = model.predict(content_test_tfidf)

    # Calculate performance metrics
    accuracy = accuracy_score(sent_predict, sent_test)
    cm = confusion_matrix(sent_test, sent_predict)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
    plt.title('Confusion Matrix - Logistic Regression; Binary Classification')
    plt.ylabel('Actual Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_binary.png', dpi=300)
    plt.close()
    
    
    # Save predictions to CSV with original text and actual sentiment
    predictions_df = pd.DataFrame({
        'text': content_test.values,
        'actual_sentiment': sent_test.values,
        'predicted_sentiment': sent_predict
    })
    predictions_df.to_csv(OUTPUT_DIR / 'predicted_data_binary.csv', index=False)
    print(f"\nPredictions saved to output/predicted_data_binary.csv")

    # Step 6: Results Logging
    # Export comprehensive training report to log file
    # Save original stdout to restore it later
    original_stdout = sys.stdout
    
    with open(OUTPUT_DIR / 'training_binary.log', 'w') as log_file:
        sys.stdout = log_file
        
        # Dataset Summary
        print("\n")
        print("SENTIMENT ANALYSIS MODEL TRAINING REPORT")
        print("\n")
        print(f"\nDataset Split:")
        print(f"  Training samples: {len(content_train)}")
        print(f"  Test samples: {len(content_test)}")

        print(f"\nTraining Sentiment Distribution:")
        print(sent_train.value_counts())

        # Feature Engineering Details
        print(f"\nText Vectorization:")
        print(f"  Feature matrix shape: {content_train_tfidf.shape}")
        print(f"  Vocabulary size: {len(vectorizer.vocabulary_)} unique terms")
        
        # Model Information
        print(f"\nModel Configuration:")
        print(f"  Algorithm: {type(model).__name__}")
        print(f"  Classes: {list(model.classes_)}")
        print(f"  Total predictions made: {len(sent_predict)}")
        
        # Performance Metrics
        print(f"\n")
        print("MODEL PERFORMANCE METRICS")
        print(f"\n")
        print('\nClassification Report:')
        print(classification_report(sent_test, sent_predict))

        print(f'Overall Accuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)')

        print(f'\nConfusion Matrix:')
        print(cm)
        print("\nInterpretation: [True Negatives, False Positives]")
        print("                [False Negatives, True Positives]")

        # Distribution Analysis
        print(f"\n")
        print("Sentiment Distribution Comparison:")
        print(f"\n")
        print("\nPredicted sentiment distribution:")
        print(pd.Series(sent_predict).value_counts())
        print("\nActual sentiment distribution:")
        print(sent_test.value_counts())
    
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
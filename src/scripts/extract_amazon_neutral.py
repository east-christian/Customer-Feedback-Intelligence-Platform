import gzip
import json
import re
import pandas as pd
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "sample_data"
TRAIN_FILE = DATA_DIR / "training_testing_data.csv"

# Locate the Amazon file (check root and sample_data)
AMAZON_FILE = DATA_DIR / "All_Beauty.jsonl.gz"
if not AMAZON_FILE.exists():
    AMAZON_FILE = PROJECT_ROOT / "All_Beauty.jsonl.gz"

if not AMAZON_FILE.exists():
    raise FileNotFoundError(f"Could not find All_Beauty.jsonl.gz in {PROJECT_ROOT} or {DATA_DIR}")

def main():
    print(f"Reading existing training data from {TRAIN_FILE.name}...")
    df_current = pd.read_csv(TRAIN_FILE)
    
    # Drop rows from previous extraction that missed 'raw_text' or 'review_id'
    df_current = df_current.dropna(subset=['raw_text']).copy()
    
    current_neutrals = len(df_current[df_current['stars'] == 3])
    
    # Based on our estimates, we need about 5000 more 3-star reviews to match the positive class (~7000).
    target_extra = 5000
    print(f"Current 3-star reviews: {current_neutrals}. Target additional: {target_extra}.")
    print(f"Extracting from {AMAZON_FILE.name}...")

    new_rows = []
    try:
        with gzip.open(AMAZON_FILE, 'rt', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                rating = data.get("rating")
                
                if rating == 3.0:
                    text = data.get("text", "")
                    if text and len(text.strip()) > 10:  # Exclude blank or overly brief reviews
                        # Clean the raw text by lowercasing, stripping newlines and extra spaces
                        raw_text = text.strip()
                        clean = raw_text.lower()
                        clean = re.sub(r'[\r\n]+', ' ', clean)
                        clean = re.sub(r'[^\w\s\']', ' ', clean)
                        clean = re.sub(r'\s+', ' ', clean).strip()
                        
                        new_rows.append({
                            "stars": 3.0,
                            "raw_text": raw_text,
                            "clean_text": clean,
                            "raw_text_length": len(raw_text)
                        })
                        
                        if len(new_rows) >= target_extra:
                            break
    except Exception as e:
        print(f"Error reading Amazon file: {e}")
        return

    # Form dataframe and align with current columns
    df_new = pd.DataFrame(new_rows)
    print(f"Successfully extracted {len(df_new)} reviews.")
    
    # The existing dataframe might have 'text', 'sentiment', etc. Let's make sure we safely merge
    for col in df_current.columns:
        if col not in df_new.columns:
            df_new[col] = float('nan') # Fill missing fields like 'date', 'sentiment' with NaN
            
    df_new = df_new[df_current.columns] # Reorder to match perfectly
    
    # Append the new dataframe
    df_combined = pd.concat([df_current, df_new], ignore_index=True)
    
    # We drop the 'sentiment' column so that model_training_multi.py regenerates it fresh
    if 'sentiment' in df_combined.columns:
        df_combined = df_combined.drop(columns=['sentiment'])
        
    df_combined.to_csv(TRAIN_FILE, index=False)
    print(f"\nSaved! Total reviews in dataset is now {len(df_combined)}.")
    print("Dropped the 'sentiment' column to trigger a fresh label generation on next training run.")

if __name__ == "__main__":
    main()

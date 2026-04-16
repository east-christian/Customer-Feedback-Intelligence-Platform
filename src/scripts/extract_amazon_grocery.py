import gzip
import json
import re
import datetime
import pandas as pd
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "sample_data"

# Locate the Amazon file
AMAZON_FILE = DATA_DIR / "Grocery_and_Gourmet_Food.jsonl.gz"
if not AMAZON_FILE.exists():
    AMAZON_FILE = PROJECT_ROOT / "Grocery_and_Gourmet_Food.jsonl.gz"

OUTPUT_FILE = DATA_DIR / "amazon_grocery_sample_15000.csv"

def main():
    if not AMAZON_FILE.exists():
        print(f"Could not find {AMAZON_FILE.name} in {PROJECT_ROOT} or {DATA_DIR}")
        return

    target_count = 15000
    print(f"Extracting {target_count} reviews from {AMAZON_FILE.name}...")

    new_rows = []
    
    try:
        with gzip.open(AMAZON_FILE, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                
                text = data.get("text", "")
                rating = data.get("rating")
                
                if text and len(text.strip()) > 10 and rating is not None:
                    raw_text = text.strip()
                    
                    # Clean text
                    clean = raw_text.lower()
                    clean = re.sub(r'[\r\n]+', ' ', clean)
                    clean = re.sub(r'[^\w\s\']', ' ', clean)
                    clean = re.sub(r'\s+', ' ', clean).strip()
                    
                    # Process timestamp
                    unix_time = data.get("timestamp")
                    if unix_time:
                        # Convert ms to seconds
                        dt = datetime.datetime.fromtimestamp(unix_time / 1000.0)
                        date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                        year_month = dt.strftime('%Y-%m')
                    else:
                        date_str = None
                        year_month = None

                    new_rows.append({
                        "review_id": f"grocery_{len(new_rows)}",
                        "stars": float(rating),
                        "date": date_str,
                        "year_month": year_month,
                        "raw_text": raw_text,
                        "clean_text": clean,
                        "raw_text_length": len(raw_text)
                    })
                    
                    if len(new_rows) >= target_count:
                        break
    except Exception as e:
        print(f"Error reading Amazon file: {e}")
        return

    df_new = pd.DataFrame(new_rows)
    print(f"Successfully extracted {len(df_new)} reviews.")
    
    # Save the output CSV
    df_new.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved new sample file to {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()

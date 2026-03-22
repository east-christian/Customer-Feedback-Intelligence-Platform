# Extracts themes from customer reviews using local LLM (Gemma 3 4B)
# Scope: Food chains and retail businesses (Starbucks, Walmart, Target etc.)

import pandas as pd
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_client import call_llm


# Configuration
INPUT_FILE    = "data/sampled_reviews_3000.csv"
OUTPUT_FILE   = "output/theme_extracted_output.csv"
FAILED_FILE   = "output/failed_extraction_reviews.csv"
PROGRESS_FILE = "output/progress.csv"

REVIEW_COLUMN = "clean_text"  # column in CSV to read review text from
BATCH_SIZE    = 10            # reviews per LLM call (sweet spot for speed and accuracy)
MAX_WORKERS   = 2             # parallel workers (safe for 15GB RAM, CPU only machine)
MAX_RETRIES   = 3             # retry failed batches before giving up
SAVE_EVERY    = 5             # save progress checkpoint every N batches


# Theme list
"""
Mixed theme list designed to work across multiple business types:
    Food chains   -> Starbucks, McDonalds, Burger King etc.
    Retail stores -> Walmart, Target, Costco etc.
    Any business  -> that collects customer reviews

"""
THEMES = [
    "Product Quality",      # food taste and freshness OR retail item quality
    "Food Quality",         # specific to food and beverage businesses
    "Drink Quality",        # specific to coffee shops and beverage chains
    "Staff Friendliness",   # attitude and politeness of employees
    "Customer Service",     # help, support, and problem resolution
    "Wait Time",            # long queues, slow service, order delays
    "Order Accuracy",       # wrong or missing items in food and retail
    "Store Cleanliness",    # dirty environment, tables, floors, bathrooms
    "Price Value",          # whether cost is worth the product or service
    "Atmosphere",           # store layout, noise, lighting, comfort
    "Digital Experience",   # app, website, online ordering problems
    "Loyalty Rewards",      # points, membership benefits, app deals
    "Location Parking",     # store accessibility and parking availability
    "Health Safety",        # hygiene concerns and food or product safety
    "Product Availability", # out of stock items or limited menu options
    "Delivery Experience",  # shipping speed and condition for retail and food
    "Return Policy",        # refunds, exchanges, complaint handling
    "Drive Thru",           # drive thru speed and accuracy for food chains
]


def build_prompt(batch):
    """
    Build the LLM classification prompt for a batch of reviews.
    Reviews are numbered so model returns exactly the same count back.
    Truncated to 250 characters to save tokens and stay within context window.
    """
    numbered = "\n".join([
        f"{i+1}. {str(r)[:250]}"
        for i, r in enumerate(batch)
    ])

    prompt = f"""You are a professional theme classifier for customer reviews.
You work for a Feedback Intelligence Platform that analyzes reviews
for any type of business including food chains and retail stores.

Available themes: {THEMES}

RULES:
- Only assign themes from the available list above
- Assign between 1 and 3 themes per review
- If a review is unclear assign the single closest matching theme
- Never create or invent themes not in the list
- Every review must have at least 1 theme assigned
- Return ONLY a JSON array of arrays with one inner array per review
- Do not include any explanation or extra text

Example output for 3 reviews:
[
  ["Staff Friendliness", "Drink Quality"],
  ["Wait Time"],
  ["Price Value", "Store Cleanliness"]
]

Reviews to classify:
{numbered}

Return ONLY the JSON array of arrays. No explanation."""

    return prompt


def extract_themes_with_retry(batch_info):
    """
    Send one batch to the local LLM and return themes.
    Retries up to MAX_RETRIES times if something goes wrong.
    Always returns a status so failed reviews are never lost.

    Retry flow:
        Attempt 1 -> fails -> wait 2 sec -> Attempt 2
        Attempt 2 -> fails -> wait 2 sec -> Attempt 3
        Attempt 3 -> fails -> mark as FAILED -> save to failed file
    """
    batch_idx, batch = batch_info
    prompt = build_prompt(batch)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_llm(prompt)

            # extract JSON array from response
            # model sometimes adds extra text around the array
            start  = raw.find("[")
            end    = raw.rfind("]") + 1
            clean  = raw[start:end]

            themes = json.loads(clean)

            # validate result count matches batch size exactly
            if len(themes) != len(batch):
                raise ValueError(f"Expected {len(batch)} results got {len(themes)}")

            print(f"  ✓ Batch {batch_idx} done ({len(batch)} reviews)")
            return batch_idx, batch, themes, "success"

        except Exception as e:
            print(f"  ✗ Batch {batch_idx} attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)

    # all retries failed
    print(f"  ❌ Batch {batch_idx} permanently failed after {MAX_RETRIES} attempts")
    return batch_idx, batch, None, "failed"


def main():
    """
    Main pipeline that runs the full theme extraction process:
        1. Load CSV
        2. Resume from checkpoint if previous run was interrupted
        3. Split reviews into batches
        4. Run parallel theme extraction
        5. Merge themes back into original dataframe
        6. Save final output keeping all original columns
    """

    print("=" * 55)
    print("  Theme Extraction Pipeline")
    print("  Scope: Food Chains and Retail Businesses")
    print("  Model: Gemma 3 4B (local, free)")
    print("=" * 55)

    # Step 1: Load CSV
    print(f"\nLoading reviews from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    if REVIEW_COLUMN not in df.columns:
        print(f"\nERROR: Column '{REVIEW_COLUMN}' not found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        print(f"Update REVIEW_COLUMN in config to match your CSV.")
        return

    print(f"✓ Reviews loaded:    {len(df)}")
    print(f"✓ Reading column:    {REVIEW_COLUMN}")
    print(f"✓ Columns in CSV:    {list(df.columns)}")

    reviews = df[REVIEW_COLUMN].fillna("").tolist()

    # Step 2: Resume from checkpoint
    """
    If script was interrupted, resume from where it stopped.
    Progress file stores completed results so we never reprocess
    reviews that already have themes extracted.
    """
    already_done = 0

    if os.path.exists(PROGRESS_FILE):
        progress_df  = pd.read_csv(PROGRESS_FILE)
        already_done = len(progress_df)
        print(f"\n✓ Resuming from review {already_done}")
    else:
        print(f"\n✓ Starting fresh")

    reviews_remaining = reviews[already_done:]

    if len(reviews_remaining) == 0:
        print("\n✅ All reviews already processed!")
        return

    print(f"✓ Reviews remaining: {len(reviews_remaining)}")

    # Step 3: Create batches
    batches = []
    for i in range(0, len(reviews_remaining), BATCH_SIZE):
        batch = reviews_remaining[i:i + BATCH_SIZE]
        batches.append((already_done + i, batch))

    print(f"\n✓ Batch size:        {BATCH_SIZE} reviews per batch")
    print(f"✓ Total batches:     {len(batches)}")
    print(f"✓ Parallel workers:  {MAX_WORKERS}")
    print(f"✓ Retries per batch: {MAX_RETRIES}")
    print(f"\n Starting extraction...\n")

    # Step 4: Run parallel processing
    """
    ThreadPoolExecutor runs MAX_WORKERS batches at the same time.
    as_completed processes each result as soon as a batch finishes
    rather than waiting for all batches to complete first.
    Progress is saved every SAVE_EVERY batches to protect our work.
    """
    successful_results = []
    failed_results     = []
    completed_count    = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(extract_themes_with_retry, b): b
            for b in batches
        }

        for future in as_completed(futures):
            batch_idx, batch, themes, status = future.result()
            completed_count += 1

            if status == "success":
                for i, (review, theme) in enumerate(zip(batch, themes)):
                    successful_results.append({
                        "original_idx": batch_idx + i,
                        "themes": ", ".join(theme) if isinstance(theme, list) else str(theme)
                    })
            else:
                for i, review in enumerate(batch):
                    failed_results.append({
                        "original_idx": batch_idx + i,
                        "themes": "FAILED"
                    })

            # save progress checkpoint
            if completed_count % SAVE_EVERY == 0 or completed_count == len(batches):
                checkpoint = pd.DataFrame(successful_results + failed_results)
                checkpoint = checkpoint.sort_values("original_idx")
                checkpoint.to_csv(PROGRESS_FILE, index=False)
                print(f"\n Progress saved ({completed_count}/{len(batches)} batches)\n")

    # Step 5: Merge themes into original dataframe
    """
    Build a lookup dictionary mapping each row index to its themes.
    This keeps all original CSV columns intact and just adds
    the themes column at the end.
    """
    print("\nMerging themes into original dataframe...")

    themes_lookup = {}
    for r in successful_results + failed_results:
        themes_lookup[r["original_idx"]] = r["themes"]

    df["themes"] = [
        themes_lookup.get(i, "NOT PROCESSED")
        for i in range(len(df))
    ]

    # Step 6: Save outputs
    print("\n" + "=" * 55)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Output saved to:   {OUTPUT_FILE}")
    print(f"   Total rows:       {len(df)}")
    print(f"   Output columns:   {list(df.columns)}")

    if failed_results:
        # save failed reviews separately for later retry
        failed_df = df[df["themes"] == "FAILED"].copy()
        failed_df.to_csv(FAILED_FILE, index=False)
        print(f"\n⚠️  Failed reviews:   {len(failed_df)}")
        print(f"   Saved to:        {FAILED_FILE}")
    else:
        print("\n✅ No failures — all reviews processed successfully!")

    # clean up checkpoint file now that everything is complete
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("✓ Progress file cleaned up")

    print("=" * 55)


if __name__ == "__main__":
    start = time.time()
    main()
    end   = time.time()
    print(f"\n Total time: {(end - start) / 60:.1f} minutes")
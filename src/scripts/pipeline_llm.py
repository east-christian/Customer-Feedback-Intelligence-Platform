import json
import ast
import time
import ollama
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

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
  "3": ["Price and Value"]
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
            
            # extracts the JSON block
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

                # proper dict formatting in case the LLM mistakenly formats them in an array
                parsed_dict = {}
                if isinstance(parsed_data, list):
                    for item in parsed_data:
                        if isinstance(item, dict):
                            parsed_dict.update(item)
                elif isinstance(parsed_data, dict):
                    parsed_dict = parsed_data
                else:
                    raise ValueError("Response is not a valid structured JSON.")

                # dict conversion back to array for assignment
                themes = []
                for idx in range(1, len(batch) + 1):
                    key = str(idx)
                    idx_themes = parsed_dict.get(key, [])
                    if not isinstance(idx_themes, list):
                        idx_themes = [idx_themes]
                    themes.append(idx_themes)
            
            if len(themes) != len(batch):
                raise ValueError(f"Batch mismatch: LLM returned {len(themes)} exact theme arrays, but there are {len(batch)} reviews.")

            # theme validation: checks that each review was assigned a proper number of themes.
            validated_themes = []
            for theme_list in themes:
                valid_for_review = []
                # reformats list if LLM provides a dict
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


# function for extraction of themes using LLM
def extract_themes(df, themes_list, batch_size=10, max_workers=2):
    """
    Extract themes for the reviews in the DataFrame using the LLM.
    """
    reviews = df["clean_text"].fillna("").tolist()
    batches = [(i, reviews[i:i + batch_size]) for i in range(0, len(reviews), batch_size)]

    successful_results = []
    failed_results = []

    # progress bar for LLM extraction
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
                # timeout in case of extraction hanging for too long
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
                print(f"Batch {batch_idx} crashed: {e}")

            if status == "success":
                for i, (review, valid_themes) in enumerate(zip(batch, batch_themes)):
                    # returns validated arrays
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

            # more progress bar logic (saves resources)
            completed_batches += 1
            if completed_batches % max(1, (total_batches // 100)) == 0 or completed_batches == total_batches:
                progress_bar.progress(completed_batches / total_batches)
                status_text.text(f"Processed review batch {completed_batches}/{total_batches}. Please wait, local LLM processing takes a while...")

    themes_lookup = {r["original_idx"]: r["themes"] for r in successful_results + failed_results}
    df["themes"] = [themes_lookup.get(i, "FAILED") for i in range(len(df))]

    # drops reviews that failed extraction
    initial_len = len(df)
    df = df[~df["themes"].str.contains("FAILED", na=False)]
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} reviews because the LLM repeatedly hallucinated or timed out.")

    # when extraction is complete
    progress_bar.progress(1.0)
    status_text.text("Theme extraction complete")

    return df

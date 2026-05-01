"""
pipeline_llm.py
Theme Extraction — Large Language Model Pipeline

Sends customer reviews to a locally-running LLM (Gemma via Ollama)
and assigns each review to one or more of the 8 approved business themes.

Author: Christian East; February 22 2026
Collaborators: Birajman Tamang, Kelsang Yonjan
"""

import json
import ast
import time
import ollama
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError


def call_llm(prompt, model="gemma3:4b"):
    """
    Send a prompt to the local Ollama language model and return its response.
    Uses Gemma 3 4B by default — a small, fast model that runs entirely on your Mac.
    If Ollama is not running this will raise an error.
    """
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise e


def build_prompt(batch, themes):
    """
    Build the instruction message sent to the LLM for a batch of reviews.
    Tells the LLM exactly which themes are allowed and asks it to return valid JSON only.
    """
    numbered = "\n".join([
        f"Review {i+1}:\n{str(r)[:250]}"
        for i, r in enumerate(batch)
    ])
    return f"""You are a professional theme classifier for customer reviews.
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
  "3": ["Price & Value"]
}}

Reviews to classify:
{numbered}

Return ONLY valid JSON."""


def extract_themes_with_retry(batch_info, themes_list, max_retries=5):
    """
    Send one batch of reviews to the LLM and extract their themes.
    *** Retries up to 5 times if the LLM returns invalid JSON or invents themes not in the list
    *** Rejects any theme not in the approved list (hallucination guard)
    *** Returns the batch index, reviews, assigned themes, and success/failure status
    """
    batch_idx, batch = batch_info
    prompt = build_prompt(batch, themes_list)

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(prompt)

            # extract the JSON block from the response
            first_brace  = raw.find("{")
            first_brack  = raw.find("[")
            start = min(i for i in [first_brace, first_brack] if i != -1) \
                    if any(i != -1 for i in [first_brace, first_brack]) else -1
            end   = max(raw.rfind("}"), raw.rfind("]")) + 1

            if start == -1 or end <= start:
                raise ValueError("LLM output did not contain valid JSON")

            clean = raw[start:end]

            try:
                parsed_data = json.loads(clean)
            except json.JSONDecodeError:
                try:
                    parsed_data = ast.literal_eval(clean)
                except Exception:
                    if clean.strip().startswith("{") and clean.strip().endswith("}"):
                        parsed_data = json.loads(f"[{clean}]")
                    else:
                        raise ValueError(f"Could not parse response: {clean}")

            # normalize to dict format
            parsed_dict = {}
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    if isinstance(item, dict):
                        parsed_dict.update(item)
            elif isinstance(parsed_data, dict):
                parsed_dict = parsed_data
            else:
                raise ValueError("Response is not a valid structured JSON.")

            # build theme list from dict
            themes = []
            for idx in range(1, len(batch) + 1):
                t = parsed_dict.get(str(idx), [])
                if not isinstance(t, list):
                    t = [t]
                if not t:
                    t = ["Customer Service"]
                themes.append(t)

            if len(themes) != len(batch):
                raise ValueError(
                    f"Batch mismatch: LLM returned {len(themes)} arrays "
                    f"but there are {len(batch)} reviews."
                )

            # validate — reject any invented themes (hallucination guard)
            validated = []
            for theme_list in themes:
                safe = [
                    str(list(t.values())[0]) if isinstance(t, dict) and t else str(t)
                    for t in theme_list
                ]
                valid = []
                for st_theme in safe:
                    for real in themes_list:
                        if st_theme.strip().lower() == real.lower():
                            if real not in valid:
                                valid.append(real)
                            break
                if not valid:
                    raise ValueError(
                        f"Hallucination detected: '{theme_list}' not in approved list"
                    )
                validated.append(valid)

            return batch_idx, batch, validated, "success"

        except Exception as e:
            print(f"Batch {batch_idx} attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)

    return batch_idx, batch, None, "failed"


def extract_themes(df, themes_list, batch_size=10, max_workers=2):
    """
    Assign business themes to every review using the local LLM.

    *** Splits reviews into batches of 10 so the LLM does not get overwhelmed
    *** Runs 2 batches at the same time (parallel processing) to save time
    *** Shows a progress bar while processing
    *** Any batch that fails after 5 retries is dropped from the results
    *** Results are cached — if you upload the same CSV again the LLM is skipped
    """
    reviews = df["clean_text"].fillna("").tolist()
    batches = [(i, reviews[i:i+batch_size]) for i in range(0, len(reviews), batch_size)]

    successful_results = []
    failed_results     = []

    progress_bar    = st.progress(0)
    status_text     = st.empty()
    total_batches   = len(batches)
    completed       = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_themes_with_retry, b, themes_list): b
            for b in batches
        }

        for future in as_completed(futures):
            try:
                batch_idx, batch, batch_themes, status = future.result(timeout=180)
            except TimeoutError:
                b = futures[future]
                batch_idx, batch = b
                status      = "failed"
                batch_themes = [["FAILED"]] * len(batch)
                print(f"Batch {batch_idx} timed out.")
            except Exception as e:
                b = futures[future]
                batch_idx, batch = b
                status       = "failed"
                batch_themes = [["FAILED"]] * len(batch)
                print(f"Batch {batch_idx} crashed: {e}")

            if status == "success":
                for i, (_, vt) in enumerate(zip(batch, batch_themes)):
                    successful_results.append({
                        "original_idx": batch_idx + i,
                        "themes": ", ".join(vt),
                    })
            else:
                for i in range(len(batch)):
                    failed_results.append({
                        "original_idx": batch_idx + i,
                        "themes": "FAILED",
                    })

            completed += 1
            if completed % max(1, total_batches // 100) == 0 or completed == total_batches:
                progress_bar.progress(completed / total_batches)
                status_text.text(
                    f"Processed batch {completed}/{total_batches} — please wait..."
                )

    lookup     = {r["original_idx"]: r["themes"] for r in successful_results + failed_results}
    df["themes"] = [lookup.get(i, "FAILED") for i in range(len(df))]

    # drop reviews where theme extraction failed
    before = len(df)
    df = df[~df["themes"].str.contains("FAILED", na=False)]
    if len(df) < before:
        print(f"Dropped {before - len(df)} reviews due to extraction failure.")

    progress_bar.progress(1.0)
    status_text.text("Complete!")

    return df

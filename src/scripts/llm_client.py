# Wrapper function to communicate with local Ollama server.
# All LLM calls in this project go through this file.

import requests

# Ollama server address — runs locally on our machine, no internet needed
OLLAMA_URL = "http://localhost:11434/api/chat"

# Model being used for theme extraction
# gemma3:4b is chosen for its balance of speed and accuracy on CPU
DEFAULT_MODEL = "gemma3:4b"

def call_llm(prompt):
    """
    Send a prompt to the local Ollama server and return response as plain text.

    How it works:
        1. Sends HTTP POST request to Ollama running at localhost
        2. Ollama passes prompt to gemma3:4b model
        3. Model processes and returns response
        4. Response is cleaned and returned as a plain text string
           Note: response may look like JSON but is still a string

       Parameters:
        prompt (str): The text prompt to send to the model

    Returns:
        str: Clean response text from the model
             Example: '["Drink Quality", "Staff Friendliness"]'
   
    """

    # send prompt to Ollama server via HTTP request
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_ctx": 8192},
            "stream": False
        }
    )

    # extract the actual text content from the response dictionary
    content = response.json()["message"]["content"]

    # clean any markdown formatting the model might add around JSON
    # model sometimes wraps response in ```json ... ``` blocks
    content = content.replace("```json", "").replace("```", "").strip()
    return content
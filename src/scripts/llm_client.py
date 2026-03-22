import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma3:4b"

def call_llm(prompt):
    """
    Send prompt to Ollama, return response as plain text.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_ctx": 8192},
            "stream": False
        }
    )
    content = response.json()["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()
    return content
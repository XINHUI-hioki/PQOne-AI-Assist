import requests
import json

def ask_llama(prompt: str) -> str:
    """
    Send a prompt to the local Ollama chat API and return the response.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "HiokiAssist",  
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, json=payload, stream=True)

    if response.status_code == 200:
        reply = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        reply += json_data["message"]["content"]
                except json.JSONDecodeError:
                    print(f"\n[Warning] Failed to parse line: {line}")
        return reply
    else:
        raise RuntimeError(f"Request failed: {response.status_code} - {response.text}")

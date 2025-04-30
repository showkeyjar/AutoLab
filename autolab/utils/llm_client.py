import requests
from typing import Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434/api/generate", default_model: str = "llama3"):
        self.base_url = base_url
        self.default_model = default_model

    def send_prompt(self, prompt: str, model: Optional[str] = None) -> str:
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False
        }
        try:
            resp = requests.post(self.base_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Ollama 返回格式：{"response": "..."}
            return data.get("response", "")
        except Exception as e:
            return f"[LLM调用失败] {str(e)}"

ollama_client = OllamaClient()

import requests
from typing import Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434/api/generate", default_model: str = "llama3"):
        self.base_url = base_url.rstrip('/api/generate')  # 移除末尾的 '/api/generate'
        self.available_models = self._get_available_models()
        self.default_model = self.available_models[0] if self.available_models else default_model

    def _get_available_models(self):
        """获取Ollama服务可用模型列表"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return [model['name'] for model in response.json().get('models', [])]
            return []
        except requests.exceptions.RequestException:
            return []

    def is_healthy(self):
        """检查Ollama服务是否可用"""
        try:
            return requests.get(f"{self.base_url}").status_code == 200
        except requests.exceptions.RequestException:
            return False

    def ensure_ready(self):
        """确保服务就绪"""
        if not self.is_healthy():
            raise RuntimeError("Ollama服务不可用，请检查服务状态")

    def send_prompt(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or self.default_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Ollama 返回格式：{"response": "..."}
            return data.get("response", "")
        except Exception as e:
            return f"[LLM调用失败] {str(e)}"

ollama_client = OllamaClient()

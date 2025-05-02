import requests
import threading
from typing import Optional

class OllamaClient:
    base_url = "http://localhost:11434"  # 类属性
    
    def __init__(self):
        self._stop_event = threading.Event()
        self.available_models = self._get_available_models()
        self.default_model = self.available_models[0] if self.available_models else "llama3"

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

    def stop(self):
        """触发停止信号"""
        self._stop_event.set()

    def send_prompt(self, prompt: str, model: Optional[str] = None, timeout: int = 60) -> str:
        """可中断的请求方法"""
        model = model or self.default_model
        self._stop_event.clear()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
                hooks={'response': lambda r, *args, **kwargs: self._stop_event.is_set() and r.close()}
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            if self._stop_event.is_set():
                raise RuntimeError("LLM请求已被用户中断")
            raise

ollama_client = OllamaClient()

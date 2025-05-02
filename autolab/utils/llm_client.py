"""LLM客户端统一接口"""
from typing import Dict, Any, List
import requests
from autolab.core.logger import get_logger

logger = get_logger(__name__)

class OllamaClient:
    """Ollama客户端"""
    
    def __init__(self, model: str = "gemma:2b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._available_models: List[str] = []
        self._refresh_models()
    
    @property
    def available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self._available_models
    
    def _refresh_models(self):
        """从Ollama API获取模型列表"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                self._available_models = [
                    model["name"] 
                    for model in response.json().get("models", [])
                ]
            else:
                logger.warning(f"获取模型列表失败: {response.text}")
                self._available_models = [self.model]  # 默认保留当前模型
        except Exception as e:
            logger.error(f"连接Ollama服务失败: {str(e)}")
            self._available_models = [self.model]
    
    def generate(self, prompt: str) -> str:
        """生成文本"""
        logger.info(f"LLM生成请求: {prompt[:50]}...")
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt},
                timeout=60
            )
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            return f"生成错误: {str(e)}"
    
    def ping(self) -> bool:
        """测试连接"""
        try:
            return requests.get(f"{self.base_url}").status_code == 200
        except Exception:
            return False

def get_llm_client() -> OllamaClient:
    """获取LLM客户端实例"""
    return OllamaClient()

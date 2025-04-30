import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict

class WebClient:
    """
    简单网页浏览与内容提取工具。
    """
    def fetch(self, url: str, timeout: int = 10) -> Optional[str]:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            return None

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        # 获取正文文本
        return soup.get_text(separator=' ', strip=True)

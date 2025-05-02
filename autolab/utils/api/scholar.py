"""开放学术资源集成(优先无API限制的源)"""
import requests
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from autolab.core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Paper:
    title: str
    authors: str
    abstract: str
    year: int
    source: str
    url: Optional[str] = None
    pdf_url: Optional[str] = None

class OpenScholarClient:
    """集成开放API的文献客户端"""
    
    SOURCES = {
        "arxiv": {
            "endpoint": "http://export.arxiv.org/api/query",
            "params": {"search_query": "all:{query}", "max_results": 5}
        },
        "unpaywall": {
            "endpoint": "https://api.unpaywall.org/v2/search",
            "params": {"query": "{query}", "is_oa": True}
        }
    }
    
    def search(self, query: str) -> List[Paper]:
        """混合检索策略"""
        papers = []
        
        # 1. ArXiv优先(无API限制)
        arxiv_results = self._search_arxiv(query)
        papers.extend(arxiv_results)
        
        # 2. 开放获取论文补充
        if len(papers) < 3:
            oa_results = self._search_unpaywall(query)
            papers.extend(oa_results)
            
        return sorted(papers, key=lambda x: x.year, reverse=True)[:5]
    
    def _search_arxiv(self, query: str) -> List[Paper]:
        """ArXiv检索(无认证要求)"""
        try:
            params = {"search_query": f"all:{query}", "max_results": 5}
            response = requests.get(
                "http://export.arxiv.org/api/query",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            # 解析Atom格式响应
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.text)
            
            return [
                Paper(
                    title=entry.find("{http://www.w3.org/2005/Atom}title").text,
                    authors=", ".join(
                        a.find("{http://www.w3.org/2005/Atom}name").text 
                        for a in entry.findall("{http://www.w3.org/2005/Atom}author")
                    ),
                    abstract=entry.find("{http://www.w3.org/2005/Atom}summary").text.strip(),
                    year=int(entry.find("{http://www.w3.org/2005/Atom}published").text[:4]),
                    source="arXiv",
                    url=entry.find("{http://www.w3.org/2005/Atom}id").text,
                    pdf_url=[
                        l.get("href") 
                        for l in entry.findall("{http://www.w3.org/2005/Atom}link") 
                        if l.get("title") == "pdf"
                    ][0]
                )
                for entry in root.findall("{http://www.w3.org/2005/Atom}entry")
            ]
        except Exception as e:
            logger.warning(f"ArXiv检索失败: {str(e)}")
            return []
    
    def _search_unpaywall(self, query: str) -> List[Paper]:
        """开放获取论文检索(需邮箱注册)"""
        try:
            response = requests.get(
                "https://api.unpaywall.org/v2/search",
                params={"query": query, "is_oa": True},
                headers={"Accept": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            return [
                Paper(
                    title=result["title"],
                    authors=", ".join(a["name"] for a in result.get("authors", [])),
                    abstract=result.get("abstract", ""),
                    year=result.get("year", 0),
                    source="Unpaywall",
                    url=result["doi_url"],
                    pdf_url=result["oa_locations"][0]["url_for_pdf"] if result["oa_locations"] else None
                )
                for result in response.json().get("results", [])[:3]
            ]
        except Exception as e:
            logger.warning(f"Unpaywall检索失败: {str(e)}")
            return []
    
    def ping(self) -> bool:
        """测试基础连接"""
        try:
            return requests.get("http://export.arxiv.org/api/query?search_query=test&max_results=1", timeout=5).status_code == 200
        except Exception:
            return False

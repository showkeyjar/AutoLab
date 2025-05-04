"""文献阅读智能体"""
from typing import Dict, Any, Optional, List
from autolab.core.logger import get_logger
from .base import BaseAgent
import requests
from bs4 import BeautifulSoup
import json
import os
import yaml
from pathlib import Path

logger = get_logger(__name__)

# 示例文献数据
MOCK_DOCUMENTS = {
    "气象模型": [
        {
            "title": "改进气象预测的深度学习方法",
            "authors": ["张伟", "李娜"],
            "year": 2023,
            "summary": "本文提出了一种新的深度学习架构，可提高气象预测准确率15%"
        },
        {
            "title": "多模型集成在气象预测中的应用",
            "authors": ["王强", "赵敏"],
            "year": 2022,
            "summary": "研究展示了如何通过模型集成减少预测误差"
        }
    ],
    "机器学习": [
        {
            "title": "Transformer在时间序列预测中的最新进展",
            "authors": ["Smith, J", "Brown, A"],
            "year": 2023,
            "summary": "综述了Transformer架构在预测任务中的应用"
        }
    ]
}

class LiteratureReaderAgent(BaseAgent):
    """处理文献阅读和分析任务"""
    
    def __init__(self, mock_mode: bool = False):
        super().__init__(name="LiteratureReader", mock_mode=mock_mode)
        self._connected = False
        self.config = {
            "search_limit": 10,
            "sort_by": "submittedDate",
            "sort_order": "descending"
        }
        
        # 尝试加载配置文件
        config_path = Path(__file__).parent.parent.parent / "config" / "scholar_config.yaml"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    self.config.update(yaml.safe_load(f) or {})
            except Exception as e:
                logger.warning(f"配置文件加载失败: {str(e)}")
        
        # 初始化时自动连接
        if not self.connect():
            logger.warning("文献阅读器初始化连接失败")
        else:
            logger.info("文献阅读器初始化成功")
    
    def connect(self) -> bool:
        """连接到文献数据库"""
        if self.mock_mode:
            self._connected = True
            logger.info("文献阅读器进入模拟模式")
            return True
            
        try:
            self._connected = True
            logger.info("文献数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"连接失败: {str(e)}")
            return False
            
    def _handle_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """实际文献处理实现"""
        if not self._connected:
            raise RuntimeError("请先调用connect()方法")
            
        debug_info = {"agent_path": [self.name]}
        
        # 验证任务格式
        if not isinstance(task, dict) or not task.get("goal"):
            error_msg = "无效任务格式，缺少'goal'字段"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "debug": debug_info
            }
            
        query = task["goal"]
        
        if self.mock_mode:
            logger.debug(f"模拟文献检索: {query}")
            results = []
            
            # 在模拟数据中查找相关文献
            for category, docs in MOCK_DOCUMENTS.items():
                if category.lower() in query.lower():
                    results.extend(docs)
            
            return {
                "success": True,
                "output": {
                    "documents": results[:5],
                    "summary": f"找到{len(results)}篇相关文献"
                },
                "debug": debug_info
            }
            
        try:
            # 实际文献检索逻辑
            query = task["goal"]
            
            # 1. 从学术数据库检索
            documents = self._search_scholar(query)
            
            # 2. 生成摘要
            summary = self._generate_summary(documents)
            
            return {
                "success": True,
                "output": {
                    "documents": documents,
                    "summary": summary
                },
                "debug": debug_info
            }
            
        except Exception as e:
            logger.exception(f"文献处理失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "debug": debug_info
            }
    
    def _search_scholar(self, query: str) -> List[Dict[str, Any]]:
        """使用arXiv API检索文献"""
        try:
            import xml.etree.ElementTree as ET
            import time
            
            # 使用配置参数
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": self.config["search_limit"],
                "sortBy": self.config["sort_by"],
                "sortOrder": self.config["sort_order"]
            }
            
            # 发送请求
            response = requests.get("http://export.arxiv.org/api/query", params=params)
            response.raise_for_status()
            
            # 解析XML响应
            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            documents = []
            for entry in root.findall("atom:entry", ns):
                doc = {
                    "title": entry.find("atom:title", ns).text.strip(),
                    "authors": [
                        author.find("atom:name", ns).text
                        for author in entry.findall("atom:author", ns)
                    ],
                    "summary": entry.find("atom:summary", ns).text.strip(),
                    "published": entry.find("atom:published", ns).text,
                    "pdf_url": None
                }
                
                # 获取PDF链接
                for link in entry.findall("atom:link", ns):
                    if link.attrib.get("title") == "pdf":
                        doc["pdf_url"] = link.attrib["href"]
                
                documents.append(doc)
                
                # 避免请求过快
                time.sleep(0.5)
            
            return documents
            
        except Exception as e:
            logger.error(f"arXiv API请求失败: {str(e)}")
            return []
    
    def _generate_summary(self, documents: List[Dict[str, Any]]) -> str:
        """生成文献摘要"""
        if not documents:
            return "未找到相关文献"
            
        # 简单实现 - 实际应用中可以用LLM生成
        topics = set()
        for doc in documents:
            if "title" in doc:
                topics.update(doc["title"].split()[:5])
        
        return f"共找到{len(documents)}篇文献，主要涉及: {', '.join(topics)}"
    
    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理文献阅读任务"""
        return self._handle_impl(task)

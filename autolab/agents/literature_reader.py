from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client
import json

class LiteratureReaderAgent(BaseAgent):
    """
    文献阅读智能体：自动检索、分析和整合科学文献，为实验设计和优化提供知识支持。
    """
    def __init__(self):
        super().__init__(name="LiteratureReaderAgent")

    def handle(self, task: Dict[str, Any]) -> Any:
        user_goal = task.get("user_task", {}).get("goal") if isinstance(task.get("user_task"), dict) else task.get("goal")
        if not user_goal:
            user_goal = "未知目标"
        # 构造结构化prompt，要求LLM输出JSON
        prompt = (
            f"请作为化学领域的智能文献分析助手，根据最新文献，列举3条与‘{user_goal}’相关的化学实验研究进展，输出JSON数组，每条包含title, year, summary字段。例如："
            '{"papers": [{"title": "...", "year": 2024, "summary": "..."}, ...]}'
        )
        llm_response = ollama_client.send_prompt(prompt)
        # 尝试结构化解析
        papers = []
        try:
            data = json.loads(llm_response)
            papers = data.get("papers", [])
        except Exception:
            papers = [{"desc": llm_response}]
        print(f"[LiteratureReaderAgent] LLM文献检索结果: {papers}")
        return {"papers": papers, "goal": user_goal, "llm_raw": llm_response}

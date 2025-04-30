from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client

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
        # 构造prompt
        prompt = f"请根据最新文献，列举3条与‘{user_goal}’相关的化学实验研究进展，并简要说明每条的意义。格式：1. ... 2. ... 3. ..."
        llm_response = ollama_client.send_prompt(prompt)
        # 简单结构化解析（如遇异常则原样返回）
        papers = []
        try:
            for idx, line in enumerate(llm_response.split("\n")):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    papers.append({"desc": line})
            if not papers:
                papers = [{"desc": llm_response}]
        except Exception:
            papers = [{"desc": llm_response}]
        print(f"[LiteratureReaderAgent] LLM文献检索结果: {papers}")
        return {"papers": papers, "goal": user_goal, "llm_raw": llm_response}

from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client
import json

class ExperimentDesignerAgent(BaseAgent):
    """
    实验设计智能体：根据目标和知识生成实验方案，进行流程逻辑验证。
    """
    def __init__(self):
        super().__init__(name="ExperimentDesignerAgent")

    def handle(self, task: Dict[str, Any]) -> Any:
        user_goal = task.get("user_task", {}).get("goal") if isinstance(task.get("user_task"), dict) else task.get("goal")
        papers = task.get("literature", {}).get("papers", [])
        # 构造结构化 prompt，要求 LLM 输出 JSON
        paper_titles = ", ".join([p.get("title", p.get("desc", str(p))) for p in papers]) if papers else "无相关文献"
        prompt = (
            f"你是自动化实验设计智能体。请根据实验目标“{user_goal}”和相关文献“{paper_titles}”，输出JSON格式的实验方案，包含goal, steps(步骤数组，每步含desc和参数), key_parameters, 注意事项。例如："
            '{"goal": "...", "steps": [{"desc": "...", "params": {...}}, ...], "key_parameters": [...], "notes": "..."}'
        )
        llm_response = ollama_client.send_prompt(prompt)
        # 尝试结构化解析
        scheme = {}
        try:
            scheme = json.loads(llm_response)
        except Exception:
            scheme = {"goal": user_goal, "steps": [], "llm_raw": llm_response}
        print(f"[ExperimentDesignerAgent] LLM实验方案: {scheme}")
        return scheme

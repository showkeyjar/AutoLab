from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client
import json

class TaskManagerAgent(BaseAgent):
    """
    任务管理智能体：统一协调、分配和监控所有子智能体任务。
    """
    def __init__(self):
        super().__init__(name="TaskManagerAgent")

    def handle(self, task: Dict[str, Any]) -> Any:
        user_goal = task.get("goal") or task.get("user_task", {}).get("goal")
        if not user_goal:
            user_goal = "未知目标"
        # 明确要求LLM输出结构化JSON
        prompt = (
            f"你是自动化实验室的任务管理智能体。请将实验目标“{user_goal}”分解为3~5个具体子任务，输出JSON数组，每个子任务包含id, desc, priority字段。例如："
            '{"subtasks": [{"id": 1, "desc": "...", "priority": "high"}, ...]}'
        )
        llm_response = ollama_client.send_prompt(prompt)
        subtasks = []
        try:
            data = json.loads(llm_response)
            subtasks = data.get("subtasks", [])
        except Exception:
            # fallback: 尝试逐行解析
            for line in llm_response.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    subtasks.append({"desc": line})
            if not subtasks:
                subtasks = [{"desc": llm_response}]
        print(f"[TaskManagerAgent] LLM分解子任务: {subtasks}")
        return {"goal": user_goal, "subtasks": subtasks, "llm_raw": llm_response}

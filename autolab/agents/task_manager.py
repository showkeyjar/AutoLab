from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client

class TaskManagerAgent(BaseAgent):
    """
    任务管理智能体：统一协调、分配和监控所有子智能体任务。
    """
    def __init__(self):
        super().__init__(name="TaskManagerAgent")

    def handle(self, task: Dict[str, Any]) -> Any:
        # 使用LLM自动分解用户目标为子任务
        user_goal = task.get("goal") or task.get("user_task", {}).get("goal")
        if not user_goal:
            user_goal = "未知目标"
        prompt = f"你是一位自动化实验室的任务管理智能体。请将用户提出的实验目标“{user_goal}”分解为3~5个具体的子任务，每个子任务用一句话描述，按执行顺序编号。"
        llm_response = ollama_client.send_prompt(prompt)
        # 结构化解析LLM输出
        subtasks = []
        try:
            for line in llm_response.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    subtasks.append(line)
            if not subtasks:
                subtasks = [llm_response]
        except Exception:
            subtasks = [llm_response]
        print(f"[TaskManagerAgent] LLM分解子任务: {subtasks}")
        return {"goal": user_goal, "subtasks": subtasks, "llm_raw": llm_response}

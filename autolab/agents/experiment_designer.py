from .base import BaseAgent
from typing import Any, Dict

class ExperimentDesignerAgent(BaseAgent):
    """
    实验设计智能体：根据目标和知识生成实验方案，进行流程逻辑验证。
    """
    def __init__(self):
        super().__init__(name="ExperimentDesignerAgent")

    def handle(self, task: Dict[str, Any]) -> Any:
        # 简单模拟实验方案生成与逻辑验证
        user_goal = task.get("user_task", {}).get("goal") if isinstance(task.get("user_task"), dict) else task.get("goal")
        papers = task.get("literature", {}).get("papers", [])
        # 生成实验步骤
        steps = [
            f"查阅文献：{papers[0]['title']}" if papers else "查阅相关文献",
            f"准备实验材料，目标：{user_goal}",
            "设置实验参数，执行主实验流程",
            "采集实验数据，进行初步分析"
        ]
        scheme = {
            "goal": user_goal,
            "reference": papers[0] if papers else None,
            "steps": steps
        }
        print(f"[ExperimentDesignerAgent] 生成实验方案: {scheme}")
        return scheme

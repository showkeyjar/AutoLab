from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client

class ExperimentDesignerAgent(BaseAgent):
    """
    实验设计智能体：根据目标和知识生成实验方案，进行流程逻辑验证。
    """
    def __init__(self):
        super().__init__(name="ExperimentDesignerAgent")

    def handle(self, task: Dict[str, Any]) -> Any:
        user_goal = task.get("user_task", {}).get("goal") if isinstance(task.get("user_task"), dict) else task.get("goal")
        papers = task.get("literature", {}).get("papers", [])
        # 构造 prompt，融合用户目标和文献知识
        paper_descs = "\n".join([p.get("desc", str(p)) for p in papers]) if papers else "无相关文献"
        prompt = (
            f"你是一位自动化化学实验设计智能体。请根据以下实验目标和相关文献，生成一套详细的实验方案，包含实验目的、主要步骤、关键参数和注意事项。\n"
            f"实验目标：{user_goal}\n相关文献：\n{paper_descs}"
        )
        llm_response = ollama_client.send_prompt(prompt)
        # 尝试结构化解析 LLM 输出
        scheme = {
            "goal": user_goal,
            "reference": papers[0] if papers else None,
            "steps": [],
            "llm_raw": llm_response
        }
        try:
            lines = [l.strip() for l in llm_response.split("\n") if l.strip()]
            for line in lines:
                if line.startswith("- ") or line[0].isdigit():
                    scheme["steps"].append(line)
                elif "目的" in line or "步骤" in line or "参数" in line or "注意" in line:
                    scheme.setdefault("sections", []).append(line)
            if not scheme["steps"]:
                scheme["steps"] = lines
        except Exception:
            scheme["steps"] = [llm_response]
        print(f"[ExperimentDesignerAgent] LLM实验方案: {scheme}")
        return scheme

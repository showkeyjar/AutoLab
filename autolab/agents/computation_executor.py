from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client

class ComputationExecutorAgent(BaseAgent):
    """
    计算执行智能体：负责模型预测、数据分析与实验闭环优化。
    """
    def __init__(self):
        super().__init__(name="ComputationExecutorAgent")

    def handle(self, task: Dict[str, Any]) -> Any:
        robot_result = task.get("robot", {})
        executed = robot_result.get("executed_instructions", [])
        log = robot_result.get("log", "无实验日志")
        # 构造 prompt，融合实验指令与日志
        instructions_text = "\n".join(executed) if executed else "无实验指令"
        prompt = (
            f"你是一位自动化实验室的数据分析与优化智能体。请根据以下实验执行指令和日志，总结实验结果，分析潜在问题，并提出优化建议。\n"
            f"实验指令：\n{instructions_text}\n实验日志：{log}"
        )
        llm_response = ollama_client.send_prompt(prompt)
        # 结构化解析 LLM 输出
        analysis = {
            "summary": "",
            "recommendation": "",
            "raw_log": log,
            "llm_raw": llm_response
        }
        try:
            lines = [l.strip() for l in llm_response.split("\n") if l.strip()]
            for line in lines:
                if "总结" in line or "结果" in line:
                    analysis["summary"] += line + " "
                elif "建议" in line or "优化" in line:
                    analysis["recommendation"] += line + " "
            if not (analysis["summary"] or analysis["recommendation"]):
                analysis["summary"] = llm_response
        except Exception:
            analysis["summary"] = llm_response
        print(f"[ComputationExecutorAgent] LLM分析与优化: {analysis}")
        return analysis

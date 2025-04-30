from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client
import json

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
        # 构造结构化 prompt，要求 LLM 输出 JSON
        instructions_text = ", ".join([i.get("instruction", str(i)) if isinstance(i, dict) else str(i) for i in executed]) if executed else "无实验指令"
        prompt = (
            f"你是自动化实验室的数据分析与优化智能体。请根据实验执行指令“{instructions_text}”和日志“{log}”，输出JSON格式的分析结果，包含summary, recommendation字段。例如："
            '{"summary": "...", "recommendation": "..."}'
        )
        llm_response = ollama_client.send_prompt(prompt)
        # 尝试结构化解析
        analysis = {"summary": "", "recommendation": "", "raw_log": log, "llm_raw": llm_response}
        try:
            data = json.loads(llm_response)
            analysis["summary"] = data.get("summary", "")
            analysis["recommendation"] = data.get("recommendation", "")
        except Exception:
            analysis["summary"] = llm_response
        print(f"[ComputationExecutorAgent] LLM分析与优化: {analysis}")
        return analysis

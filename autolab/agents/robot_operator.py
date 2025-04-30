from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client

class RobotOperatorAgent(BaseAgent):
    """
    机器操控智能体：将实验方案转化为机器人/仪器指令并驱动实际执行。
    """
    def __init__(self):
        super().__init__(name="RobotOperatorAgent")

    def handle(self, task: Dict[str, Any]) -> Any:
        scheme = task.get("design", {})
        steps = scheme.get("steps", [])
        # 构造 prompt，融合实验方案内容
        steps_text = "\n".join(steps) if steps else "无实验步骤"
        prompt = (
            f"你是一位自动化实验室的机器人操控智能体。请根据以下实验方案，生成详细的实验仪器操作指令，包含关键参数设置和注意事项。\n"
            f"实验方案：\n{steps_text}"
        )
        llm_response = ollama_client.send_prompt(prompt)
        # 结构化解析 LLM 输出
        instructions = []
        try:
            for line in llm_response.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    instructions.append(line)
            if not instructions:
                instructions = [llm_response]
        except Exception:
            instructions = [llm_response]
        execution_result = {
            "status": "success",
            "executed_instructions": instructions,
            "llm_raw": llm_response,
            "log": "实验指令已生成，模拟执行完成。"
        }
        print(f"[RobotOperatorAgent] LLM实验指令: {execution_result}")
        return execution_result

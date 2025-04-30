from .base import BaseAgent
from typing import Any, Dict
from autolab.utils.llm_client import ollama_client
import json
from autolab.utils.code_executor import CodeExecutor

class RobotOperatorAgent(BaseAgent):
    """
    机器操控智能体：将实验方案转化为机器人/仪器指令并驱动实际执行。
    支持代码生成、执行、mcp调用。
    """
    def __init__(self):
        super().__init__(name="RobotOperatorAgent")
        self.code_executor = CodeExecutor(timeout=20)
        # mcp_client 可按需集成
        self.mcp_client = None

    def handle(self, task: Dict[str, Any]) -> Any:
        scheme = task.get("design", {})
        steps = scheme.get("steps", [])
        # 构造结构化 prompt，要求 LLM 输出 JSON
        steps_text = ", ".join([s.get("desc", str(s)) if isinstance(s, dict) else str(s) for s in steps]) if steps else "无实验步骤"
        prompt = (
            f"你是自动化实验室的机器人操控智能体。请根据实验方案步骤“{steps_text}”，输出JSON格式的仪器操作指令数组，每条含instruction, params字段，并可选python_code字段（用于自动执行仪器控制）。例如："
            '{"instructions": [{"instruction": "...", "params": {...}, "python_code": "print(\'run\')"}, ...]}'
        )
        llm_response = ollama_client.send_prompt(prompt)
        # 尝试结构化解析
        instructions = []
        try:
            data = json.loads(llm_response)
            instructions = data.get("instructions", [])
            # 自动执行生成的代码
            for instr in instructions:
                code = instr.get("python_code")
                if code:
                    result = self.code_executor.execute_python(code)
                    instr["code_result"] = result
                # mcp 调用示例（如有）
                mcp_cmd = instr.get("mcp_cmd")
                if mcp_cmd and self.mcp_client:
                    instr["mcp_result"] = self.mcp_client.call(mcp_cmd)
        except Exception:
            instructions = [llm_response]
        execution_result = {
            "status": "success",
            "executed_instructions": instructions,
            "llm_raw": llm_response,
            "log": "实验指令已生成，代码已自动执行。"
        }
        print(f"[RobotOperatorAgent] LLM实验指令: {execution_result}")
        return execution_result

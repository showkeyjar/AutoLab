"""实验设计智能体"""
from typing import Dict, Any, List, Optional
from autolab.core.logger import get_logger
from .base import BaseAgent

logger = get_logger(__name__)

class ExperimentDesignerAgent(BaseAgent):
    """生成实验设计方案"""
    
    PROMPT_TEMPLATE = """
    根据以下研究目标设计实验方案:
    领域: {field}
    目标: {goal}
    约束条件: {constraints}
    
    请包含:
    1. 研究假设
    2. 自变量和因变量
    3. 实验步骤
    4. 所需材料
    5. 评估指标分析 (根据用户提供的基准: {benchmark})
    6. 建议的深度学习模型架构:
        根据目标、任务类型和约束条件，提出一个具体的深度学习模型架构。描述:
        - 整体架构类型 (例如, CNN, RNN, Transformer, MLP).
        - 关键层及其参数 (例如, Conv2D(filters=32, kernel_size=3), LSTM(units=64), Dense(units=10, activation='softmax')).
        - 激活函数.
        - 优化器建议 (例如, Adam, SGD).
        - 损失函数建议 (根据任务类型).
        请将此部分作为结构化的JSON对象提供，键为 'suggested_model_architecture'.
    
    对于评估指标部分，请分析:
    - 适用的量化指标
    - 每个指标的权重
    - 达标阈值
    """
    
    def __init__(self, mock_mode: bool = False):
        super().__init__(name="ExperimentDesigner", mock_mode=mock_mode)
        self._llm = None
        self._connected = False
    
    def connect(self) -> bool:
        """连接到实验设计服务"""
        if self.mock_mode:
            self._connected = True
            return True
            
        try:
            from autolab.utils.llm_client import OllamaClient
            self._llm = OllamaClient()
            self._connected = True
            logger.info(f"{self.name} 连接成功")
            return True
        except Exception as e:
            logger.error(f"连接失败: {str(e)}")
            self._connected = False
            return False
    
    def _handle_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """实际实验设计实现"""
        if not self._connected:
            raise RuntimeError("请先调用connect()方法")
            
        if self.mock_mode:
            return {
                "success": True,
                "output": {"design": "模拟实验方案"},
                "debug": {"agent_path": [self.name]}
            }
            
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                field="科学研究",
                goal=task.get("goal", ""),
                constraints=task.get("constraints", ""),
                benchmark=task.get("benchmark", "未提供具体基准")
            )
            response = self._llm.generate(prompt)
            design = self._parse_response(response)
            return {
                'success': True,
                'output': {"design": design},
                'debug': {"agent_path": [self.name]}
            }
        except Exception as e:
            logger.error(f"设计失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'debug': {"agent_path": [self.name]}
            }
    
    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理实验设计请求"""
        return self._handle_impl(task)
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """解析LLM响应"""
        import re
        import json
        
        parsed_data = {}
        
        # Extract suggested_model_architecture JSON
        model_architecture_match = re.search(r'"suggested_model_architecture":\s*(\{.*?\})\s*(?=\n\d+\.|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        suggested_model_architecture = None
        if model_architecture_match:
            json_string = model_architecture_match.group(1)
            try:
                suggested_model_architecture = json.loads(json_string)
                # Remove the JSON block from the text to avoid parsing issues with sections
                text = text.replace(model_architecture_match.group(0), "")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse model architecture JSON: {e}")
                suggested_model_architecture = {"error": "Failed to parse JSON"}

        sections = re.split(r'\n\d+\. ', text.strip())
        
        # Helper to safely get section content
        def get_section_content(index, split_lines=False):
            if len(sections) > index:
                content = sections[index].strip()
                if split_lines:
                    return [s.strip() for s in content.split('\n') if s.strip()]
                return [content]
            return []

        parsed_data["hypothesis"] = get_section_content(1)
        parsed_data["variables"] = get_section_content(2)
        parsed_data["procedure"] = get_section_content(3, split_lines=True)
        parsed_data["materials"] = get_section_content(4, split_lines=True)
        
        # Metrics parsing - assuming it's section 5
        metrics_section_text = ""
        if len(sections) > 5:
             # Remove "评估指标分析 (...)" header and "对于评估指标部分，请分析:" if present
            metrics_section_text = sections[5].split("对于评估指标部分，请分析:", 1)[-1].strip()
            # Further clean up potential leading "评估指标分析" and similar headers
            metrics_section_text = re.sub(r"^\s*评估指标分析\s*\(根据用户提供的基准:.*?\)\s*", "", metrics_section_text, flags=re.IGNORECASE).strip()


        metrics_lines = [
            line.strip()
            for line in metrics_section_text.split('\n')
            if line.strip() and line.startswith('-') # Expecting metrics to be bullet points
        ]
        parsed_data["metrics"] = metrics_lines
        
        if suggested_model_architecture is not None:
            parsed_data["suggested_model_architecture"] = suggested_model_architecture
        
        return parsed_data

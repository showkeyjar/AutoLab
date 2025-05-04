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
    
    def _parse_response(self, text: str) -> Dict[str, List[str]]:
        """解析LLM响应"""
        import re
        sections = re.split(r'\n\d+\. ', text.strip())
        metrics = []
        if len(sections) > 5:
            # 解析评估指标部分
            metrics = [
                line.strip() 
                for line in sections[5].split('\n') 
                if line.strip() and not line.startswith('-')
            ]
        
        return {
            "hypothesis": [sections[1].strip()] if len(sections) > 1 else [],
            "variables": [sections[2].strip()] if len(sections) > 2 else [],
            "procedure": [s.strip() for s in sections[3].split('\n')] if len(sections) > 3 else [],
            "materials": [s.strip() for s in sections[4].split('\n')] if len(sections) > 4 else [],
            "metrics": metrics
        }

"""Agent基类"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime
from autolab.core.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """所有Agent的抽象基类"""
    
    def __init__(self, name: str = None, mock_mode: bool = False):
        self.name = name or self.__class__.__name__.replace("Agent", "")
        self.mock_mode = mock_mode
        self._connected = False
        self._last_active = "未初始化"
        logger.info(f"{self.name} Agent初始化 (mock_mode={mock_mode})")
    
    @property
    def is_ready(self) -> bool:
        """检查Agent是否就绪"""
        return self._connected
    
    @abstractmethod
    def connect(self) -> bool:
        """连接所需服务"""
        pass
    
    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """记录活动时间并处理任务"""
        self._last_active = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.debug(f"{self.name} 开始处理任务: {task.get('goal','')}")
        return self._handle_impl(task)
    
    @abstractmethod
    def _handle_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """实际的任务处理逻辑"""
        pass

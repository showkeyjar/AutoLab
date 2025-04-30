from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    """
    智能体基类，所有具体智能体需继承此类。
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def handle(self, task: Dict[str, Any]) -> Any:
        """
        处理分配到的任务，返回结果。
        """
        pass

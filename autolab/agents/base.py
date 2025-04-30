from abc import ABC, abstractmethod
from typing import Any, Dict
import json

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

    def to_json(self):
        return json.dumps({"name": self.name, "type": self.__class__.__name__})

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(name=data.get("name", "UnknownAgent"))

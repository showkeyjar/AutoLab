from abc import ABC, abstractmethod
from typing import Any, Dict
import json
import time

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

    def _execute(self, task):
        return self.handle(task)

    def execute(self, task):
        """执行任务并返回详细结果"""
        self.recorded_actions = []
        result = {
            'success': False,
            'output': None,
            'actions': [],
            'details': {}
        }
        
        try:
            # 记录详细执行过程
            start_time = time.time()
            output = self._execute(task)
            
            result.update({
                'success': True,
                'output': output,
                'actions': self.recorded_actions,
                'details': {
                    'start_time': start_time,
                    'end_time': time.time(),
                    'steps': len(self.recorded_actions)
                }
            })
        except Exception as e:
            result['error'] = str(e)
            result['actions'] = self.recorded_actions
            
        return result

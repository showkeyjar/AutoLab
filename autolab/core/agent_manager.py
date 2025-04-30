from autolab.agents.task_manager import TaskManagerAgent
from autolab.agents.literature_reader import LiteratureReaderAgent
from autolab.agents.experiment_designer import ExperimentDesignerAgent
from autolab.agents.robot_operator import RobotOperatorAgent
from autolab.agents.computation_executor import ComputationExecutorAgent

class AgentManager:
    """
    智能体注册与调度管理器，统一管理所有智能体实例。
    支持动态注册/注销智能体。
    """
    def __init__(self):
        self.agents = {
            "task_manager": TaskManagerAgent(),
            "literature_reader": LiteratureReaderAgent(),
            "experiment_designer": ExperimentDesignerAgent(),
            "robot_operator": RobotOperatorAgent(),
            "computation_executor": ComputationExecutorAgent(),
        }

    def register_agent(self, name: str, agent):
        self.agents[name] = agent

    def unregister_agent(self, name: str):
        if name in self.agents:
            del self.agents[name]

    def get_agent(self, agent_name: str):
        return self.agents.get(agent_name)

    def dispatch(self, agent_name: str, task):
        agent = self.get_agent(agent_name)
        if agent:
            return agent.handle(task)
        else:
            raise ValueError(f"Agent '{agent_name}' not found.")

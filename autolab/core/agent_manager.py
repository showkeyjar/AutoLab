from autolab.agents.task_manager import TaskManagerAgent
from autolab.agents.literature_reader import LiteratureReaderAgent
from autolab.agents.experiment_designer import ExperimentDesignerAgent
from autolab.agents.robot_operator import RobotOperatorAgent
from autolab.agents.computation_executor import ComputationExecutorAgent
from autolab.agents.task_enhancer import TaskEnhancerAgent
from autolab.agents.data_acquisition import DataAcquisitionAgent # Added import
from autolab.core.logger import get_logger
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

logger = get_logger(__name__)

class AgentManager:
    """
    智能体注册与调度管理器，统一管理所有智能体实例。
    支持动态注册/注销智能体。
    """
    def __init__(self, mock_mode=False):
        """只初始化核心Agent"""
        # 先创建AgentManager实例
        self.agents: Dict[str, Any] = {}
        
        # 创建TaskManagerAgent时需要传入当前AgentManager实例
        self.agents["task_manager"] = TaskManagerAgent(
            mock_mode=mock_mode, 
            agent_manager=self
        )
        
        # 其他Agent初始化
        self.agents.update({
            "literature_reader": LiteratureReaderAgent(mock_mode=mock_mode),
            "experiment_designer": ExperimentDesignerAgent(mock_mode=mock_mode),
            "robot_operator": RobotOperatorAgent(mock_mode=mock_mode),
            "computation_executor": ComputationExecutorAgent(mock_mode=mock_mode),
            "task_enhancer": TaskEnhancerAgent(mock_mode=mock_mode),
            "data_acquisition_agent": DataAcquisitionAgent(mock_mode=mock_mode) # Added DataAcquisitionAgent
        })
        
        logger.info(f"核心Agent初始化完成: {list(self.agents.keys())}")
        
        # 记录每个Agent的初始化状态
        for name, agent in self.agents.items():
            logger.debug(
                f"Agent '{name}' 初始化: "
                f"类型={type(agent).__name__}, "
                f"mock_mode={getattr(agent, 'mock_mode', 'N/A')}"
            )
        
        # 连接所有智能体
        for agent_name, agent in self.agents.items():
            if not agent.connect():
                logger.warning(f"智能体 {agent_name} 连接失败")

    def register_agent(self, name: str, agent):
        self.agents[name] = agent

    def unregister_agent(self, name: str):
        if name in self.agents:
            del self.agents[name]

    def get_agent(self, agent_name: str):
        """获取指定Agent并记录调用"""
        agent = self.agents.get(agent_name)
        if agent:
            logger.debug(f"获取Agent: {agent_name} (类型: {type(agent).__name__})")
        else:
            logger.error(f"请求了不存在的Agent: {agent_name}")
        return agent

    def dispatch(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """分发任务并记录详细日志"""
        logger.info(f"将任务分发给 {agent_name}: {task.get('goal','')}")
        if agent_name not in self.agents:
            logger.error(f"拒绝请求不存在的Agent: {agent_name}")
            return {
                "success": False,
                "error": f"Agent {agent_name} 未注册",
                "metrics": {"accuracy": 0.0}  # 提供默认指标
            }
            
        agent = self.get_agent(agent_name)
        if not agent:
            return {"error": f"Agent {agent_name} 不存在"}
            
        try:
            result = agent.handle(task)
            logger.debug(f"Agent {agent_name} 返回结果: {result.get('success', False)}")
            return result
        except Exception as e:
            logger.exception(f"Agent {agent_name} 处理异常")
            return {"error": str(e)}

    def test_connections(self):
        """测试所有Agent的连接状态"""
        results = {}
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'is_ready') and callable(agent.is_ready):
                    results[name] = "🟢 在线" if agent.is_ready() else "🔴 离线"
                else:
                    results[name] = "⚪ 无状态检查方法"
            except Exception as e:
                results[name] = f"🔴 连接失败: {str(e)}"
        return results

    def start_collaboration(self):
        """初始化所有Agent连接"""
        logger.info(f"正在初始化{len(self.agents)}个Agent...")
        
        # 并行测试所有连接
        with ThreadPoolExecutor() as executor:
            futures = {
                name: executor.submit(agent.connect)
                for name, agent in self.agents.items()
            }
        
        # 检查结果
        online_count = 0
        for name, future in futures.items():
            try:
                if future.result():
                    online_count += 1
                    logger.info(f"{name} 连接成功")
                else:
                    logger.error(f"{name} 连接失败")
            except Exception as e:
                logger.error(f"{name} 初始化异常: {str(e)}")
        
        if online_count == 0:
            raise RuntimeError("所有Agent初始化失败！请检查日志")
        
        return online_count

    def _format_response(self, response):
        if not isinstance(response, dict):
            response = {'output': response}
        
        response.setdefault('history', [])
        response.setdefault('success', True)
        
        return response

    def _format_error_response(self, agent_name, error_msg):
        return {
            'output': None,
            'success': False,
            'error': error_msg,
            'history': []
        }

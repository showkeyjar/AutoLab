from autolab.core.agent_manager import AgentManager

class TaskFlow:
    """
    细化任务流转：每个智能体的输出作为下一个智能体的输入，实现阶段性数据传递。
    """
    def __init__(self):
        self.agent_manager = AgentManager()

    def run_flow(self, user_task: dict):
        # 1. 任务管理智能体分解目标
        task_manager_result = self.agent_manager.dispatch("task_manager", user_task)

        # 2. 文献阅读智能体处理（以任务管理结果为输入）
        literature_input = {"user_task": user_task, "task_manager": task_manager_result}
        literature_result = self.agent_manager.dispatch("literature_reader", literature_input)

        # 3. 实验设计智能体处理（以前两步结果为输入）
        design_input = {"user_task": user_task, "task_manager": task_manager_result, "literature": literature_result}
        design_result = self.agent_manager.dispatch("experiment_designer", design_input)

        # 4. 机器操控智能体处理（以前面结果为输入）
        robot_input = {"user_task": user_task, "task_manager": task_manager_result, "literature": literature_result, "design": design_result}
        robot_result = self.agent_manager.dispatch("robot_operator", robot_input)

        # 5. 计算执行智能体处理（以全部阶段数据为输入）
        computation_input = {
            "user_task": user_task,
            "task_manager": task_manager_result,
            "literature": literature_result,
            "design": design_result,
            "robot": robot_result
        }
        computation_result = self.agent_manager.dispatch("computation_executor", computation_input)

        # 汇总结果
        return {
            "task_manager": task_manager_result,
            "literature_reader": literature_result,
            "experiment_designer": design_result,
            "robot_operator": robot_result,
            "computation_executor": computation_result
        }

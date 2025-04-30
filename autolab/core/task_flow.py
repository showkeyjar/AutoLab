from autolab.core.agent_manager import AgentManager
import uuid
import time

class TaskFlow:
    """
    细化任务流转：每个智能体的输出作为下一个智能体的输入，实现阶段性数据传递。
    """
    def __init__(self):
        self.agent_manager = AgentManager()

    def run_flow(self, user_task: dict):
        # 分配唯一实验ID
        experiment_id = user_task.get("experiment_id") or str(uuid.uuid4())
        user_task["experiment_id"] = experiment_id
        logs = []
        evaluation_history = []
        def log_stage(stage, status, detail=None):
            logs.append({
                "stage": stage,
                "status": status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detail": detail or ""
            })
        # 设定目标评测标准
        evaluation_criteria = user_task.get("evaluation_criteria") or {
            "success_metric": "result meets or exceeds user goal",
            "max_iterations": 5,
            "improvement_required": True
        }
        user_task["evaluation_criteria"] = evaluation_criteria
        best_result = None
        best_score = float('-inf')
        for iteration in range(evaluation_criteria["max_iterations"]):
            try:
                # 1. 任务管理
                task_manager_result = self.agent_manager.dispatch("task_manager", user_task)
                log_stage("task_manager", "success")
                # 2. 文献阅读
                literature_input = {"user_task": user_task, "task_manager": task_manager_result, "experiment_id": experiment_id}
                literature_result = self.agent_manager.dispatch("literature_reader", literature_input)
                log_stage("literature_reader", "success")
                # 3. 实验设计
                design_input = {"user_task": user_task, "task_manager": task_manager_result, "literature": literature_result, "experiment_id": experiment_id}
                design_result = self.agent_manager.dispatch("experiment_designer", design_input)
                log_stage("experiment_designer", "success")
                # 4. 机器操控
                robot_input = {"user_task": user_task, "task_manager": task_manager_result, "literature": literature_result, "design": design_result, "experiment_id": experiment_id}
                robot_result = self.agent_manager.dispatch("robot_operator", robot_input)
                log_stage("robot_operator", "success")
                # 5. 计算执行
                computation_input = {
                    "user_task": user_task,
                    "task_manager": task_manager_result,
                    "literature": literature_result,
                    "design": design_result,
                    "robot": robot_result,
                    "experiment_id": experiment_id,
                    "evaluation_criteria": evaluation_criteria
                }
                computation_result = self.agent_manager.dispatch("computation_executor", computation_input)
                log_stage("computation_executor", "success")
                # 6. 结果自我评价
                score, eval_detail = self.self_evaluate(computation_result, evaluation_criteria)
                evaluation_history.append({
                    "iteration": iteration+1,
                    "score": score,
                    "detail": eval_detail,
                    "result": computation_result
                })
                if score > best_score:
                    best_score = score
                    best_result = computation_result
                # 若达到目标则提前终止
                if evaluation_criteria["improvement_required"] and score >= 1.0:
                    break
            except Exception as e:
                log_stage("iteration_%d" % (iteration+1), "failed", str(e))
                break
        # 汇总
        return {
            "experiment_id": experiment_id,
            "status": "success" if best_result else "failed",
            "logs": logs,
            "evaluation_history": evaluation_history,
            "best_result": best_result
        }

    def self_evaluate(self, computation_result, evaluation_criteria):
        """
        智能体自我评价：根据评测标准对实验结果打分，1.0为完全达标。
        """
        # 示例：若 computation_result 含有 "recommendation" 或 "summary" 满足目标则得分高
        goal = evaluation_criteria.get("goal") or ""
        summary = computation_result.get("summary", "")
        recommendation = computation_result.get("recommendation", "")
        # 简单规则：包含目标关键词则判为达标
        if goal and (goal in summary or goal in recommendation):
            return 1.0, "目标关键词已覆盖，达标"
        # 若有明确优化建议也给较高分
        if recommendation:
            return 0.8, "有优化建议"
        if summary:
            return 0.5, "有实验总结"
        return 0.0, "无有效结果"

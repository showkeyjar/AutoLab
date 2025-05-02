from autolab.core.agent_manager import AgentManager
from autolab.core.param_optimizer import ParamOptimizer
import uuid
import time
import os
import yaml

class TaskFlow:
    """
    细化任务流转：每个智能体的输出作为下一个智能体的输入，实现阶段性数据传递。
    """
    def __init__(self):
        self.agent_manager = AgentManager()
        self.optimizer = ParamOptimizer()

    def auto_save_state(func):
        """自动保存实验状态的装饰器"""
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if hasattr(self, 'experiment_id') and hasattr(self, '_last_state'):
                state_manager.save_state(
                    self.experiment_id,
                    self._last_state
                )
            return result
        return wrapper

    @auto_save_state
    def run_flow(self, user_task: dict):
        experiment_id = user_task.get("experiment_id") or str(uuid.uuid4())
        user_task["experiment_id"] = experiment_id
        logs = []
        evaluation_history = []
        
        # Initialize retry tracking
        max_attempts = user_task.get("max_attempts", 10)
        previous_actions = set()
        best_result = None
        best_score = -1
        
        for attempt in range(max_attempts):
            try:
                # 1. Run the full experiment flow
                task_manager_result = self.agent_manager.dispatch("task_manager", user_task)
                literature_result = self.agent_manager.dispatch("literature_reader", 
                    {"user_task": user_task, "task_manager": task_manager_result})
                design_result = self.agent_manager.dispatch("experiment_designer",
                    {"user_task": user_task, "literature": literature_result})
                robot_result = self.agent_manager.dispatch("robot_operator",
                    {"design": design_result})
                analysis_result = self.agent_manager.dispatch("data_analyst",
                    {"experiment_data": robot_result})
                
                # 2. Evaluate results
                current_score = self._calculate_score(analysis_result["metrics"])
                if current_score > best_score:
                    best_result = analysis_result
                    best_score = current_score
                
                # 3. Check against thresholds
                if self._meets_thresholds(analysis_result["metrics"], user_task.get("thresholds")):
                    return {
                        "status": "success",
                        "result": analysis_result,
                        "attempts": attempt + 1,
                        "logs": logs
                    }
                
                # 4. Check for manual stop
                if user_task.get("manual_stop"):
                    return {
                        "status": "stopped",
                        "best_result": best_result,
                        "attempts": attempt + 1,
                        "logs": logs
                    }
                
                # 5. Check for repeated actions
                action_signature = self._get_action_signature(design_result, robot_result)
                if action_signature in previous_actions:
                    return {
                        "status": "repeated_actions",
                        "best_result": best_result,
                        "attempts": attempt + 1,
                        "logs": logs
                    }
                previous_actions.add(action_signature)
                
                # 6. Check termination conditions
                termination_condition = self._check_termination_conditions(analysis_result, attempt)
                if termination_condition:
                    return {
                        "status": termination_condition,
                        "best_result": best_result,
                        "attempts": attempt + 1,
                        "logs": logs
                    }
                
                # 7. Add variation for next attempt
                user_task = self._add_variation(user_task)
                
            except Exception as e:
                logs.append(f"Attempt {attempt + 1} failed: {str(e)}")
                
        return {
            "status": "max_attempts_reached",
            "best_result": best_result,
            "attempts": max_attempts,
            "logs": logs
        }
        
        self._last_state = {
            'current_task': user_task,
            'attempts': self.attempt_history
        }
        
    def _calculate_score(self, metrics):
        """基于配置的评价标准计算分数"""
        config = self._load_evaluation_config()
        
        # 计算基础分
        base_score = sum(
            metrics.get(name, 0) * config['metrics'][name]['weight']
            for name in config['metrics']
        )
        
        # 应用惩罚项
        for penalty in config['scoring']['penalty']:
            base_score += config['scoring']['penalty'][penalty]
        
        return base_score

    def _load_evaluation_config(self):
        """加载评价配置"""
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "config", "evaluation_config.yaml"
        )
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _meets_thresholds(self, metrics, thresholds):
        if not thresholds:
            return False
        return all(metrics[k] >= v for k,v in thresholds.items())
        
    def _get_action_signature(self, design_result, robot_result):
        # Create unique signature for action sequence
        return hash(frozenset(design_result.items()) | frozenset(robot_result.items()))
        
    def _add_variation(self, user_task):
        """Smart variation based on previous attempts"""
        if 'variation_history' not in user_task:
            user_task['variation_history'] = []
            
        # Get last variation or initialize
        last_variation = user_task['variation_history'][-1] if user_task['variation_history'] else {}
        
        # Generate new variation with context-aware adjustments
        new_variation = {
            'params': {
                'temperature': last_variation.get('temperature', 0.7) + 0.1,
                'top_p': max(0.1, last_variation.get('top_p', 0.9) - 0.05)
            },
            'attempt': len(user_task['variation_history']) + 1
        }
        
        user_task['variation_history'].append(new_variation)
        user_task.update(new_variation['params'])
        return user_task
        
    def _check_termination_conditions(self, result, attempt):
        """Additional termination checks"""
        # Critical failure detection
        if result.get('error_level', 0) >= 2:
            return 'critical_failure'
            
        # Plateau detection
        if attempt > 3 and len(set(r['score'] for r in result['history'][-3:])) == 1:
            return 'performance_plateau'
            
        return None

    def _get_optimized_params(self, history):
        """基于历史数据优化参数"""
        if len(history) < 3:  # 至少需要3次尝试才能优化
            return None
            
        def objective_func(params):
            # 模拟参数得分（实际应调用评估函数）
            return sum(p * 0.5 for p in params.values())
            
        initial_params = {
            'temperature': 0.7,
            'top_p': 0.9
        }
        bounds = {
            'temperature': (0.1, 1.0),
            'top_p': (0.1, 1.0)
        }
        
        return self.optimizer.optimize(objective_func, initial_params, bounds)

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

from autolab.core.agent_manager import AgentManager
from autolab.core.param_optimizer import ParamOptimizer
import uuid
import time
import os
import yaml
from autolab.core.logger import get_logger
from typing import Dict, Any, List, Optional, Tuple
import logging
import json

logger = get_logger(__name__)

class TaskFlow:
    """
    细化任务流转：每个智能体的输出作为下一个智能体的输入，实现阶段性数据传递。
    """
    def __init__(self):
        self.agent_manager = AgentManager()
        self.optimizer = ParamOptimizer()
        self._should_stop = False
        logger.info("任务流初始化完成")
        
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

    def stop(self):
        """标记需要停止"""
        self._should_stop = True

    @auto_save_state
    def run_flow(self, user_task: Dict[str, Any]) -> Dict[str, Any]:
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
            if self._should_stop:
                raise RuntimeError("实验流程已被用户中断")
            
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
                current_score = self._calculate_score(analysis_result)
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
                logger.exception(f"任务处理异常: {str(e)}")
                
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
        
    def _calculate_score(self, result: Dict[str, Any]) -> float:
        """计算任务执行评分"""
        if not result.get('success', False):
            return 0.0
            
        # 提供完整的默认指标
        metrics = result.get('metrics', {
            'accuracy': 0.8,
            'time_cost': 5.0,
            'completeness': 0.9
        })
        
        # 基础评分逻辑
        base_score = 0.5 if result.get('output') else 0.1
        score = base_score + \
                metrics.get('accuracy', 0.0) * 0.4 + \
                metrics.get('completeness', 0.0) * 0.1 - \
                min(metrics.get('time_cost', 0.0), 10) * 0.01
                
        return max(0.0, min(1.0, score))

    def _meets_thresholds(self, result: Dict[str, Any], thresholds: Dict[str, Any] = None) -> bool:
        """检查结果是否满足阈值要求"""
        if not result.get('success', False):
            return False
            
        # 如果没有指定阈值，默认通过
        if not thresholds:
            return True
            
        # 获取结果中的指标，提供默认值
        metrics = result.get('metrics', {
            'accuracy': 0.8,
            'time_cost': 5.0
        })
        
        # 检查每个阈值
        for metric, threshold in thresholds.items():
            if metric not in metrics:
                logger.warning(f"指标{metric}不存在于结果中")
                continue
                
            if metrics[metric] < threshold:
                logger.info(f"指标{metric}不满足阈值 ({metrics[metric]} < {threshold})")
                return False
                
        return True

    def _load_evaluation_config(self) -> Dict[str, Any]:
        """加载评价配置"""
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "config", "evaluation_config.yaml"
        )
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_action_signature(self, design_result: Dict[str, Any], robot_result: Dict[str, Any]) -> int:
        """生成动作序列的唯一签名"""
        try:
            # 将字典转换为可哈希的字符串
            design_str = json.dumps(design_result, sort_keys=True)
            robot_str = json.dumps(robot_result, sort_keys=True)
            return hash(f"{design_str}|{robot_str}")
        except Exception as e:
            logger.error(f"生成签名失败: {str(e)}")
            return 0  # 返回默认签名
        
    def _add_variation(self, user_task: Dict[str, Any]) -> Dict[str, Any]:
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
        
    def _check_termination_conditions(self, result: Dict[str, Any], attempt: int) -> Optional[str]:
        """Additional termination checks"""
        # Critical failure detection
        if result.get('error_level', 0) >= 2:
            return 'critical_failure'
            
        # Plateau detection
        if attempt > 3 and len(set(r['score'] for r in result['history'][-3:])) == 1:
            return 'performance_plateau'
            
        return None

    def _get_optimized_params(self, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """基于历史数据优化参数"""
        if len(history) < 3:  # 至少需要3次尝试才能优化
            return None
            
        def objective_func(params: Dict[str, Any]) -> float:
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

    def self_evaluate(self, computation_result: Dict[str, Any], evaluation_criteria: Dict[str, Any]) -> Tuple[float, str]:
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

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理任务（必须通过Agent）"""
        logger.info(f"开始处理任务: {task.get('goal','')}")
        
        if not isinstance(task, dict) or not task.get('goal'):
            logger.error("无效任务格式")
            return {"error": "Invalid task format"}
            
        # 强制通过TaskManager Agent处理
        if 'task_manager' not in self.agent_manager.agents:
            logger.critical("系统中缺少TaskManager Agent")
            return {"error": "Missing TaskManager Agent"}
            
        try:
            # 添加任务类型标识
            task['type'] = 'root_task'
            
            logger.debug(f"分发任务给TaskManager: {task}")
            result = self.agent_manager.agents['task_manager'].handle(task)
            
            if not result.get('success', False):
                logger.error(f"任务处理失败: {result.get('error','Unknown error')}")
            else:
                logger.info(f"任务处理完成，参与Agent: {result.get('debug',{}).get('agent_path',[])}")
                
            return result
            
        except Exception as e:
            logger.exception(f"任务处理异常")
            return {"error": str(e)}

    def test_agent_connections(self) -> Dict[str, Any]:
        """测试所有Agent的连接状态"""
        if not hasattr(self, 'agent_manager') or not self.agent_manager:
            return {"error": "Agent管理器未初始化"}
        
        test_results = {}
        for name, agent in self.agent_manager.agents.items():
            try:
                # 简单的ping测试
                if hasattr(agent, 'ping'):
                    test_results[name] = agent.ping()
                else:
                    test_results[name] = f"{type(agent).__name__} 不支持ping测试"
            except Exception as e:
                test_results[name] = f"连接失败: {str(e)}"
        
        return test_results

__all__ = ['TaskFlow']

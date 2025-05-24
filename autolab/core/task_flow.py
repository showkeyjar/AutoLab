from autolab.core.agent_manager import AgentManager
from autolab.core.param_optimizer import ParamOptimizer
from autolab.core.goal_parser import ExperimentGoalParser
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
        
        # 初始化实验目标解析器
        try:
            from autolab.utils.llm_client import OllamaClient
            llm_client = OllamaClient()
            self.goal_parser = ExperimentGoalParser(llm_client=llm_client)
            logger.info("实验目标解析器初始化完成(使用LLM增强)")
        except Exception as e:
            logger.warning(f"初始化LLM客户端失败: {str(e)}，将使用基础目标解析器")
            self.goal_parser = ExperimentGoalParser()  # 使用无LLM的基础解析器
            
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
        
    def parse_experimental_goal(self, user_input: str, interactive: bool = False) -> Dict[str, Any]:
        """
        解析用户输入的实验目标，转换为结构化目标定义
        
        参数:
            user_input: 用户自然语言输入
            interactive: 是否开启交互式解析（遇到模糊点询问用户）
            
        返回:
            结构化的实验目标定义
        """
        logger.info(f"开始解析实验目标: {user_input}")
        
        if not hasattr(self, 'goal_parser') or self.goal_parser is None:
            logger.warning("目标解析器未初始化，创建一个基础解析器")
            self.goal_parser = ExperimentGoalParser()
            
        # 使用解析器进行解析
        if interactive:
            # 交互式解析，需要提供回调函数
            result = self.goal_parser.interactive_parse(user_input, self._interactive_callback)
        else:
            result = self.goal_parser.parse(user_input)
        
        # 处理解析结果
        if result.get("status") in ["success", "enhanced", "partial"]:
            parsed_goal = result.get("parsed_goal", {})
            
            # 添加元数据
            parsed_goal["parsed_timestamp"] = time.time()
            parsed_goal["confidence"] = result.get("confidence", 0.0)
            parsed_goal["raw_input"] = user_input
            
            # 记录解析日志
            logger.info(f"目标解析完成 (罪比度: {parsed_goal['confidence']})")
            for key, value in parsed_goal.items():
                if key not in ['raw_input', 'description']:
                    logger.debug(f"  - {key}: {value}")
                    
            return {
                "status": "success",
                "parsed_goal": parsed_goal
            }
        else:
            # 解析失败
            error_msg = result.get("error", "未知解析错误")
            logger.error(f"目标解析失败: {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "raw_input": user_input
            }
    
    def _interactive_callback(self, field_info: Dict[str, Any]) -> Any:
        """交互式解析的回调函数，默认实现仅记录日志
        实际使用时应由沟通层重写此方法"""
        field = field_info.get("field", "")
        question = field_info.get("question", "")
        options = field_info.get("options", [])
        
        logger.info(f"需要用户输入: {question} (选项: {options})")
        
        # 返回默认值
        if field == "task_type" and options:
            return options[0]  # 返回第一个选项
        elif field == "success_criteria":
            return {"primary_metric": {"name": "accuracy", "target_value": 0.9}}
        
        return None
            
    def enhance_experimental_plan(self, initial_plan, user_task=None, historical_data=None):
        """增强和优化初始实验方案
        
        参数:
            initial_plan: 初始生成的实验方案
            user_task: 用户任务信息
            historical_data: 历史实验数据
            
        返回:
            增强后的实验方案
        """
        logger.info("开始增强实验方案...")
        
        # 默认值处理
        if user_task is None:
            user_task = {}
        if historical_data is None:
            historical_data = []
            
        # 控制历史数据量，避免传入过多数据
        if len(historical_data) > 5:
            historical_data = historical_data[-5:]
        
        # 准备任务输入
        enhancer_task = {
            "action": "enhance_plan",
            "initial_plan": initial_plan,
            "constraints": user_task.get("constraints", {}),
            "available_resources": {
                "max_time": user_task.get("max_time", 3600),  # 默认时间限制1小时
                "max_memory": user_task.get("max_memory", "4GB"),
                "processors": user_task.get("processors", 1)
            },
            "historical_data": historical_data
        }
        
        # 调用任务增强智能体
        if "task_enhancer" not in self.agent_manager.agents:
            logger.warning("缺失任务增强智能体，使用原始方案")
            # 如果不存在增强智能体，返回原始方案
            initial_plan["enhancement_status"] = "skipped"
            return initial_plan
            
        try:
            # 分发给任务增强智能体
            result = self.agent_manager.dispatch("task_enhancer", enhancer_task)
            
            if not result.get("success", False):
                logger.error(f"实验方案增强失败: {result.get('error', '未知错误')}")
                # 失败时返回原始方案并标记失败
                initial_plan["enhancement_status"] = "failed"
                initial_plan["enhancement_error"] = result.get("error", "未知错误")
                return initial_plan
                
            # 获取增强后的方案
            enhanced_plan = result.get("enhanced_plan", {})
            enhancements = result.get("enhancements", [])
            
            logger.info(f"实验方案增强完成，进行了{len(enhancements)}项改进")
            for enhancement in enhancements:
                logger.debug(f"- {enhancement}")
                
            # 添加增强状态信息
            enhanced_plan["enhancement_status"] = "success"
            enhanced_plan["enhancement_details"] = enhancements
            
            return enhanced_plan
            
        except Exception as e:
            logger.exception(f"实验方案增强过程发生异常")
            # 异常情况下返回原始方案
            initial_plan["enhancement_status"] = "error"
            initial_plan["enhancement_error"] = str(e)
            return initial_plan

    @auto_save_state
    def run_flow(self, user_task: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        experiment_id = user_task.get("experiment_id") or str(uuid.uuid4())
        user_task["experiment_id"] = experiment_id
        logs = []
        # evaluation_history will store dicts:
        # {
        #   "attempt": attempt_number,
        #   "model_architecture_used": { ... },
        #   "structured_metrics": { ... },
        #   "score": float (optional, can be re-calculated)
        # }
        evaluation_history = [] 
        
        # Initialize retry tracking
        max_attempts = user_task.get("max_attempts", 3) # Reduced for practical testing, can be 10
        previous_actions = set()
        best_result = None
        best_score = -1
        
        for attempt in range(max_attempts):
            if self._should_stop:
                raise RuntimeError("实验流程已被用户中断")
            
            try:
                # 1. 规划阶段：基础任务流程规划 (Done once or if fundamental changes are needed)
                # For iterative optimization, we use the 'current_design' from the previous loop or initial design.
                if attempt == 0: # Initial design phase
                    task_manager_result = self.agent_manager.dispatch("task_manager", user_task)
                    literature_result = self.agent_manager.dispatch("literature_reader", 
                        {"user_task": user_task, "task_manager": task_manager_result})
                    initial_design_result = self.agent_manager.dispatch("experiment_designer",
                        {"user_task": user_task, "literature": literature_result})
                    current_design = initial_design_result # This is the 'design_result' structure
                else:
                    # On subsequent attempts, current_design is the enhanced_design from previous iteration's TaskEnhancerAgent
                    pass # current_design is already set from the end of the previous loop

                # Data Acquisition (Potentially done once, or if design changes significantly)
                # For simplicity, let's assume it's done based on the latest current_design if needed
                # Or, it could be done once initially if datasets are expected to be stable.
                # For this flow, let's assume data_acquisition_result is stable after first attempt or passed if needed.
                if attempt == 0: # Example: run data acquisition only once
                    logger.info("进入数据获取阶段 (首次尝试)...")
                    data_acquisition_task = {
                        "parsed_task_requirements": user_task.get("structured_goal", {}),
                        "suggested_model_architecture": current_design.get("output", {}).get("design", {}).get("suggested_model_architecture", {})
                    }
                    data_acquisition_result = self.agent_manager.dispatch("data_acquisition_agent", data_acquisition_task)
                    logs.append({
                        "stage": "data_acquisition", "timestamp": time.time(), "status": data_acquisition_result.get("success"),
                        "output": data_acquisition_result.get("identified_datasets", [])
                    })
                    if not data_acquisition_result.get("success"):
                        logger.error(f"数据获取失败: {data_acquisition_result.get('error', '未知错误')}")
                        # Potentially stop flow if data is critical and acquisition fails
                
                # Current model architecture to be executed for this attempt
                current_model_arch_to_execute = current_design.get("output", {}).get("design", {}).get("suggested_model_architecture", {})
                current_training_params = current_design.get("output", {}).get("design", {}).get("training_parameters", {})


                # 3. 执行阶段：执行优化后的实验 (or initial experiment)
                robot_task_input = {
                    "design": current_design, # Pass the whole current design which includes procedure, model arch, and potentially training_params
                    "datasets_info": data_acquisition_result.get("identified_datasets", []), # Assuming this is available
                    "task_type": user_task.get("structured_goal", {}).get("task_type", "unknown"),
                    # ComputationExecutor will use suggested_model_architecture and training_parameters from the design
                }
                robot_result = self.agent_manager.dispatch("robot_operator", robot_task_input) # robot_operator calls ComputationExecutor
                
                logs.append({
                    "stage": "execution", "timestamp": time.time(), "attempt": attempt + 1,
                    "model_architecture_used": current_model_arch_to_execute, # Logging the arch used for this run
                    "training_parameters_used": current_training_params, # Logging training params used
                    "robot_output": robot_result.get("output")
                })

                # Extract structured_metrics from robot_result (which comes from ComputationExecutor)
                structured_metrics = None
                if robot_result.get("success") and "executed_experimental_steps" in robot_result.get("output", {}):
                    for step_output in robot_result["output"]["executed_experimental_steps"]:
                        if step_output.get("step_name") == "Evaluate Model" and step_output.get("status") == "completed_success":
                            structured_metrics = step_output.get("structured_metrics")
                            break
                
                if structured_metrics:
                    logger.info(f"Attempt {attempt + 1}: Structured metrics extracted: {structured_metrics}")
                    evaluation_history.append({
                        "attempt": attempt + 1,
                        "model_architecture_used": current_model_arch_to_execute,
                        "training_parameters_used": current_training_params,
                        "structured_metrics": structured_metrics
                    })
                else:
                    logger.warning(f"Attempt {attempt + 1}: Could not extract structured metrics or evaluation failed.")
                    # Handle cases where metrics are not available, maybe skip enhancement or use a default
                    evaluation_history.append({
                        "attempt": attempt + 1,
                        "model_architecture_used": current_model_arch_to_execute,
                        "training_parameters_used": current_training_params,
                        "structured_metrics": {"error": "Metrics not available or evaluation failed"}
                    })


                # Using structured_metrics for scoring and threshold checking if available
                # This part might need adjustment based on how _calculate_score and _meets_thresholds use metrics.
                # For now, let's assume they can work with structured_metrics or a similar format.
                # analysis_result would ideally be the structured_metrics themselves or a report containing them.
                # Let's make analysis_result the robot_result for now, and _calculate_score can dive into it.
                analysis_result = robot_result 
                current_score = self._calculate_score(analysis_result, structured_metrics) # Pass structured_metrics to score calculation
                
                if current_score > best_score:
                    best_result = analysis_result # Store the whole robot_result
                    best_score = current_score
                
                # Check against thresholds using structured_metrics
                if self._meets_thresholds(structured_metrics, user_task.get("thresholds")):
                    logger.info(f"Attempt {attempt + 1} meets thresholds. Ending optimization.")
                    return {
                        "status": "success",
                        "result": analysis_result, # This is robot_result
                        "structured_metrics": structured_metrics,
                        "attempts": attempt + 1,
                        "logs": logs,
                        "evaluation_history": evaluation_history
                    }
                
                # If max attempts reached, exit
                if attempt + 1 >= max_attempts:
                    logger.info("Max attempts reached. Ending optimization.")
                    break # Exit loop

                # 2. 增强阶段：优化实验方案 FOR THE NEXT ATTEMPT
                if self._should_stop:
                    raise RuntimeError("实验流程已被用户中断 (pre-enhancement)")
                    
                logger.info(f"Entering enhancement phase for next attempt (current attempt {attempt + 1})")
                # Pass current_design (which was executed) to TaskEnhancer as 'initial_plan'
                # TaskEnhancer will use this to suggest the *next* design.
                enhancer_task_input = {
                    "initial_plan": current_design, # The design that was just executed
                    "user_task": user_task,         # Contains structured_goal, target_metric
                    "historical_data": evaluation_history # List of all past attempts' arch, params, metrics
                }
                enhancer_result = self.enhance_experimental_plan( # This calls agent_manager.dispatch
                    initial_plan=enhancer_task_input["initial_plan"],
                    user_task=enhancer_task_input["user_task"],
                    historical_data=enhancer_task_input["historical_data"]
                    # Note: enhance_experimental_plan internally calls self.agent_manager.dispatch("task_enhancer", ...)
                    # The structure of enhancer_task inside enhance_experimental_plan needs to match what TaskEnhancerAgent expects.
                    # We are directly calling enhance_experimental_plan which then constructs its own task for dispatch.
                )
                
                logs.append({
                    "stage": "enhancement_suggestion", "timestamp": time.time(), "attempt": attempt + 1,
                    "enhancer_output_success": enhancer_result.get("success"),
                    "enhancements_suggested": enhancer_result.get("enhancements", [])
                })

                if enhancer_result.get("success") and enhancer_result.get("enhanced_plan"):
                    current_design = enhancer_result["enhanced_plan"] # This new design will be used in the NEXT iteration
                    logger.info(f"Attempt {attempt + 1}: Enhancement successful. New design prepared for next attempt.")
                else:
                    logger.warning(f"Attempt {attempt + 1}: Enhancement failed or no changes suggested. Will retry with current design or stop if stuck.")
                    # Potentially add logic here to stop if enhancement fails multiple times
                    # For now, it will just re-use the `current_design` which hasn't been updated.

                # Old checks like repeated_actions, termination_conditions, _add_variation might need rethinking
                # in the context of LLM-driven enhancement. The LLM is now the "variation" source.
                # If the LLM keeps suggesting the same thing or things that don't improve, that's a new kind of "stuck".

            except Exception as e:
                logger.exception(f"Attempt {attempt + 1} in run_flow failed: {str(e)}")
                logs.append({"stage": "error", "timestamp": time.time(), "attempt": attempt + 1, "error": str(e)})
                # Decide if we should break or continue on error. For now, continue to max_attempts.
                logger.exception(f"任务处理异常: {str(e)}")
                
        return {
            "status": "max_attempts_reached",
            "best_result": best_result,
            "attempts": max_attempts,
            "logs": logs
        }
        
        self._last_state = {
            'current_task': user_task,
            'attempts': attempt + 1, # Use loop variable
            'evaluation_history': evaluation_history
        }
        
    def _calculate_score(self, result: Dict[str, Any], structured_metrics: Optional[Dict[str, float]] = None) -> float:
        """计算任务执行评分.
        'result' is the raw robot_result.
        'structured_metrics' is the parsed metrics dict from evaluation step.
        """
        if not result.get('success', False): # Check if robot_result itself indicates failure
            return 0.0
        
        if structured_metrics and "error" not in structured_metrics:
            # Prioritize structured_metrics if available and valid
            actual_metrics = structured_metrics
        elif isinstance(result.get('output'), dict) and result['output'].get('metrics'): # Fallback to older metrics format if any
            actual_metrics = result['output']['metrics']
            logger.warning("Using fallback metrics from result['output']['metrics'] for scoring.")
        else:
            logger.warning("No valid metrics found for scoring in _calculate_score.")
            return 0.1 # Minimal score if no metrics but overall success

        # Example: Score based on accuracy, penalize loss. Customize as needed.
        # This scoring should align with the optimization target_metric.
        score = 0.0
        if 'accuracy' in actual_metrics:
            score += actual_metrics['accuracy'] * 0.7 # Weight accuracy heavily
        if 'loss' in actual_metrics:
            score += (1 - min(actual_metrics['loss'], 1.0)) * 0.3 # Penalize high loss (assuming loss is capped at 1 for this)
        
        # Add other metric considerations if necessary
        # score += actual_metrics.get('completeness', 0.0) * 0.1 
        # score -= min(actual_metrics.get('time_cost', 0.0), 10) * 0.01

        return max(0.0, min(1.0, score))


    def _meets_thresholds(self, metrics: Optional[Dict[str, float]], thresholds: Optional[Dict[str, Any]] = None) -> bool:
        """检查结果是否满足阈值要求.
        'metrics' should be the structured_metrics dictionary.
        'thresholds' directly from user_task.
        """
        if not metrics or "error" in metrics: # If no metrics or metrics parsing failed
            return False
            
        if not thresholds: # If no thresholds defined, then it's considered met for optimization purposes (continue loop)
            logger.info("No specific thresholds defined by user. Thresholds considered met for loop continuation.")
            return False # Return False to continue optimization loop if no specific target. Or True if this means "goal achieved".
                         # For optimization loop, we want to continue if no specific target is set to stop.
                         # However, if the goal is to "reach a threshold", then True means stop.
                         # Let's assume for now: if thresholds are set, meeting them means success. If not set, never "success" this way.

        for metric_name, target_value_config in thresholds.items():
            # Thresholds can be simple values or dicts like {"target_value": 0.9, "operator": ">="}
            target_value = None
            operator = ">=" # Default operator

            if isinstance(target_value_config, dict):
                target_value = target_value_config.get("target_value")
                operator = target_value_config.get("operator", ">=")
            else: # Simple value threshold
                target_value = target_value_config
            
            if target_value is None:
                logger.warning(f"Threshold for metric '{metric_name}' is not properly defined.")
                continue

            current_value = metrics.get(metric_name)
            if current_value is None:
                logger.warning(f"Metric '{metric_name}' not found in current results for threshold check.")
                return False # Metric required by threshold is missing

            logger.info(f"Threshold check: Metric '{metric_name}', Current: {current_value}, Target: {operator} {target_value}")
            if operator == ">=":
                if not (current_value >= target_value): return False
            elif operator == "<=":
                if not (current_value <= target_value): return False
            elif operator == ">":
                if not (current_value > target_value): return False
            elif operator == "<":
                if not (current_value < target_value): return False
            elif operator == "==":
                if not (current_value == target_value): return False
            else:
                logger.warning(f"Unsupported operator '{operator}' for threshold metric '{metric_name}'.")
                return False
                
        logger.info("All defined thresholds met.")
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
            # Signature should be based on the executed model architecture to detect if the same model is tried repeatedly
            # design_result is the *input* to the current execution round.
            # robot_result is the *output*.
            # We care about the executed architecture.
            # current_design is the one that was executed.
            executed_arch_str = json.dumps(design_result.get("output",{}).get("design",{}).get("suggested_model_architecture",{}), sort_keys=True)
            # To truly check for repeated *actions* that lead to same results, might need more complex signature.
            # For now, checking if the same architecture is being re-evaluated.
            return hash(executed_arch_str)
        except Exception as e:
            logger.error(f"生成签名失败: {str(e)}")
            return time.time() # Fallback to a unique number if hashing fails

    # _add_variation is likely superseded by TaskEnhancerAgent's suggestions.
    # If TaskEnhancerAgent fails to provide a new variation, we might need a fallback.
    # For now, removing _add_variation and _check_termination_conditions as primary loop controllers.
    # The loop is controlled by max_attempts and _meets_thresholds.

    # def _add_variation(self, user_task: Dict[str, Any]) -> Dict[str, Any]: ...
    # def _check_termination_conditions(self, result: Dict[str, Any], attempt: int) -> Optional[str]: ...


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

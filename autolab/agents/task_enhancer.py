from autolab.agents.base import BaseAgent  # 修正导入路径
from autolab.core.logger import get_logger
from typing import Dict, Any, List, Optional
import time

logger = get_logger(__name__)

class TaskEnhancerAgent(BaseAgent):
    """
    任务增强智能体：专门用于优化和增强实验方案
    通过分析初始计划、历史数据和可用资源，提供更高质量的实验方案
    """
    
from autolab.agents.base import BaseAgent
from autolab.core.logger import get_logger
from typing import Dict, Any, List, Optional
import time
import json # Added for JSON operations

logger = get_logger(__name__)

class TaskEnhancerAgent(BaseAgent):
    """
    任务增强智能体：专门用于优化和增强实验方案, 特别是针对模型架构和超参数。
    通过分析初始计划、历史数据和可用资源，提供更高质量的实验方案。
    """

    PROMPT_TEMPLATE_OPTIMIZE_MODEL = """
    You are an expert in machine learning model optimization.
    Your task is to analyze the provided experiment goal, current model architecture, and history of previous experimental results (model architectures used and their performance metrics).
    Suggest specific modifications to the model architecture OR changes to hyperparameters to improve the '{target_metric_name}'.

    Experiment Goal:
    {experiment_goal_json}

    Current Model Architecture:
    {current_model_architecture_json}

    Results History (previous attempts with their architectures and metrics):
    {results_history_json}

    Based on this information, provide your suggestions as a JSON object with the following keys:
    - "reasoning": (string) A brief explanation of why these changes are suggested.
    - "updated_model_architecture": (JSON object, optional) The complete new suggested model architecture if you recommend changes to the architecture. This should follow the same format as the input 'Current Model Architecture'. If no architectural changes, this can be null or omitted.
    - "updated_hyperparameters": (JSON object, optional) A dictionary of hyperparameter changes (e.g., {{"learning_rate": 0.0005, "epochs": 15}}). These are parameters typically used in the training step. If no hyperparameter changes, this can be null or omitted.

    Ensure your response is a single valid JSON object.
    If the history is empty or insufficient, you can base your suggestions on general best practices for the given task type and model architecture.
    Focus on incremental improvements.
    """
    
    def __init__(self, mock_mode=False):
        super().__init__(name="TaskEnhancerAgent", mock_mode=mock_mode) # Corrected name
        self.description = "负责增强和优化实验方案，特别是模型架构和超参数。"
        self._llm = None # Corrected attribute name to self._llm
        self._connected = False
        
    def connect(self) -> bool:
        """连接到LLM服务"""
        if self.mock_mode:
            self._connected = True
            logger.info(f"{self.name} is in mock mode.")
            return True
        try:
            from autolab.utils.llm_client import OllamaClient # Assuming path
            self._llm = OllamaClient()
            self._connected = True
            logger.info(f"{self.name} connected to LLM successfully.")
            return True
        except ImportError:
            logger.error("OllamaClient could not be imported. Ensure autolab.utils.llm_client exists.")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"{self.name} failed to connect to LLM: {str(e)}")
            self._connected = False
            # self._last_error = str(e) # If you need to store last error
            return False
            
    def _handle_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """实现BaseAgent抽象方法，实际的任务处理逻辑"""
        logger.info(f"{self.name} processing task: {task.get('action', 'enhance_plan')}") # Default action
        
        if not self._connected and not self.mock_mode:
            return self._error_response(f"{self.name} is not connected.")
            
        # This agent primarily focuses on "enhance_plan" which now means model optimization
        # The 'action' key might be less relevant if this agent has a single purpose.
        # We expect 'initial_plan', 'historical_data', and 'user_task' in the input `task`.
        
        initial_plan = task.get("initial_plan") # This is the design_result
        historical_data = task.get("historical_data", [])
        user_task_info = task.get("user_task", {}) # Contains structured_goal

        if not initial_plan or not user_task_info:
            return self._error_response("Missing 'initial_plan' or 'user_task' in the input for enhancement.")

        current_model_architecture = initial_plan.get("output", {}).get("design", {}).get("suggested_model_architecture", {})
        structured_goal = user_task_info.get("structured_goal", {})
        
        # Determine target metric
        target_metric_name = structured_goal.get("success_criteria", {}).get("primary_metric", {}).get("name", "loss") # Default to 'loss' if not specified
        if target_metric_name == "loss": # Or other metrics where lower is better
             # Could add logic here to tell LLM to minimize if necessary, but prompt is general "improve"
            pass

        if self.mock_mode:
            logger.info(f"{self.name} is in mock mode. Generating mock optimization.")
            time.sleep(1) 
            mock_llm_response = {
                "reasoning": "Mock: Increased layer size and reduced learning rate based on simulated analysis.",
                "updated_model_architecture": {**current_model_architecture, "layers": [{"name": "dense_1_mock_updated", "units": 128, "activation": "relu"}]},
                "updated_hyperparameters": {"learning_rate": 0.0008, "epochs": (current_model_architecture.get("training_parameters",{}).get("epochs",10) + 5)}
            }
        else:
            if not self._llm:
                 return self._error_response(f"{self.name} LLM client not initialized.")
            try:
                prompt = self.PROMPT_TEMPLATE_OPTIMIZE_MODEL.format(
                    current_model_architecture_json=json.dumps(current_model_architecture, indent=2),
                    experiment_goal_json=json.dumps(structured_goal, indent=2),
                    results_history_json=json.dumps(historical_data, indent=2),
                    target_metric_name=target_metric_name
                )
                # logger.debug(f"Optimization Prompt for LLM:\n{prompt}") # Uncomment for debugging
                
                raw_llm_response_str = self._llm.generate(prompt)
                logger.debug(f"Raw LLM response for optimization: {raw_llm_response_str}")
                
                # Attempt to parse the JSON response from LLM
                # LLM might return text before or after the JSON object.
                json_start_index = raw_llm_response_str.find('{')
                json_end_index = raw_llm_response_str.rfind('}')
                
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_str = raw_llm_response_str[json_start_index : json_end_index + 1]
                    llm_response_data = json.loads(json_str)
                else:
                    logger.error(f"Could not find JSON object in LLM optimization response: {raw_llm_response_str}")
                    return self._error_response("Failed to parse optimization suggestions from LLM (JSON object not found).")

            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding failed for optimization response: {e}. Response was: {raw_llm_response_str}")
                return self._error_response(f"Failed to parse JSON optimization response from LLM: {str(e)}")
            except Exception as e:
                logger.error(f"LLM optimization call failed: {str(e)}")
                return self._error_response(f"LLM call for optimization failed: {str(e)}")
        
        # Construct the enhanced_plan based on LLM response
        enhanced_plan = initial_plan.copy() # Start with the original plan
        
        # Ensure the path to suggested_model_architecture exists
        if "output" not in enhanced_plan: enhanced_plan["output"] = {}
        if "design" not in enhanced_plan["output"]: enhanced_plan["output"]["design"] = {}

        updated_arch = llm_response_data.get("updated_model_architecture")
        if updated_arch and isinstance(updated_arch, dict) and updated_arch: # Check if not null and is a non-empty dict
            enhanced_plan["output"]["design"]["suggested_model_architecture"] = updated_arch
            logger.info("Applied updated model architecture from LLM.")
        else:
            logger.info("No updated model architecture provided by LLM, or it was empty/null.")

        updated_hyperparams = llm_response_data.get("updated_hyperparameters")
        if updated_hyperparams and isinstance(updated_hyperparams, dict) and updated_hyperparams:
            # These hyperparameters are typically for training.
            # We need to decide where to store them.
            # Option 1: Directly in the design object, maybe under a specific key.
            # Option 2: As a separate key in the enhanced_plan output.
            # Let's put them in the design object for now, similar to how training_params might be handled by ComputationExecutor
            if "training_parameters" not in enhanced_plan["output"]["design"]:
                 enhanced_plan["output"]["design"]["training_parameters"] = {}
            enhanced_plan["output"]["design"]["training_parameters"].update(updated_hyperparams)
            logger.info(f"Applied updated hyperparameters from LLM: {updated_hyperparams}")
        else:
            logger.info("No updated hyperparameters provided by LLM, or it was empty/null.")
            
        enhancement_details = [llm_response_data.get("reasoning", "LLM provided no specific reasoning.")]

        return {
            "success": True,
            "enhanced_plan": enhanced_plan, # This is the modified design_result
            "enhancements": enhancement_details, # Reasoning from LLM
            "original_plan": initial_plan # For comparison if needed
        }

    # Removed _generate_mock_enhancement, _build_enhancement_prompt, _parse_llm_response
    # as they are replaced by the new optimization-focused logic and prompt.
    # The old _get_enhancement_details might be too generic now.
        
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generates a standardized error response."""
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message,
            "enhanced_plan": None,
            "enhancements": []
        }

    # Ensure handle method calls _handle_impl as per BaseAgent pattern
    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Public method to handle task enhancement requests."""
        return self._handle_impl(task)

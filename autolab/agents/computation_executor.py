"""计算执行智能体"""
from typing import Dict, Any, Optional, List
import json
import subprocess # Added for code execution
import tempfile # Added for temporary script files
import os # Added for tempfile an dpath operations
from autolab.core.logger import get_logger
from .base import BaseAgent
# Assuming OllamaClient will be in autolab.utils.llm_client
# from autolab.utils.llm_client import OllamaClient 

logger = get_logger(__name__)

class ComputationExecutorAgent(BaseAgent):
    """执行计算任务, 包括代码生成和未来可能的执行"""

    PROMPT_TEMPLATE_CODE_GENERATION = """
    Generate a Python Keras script snippet for the following experimental sub-step: "{step_name}".

    Task Type: {task_type}
    Model Architecture:
    {model_architecture_json}

    Dataset Information (use the first dataset if multiple are provided):
    {dataset_info_json}

    Training Parameters:
    {training_parameters_json}

    Evaluation Metrics (if applicable for the step):
    {evaluation_metrics_list}

    Instructions:
    - Focus *only* on the code required for the step: "{step_name}".
    - Ensure the generated code is a valid Keras (TensorFlow) Python snippet.
    - If the step is "Define Model", generate the Keras model definition and compile it using the optimizer and loss from Model Architecture.
    - If the step is "Data Loading", use the 'usage_instructions' or 'identifier_or_url' from Dataset Information to suggest how to load data. If it's a Hugging Face dataset, use the `datasets` library.
    - If the step is "Train Model", assume the model is already defined and compiled, and data ('x_train', 'y_train', 'x_val', 'y_val') is loaded. Use the provided Training Parameters.
    - If the step is "Evaluate Model":
        - Assume the model is trained and data ('x_test', 'y_test') is loaded.
        - Use the provided Evaluation Metrics (e.g., {evaluation_metrics_list}).
        - After evaluation, print a dictionary of metrics (e.g., `{'accuracy': eval_accuracy, 'loss': eval_loss}`) as a JSON string to stdout.
        - Example: `import json; print(json.dumps({'accuracy': 0.95, 'loss': 0.12}))`. Ensure `json` is imported in the script.
    - Return *only the raw Python code block*. Do not include any explanations, markdown formatting, or other text outside the code.

    Code for "{step_name}":
    """
    
    def __init__(self, mock_mode: bool = False):
        super().__init__(name="ComputationExecutor", mock_mode=mock_mode)
        self._llm = None
        self._connected = False
    
    def connect(self) -> bool:
        """连接到LLM服务 (用于代码生成)"""
        if self.mock_mode:
            self._connected = True
            logger.info(f"{self.name} is in mock mode.")
            return True
            
        try:
            from autolab.utils.llm_client import OllamaClient # Assuming path
            self._llm = OllamaClient()
            self._connected = True
            logger.info(f"{self.name} connected to LLM successfully for code generation.")
            return True
        except ImportError:
            logger.error("OllamaClient could not be imported. Ensure autolab.utils.llm_client exists.")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"{self.name} failed to connect to LLM: {str(e)}")
            self._connected = False
            return False
    
    def _handle_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """实际计算任务实现 - Currently focused on code generation"""
        if not self._connected and not self.mock_mode: # Allow mock_mode to run without LLM
            return {
                "success": False,
                "error": f"{self.name} is not connected. Call connect() first.",
                "generated_steps": [],
            }

        procedure = task.get("procedure", [])
        model_architecture = task.get("model_architecture", {})
        datasets_info = task.get("datasets_info", []) # List of datasets
        task_type = task.get("task_type", "unknown")
        
        # For now, use the first dataset if multiple are identified
        dataset_to_use = datasets_info[0] if datasets_info else {}

        # Default training parameters (can be made more dynamic later)
        default_training_params = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": model_architecture.get("optimizer_config", {}).get("learning_rate", 0.001) # Get from model arch if possible
        }
        
        # Evaluation metrics from model architecture or a default
        evaluation_metrics = model_architecture.get("metrics", ["accuracy"])
        if isinstance(evaluation_metrics, str): # Ensure it's a list
            evaluation_metrics = [evaluation_metrics]


        generated_steps_output = []

        # Mapping from Chinese step names (from ExperimentDesigner) to a more generic key / code-gen relevant name
        # This might need to be more robust or configured externally
        STEP_MAPPING = {
            "数据加载": "Data Loading", # Or "Load Data"
            "模型定义": "Define Model", # Or "Model Definition"
            "模型训练": "Train Model", # Or "Model Training"
            "模型评估": "Evaluate Model", # Or "Model Evaluation"
            "结果分析与可视化": "Results Analysis and Visualization", # May need code
            # Steps that might not need code generation by default:
            "文献回顾": "Literature Review",
            "研究假设": "Formulate Hypothesis", # Already part of design
            "实验步骤": "Outline Procedure", # This is the input 'procedure' itself
            "所需材料": "List Materials", # Already part of design
        }

        if self.mock_mode:
            logger.info(f"{self.name} is in mock mode. Generating mock code snippets.")
            for step_description in procedure:
                step_key = next((key for cn, key in STEP_MAPPING.items() if cn in step_description), None)
                if step_key and step_key in ["Data Loading", "Define Model", "Train Model", "Evaluate Model"]:
                    generated_steps_output.append({
                        "step_name": step_key,
                        "original_description": step_description,
                        "generated_code": f"# Mock code for {step_key}\nprint('Executing {step_key} for {task_type} task...')",
                        "status": "pending_execution",
                        "stdout": "",
                        "stderr": "",
                        "return_code": None,
                        "structured_metrics": None
                    })
                else:
                    generated_steps_output.append({
                        "step_name": step_key or step_description,
                        "original_description": step_description,
                        "generated_code": None,
                        "status": "not_applicable",
                        "stdout": "",
                        "stderr": "",
                        "return_code": None,
                        "structured_metrics": None
                    })
            
            # Simulate execution for mock mode
            for step_output in generated_steps_output:
                if step_output["status"] == "pending_execution":
                    step_output["status"] = "completed_success"
                    step_output["stdout"] = f"Mock execution of {step_output['step_name']} successful."
                    if step_output["step_name"] == "Evaluate Model":
                        mock_metrics = {"accuracy": 0.75, "loss": 0.5}
                        step_output["stdout"] += f"\n{json.dumps(mock_metrics)}"
                        step_output["structured_metrics"] = mock_metrics
                    step_output["stderr"] = ""
                    step_output["return_code"] = 0
                    logger.info(f"Mock execution for step: {step_output['step_name']} completed.")
            
            return {
                "success": True,
                "output": {"executed_experimental_steps": generated_steps_output},
                "debug": {"agent_path": [self.name], "task_received": task}
            }

        # Code Generation Phase (as before)
        generated_code_steps = []
        for step_description in procedure:
            current_step_name_key = None
            for chinese_name, generic_name in STEP_MAPPING.items():
                if chinese_name in step_description:
                    current_step_name_key = generic_name
                    break
            
            step_details = {
                "step_name": current_step_name_key or step_description,
                "original_description": step_description,
                "generated_code": None,
                "status": "not_applicable",
                "stdout": "",
                "stderr": "",
                "return_code": None,
                "structured_metrics": None # Initialize structured_metrics
            }

            if current_step_name_key and current_step_name_key in ["Data Loading", "Define Model", "Train Model", "Evaluate Model", "Results Analysis and Visualization"]:
                prompt = self.PROMPT_TEMPLATE_CODE_GENERATION.format(
                    step_name=current_step_name_key,
                    task_type=task_type,
                    model_architecture_json=json.dumps(model_architecture, indent=2),
                    dataset_info_json=json.dumps(dataset_to_use, indent=2),
                    training_parameters_json=json.dumps(default_training_params, indent=2),
                    evaluation_metrics_list=str(evaluation_metrics)
                )
                try:
                    logger.info(f"Generating code for step: {current_step_name_key} (Original: {step_description})")
                    generated_code_raw = self._llm.generate(prompt)
                    generated_code = generated_code_raw.strip()
                    if generated_code.startswith("```python"):
                        generated_code = generated_code[len("```python"):].strip()
                    if generated_code.startswith("```"):
                        generated_code = generated_code[len("```"):].strip()
                    if generated_code.endswith("```"):
                        generated_code = generated_code[:-len("```")].strip()
                    
                    step_details["generated_code"] = generated_code.strip()
                    step_details["status"] = "pending_execution"
                except Exception as e:
                    logger.error(f"Failed to generate code for step '{current_step_name_key}': {str(e)}")
                    step_details["generated_code"] = f"# Error generating code: {str(e)}"
                    step_details["status"] = "generation_failed"
            
            generated_code_steps.append(step_details)

        # Code Execution Phase
        executed_steps_output = []
        for step_data in generated_code_steps:
            if step_data["status"] == "pending_execution" and step_data["generated_code"]:
                logger.info(f"Executing code for step: {step_data['step_name']}")
                try:
                    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".py", encoding='utf-8') as tmp_script:
                        tmp_script.write(step_data["generated_code"])
                        tmp_file_path = tmp_script.name
                    
                    # Ensure Keras and other relevant libraries are in PYTHONPATH or globally available
                    # The execution environment for subprocess might be different.
                    process = subprocess.run(
                        ['python', tmp_file_path],
                        capture_output=True,
                        text=True,
                        timeout=300, # 5-minute timeout per script
                        check=False # Do not raise exception for non-zero exit codes
                    )
                    
                    step_data["stdout"] = process.stdout
                    step_data["stderr"] = process.stderr
                    step_data["return_code"] = process.returncode
                    
                    if process.returncode == 0:
                        step_data["status"] = "completed_success"
                        logger.info(f"Step '{step_data['step_name']}' executed successfully.")
                        # Attempt to parse structured metrics for "Evaluate Model" step
                        if step_data["step_name"] == "Evaluate Model":
                            try:
                                # Find the last non-empty line that looks like a JSON dict
                                json_line = None
                                for line in reversed(process.stdout.strip().split('\n')):
                                    line = line.strip()
                                    if line.startswith('{') and line.endswith('}'):
                                        json_line = line
                                        break
                                if json_line:
                                    metrics = json.loads(json_line)
                                    step_data["structured_metrics"] = metrics
                                    logger.info(f"Parsed structured metrics for '{step_data['step_name']}': {metrics}")
                                else:
                                    logger.warning(f"No JSON dictionary found in stdout for '{step_data['step_name']}'.")
                                    step_data["structured_metrics"] = {"error": "No JSON output found"}
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse JSON metrics from stdout for '{step_data['step_name']}': {e}")
                                step_data["structured_metrics"] = {"error": f"JSON parsing failed: {e}"}
                    else:
                        step_data["status"] = "completed_failure"
                        logger.error(f"Step '{step_data['step_name']}' failed. RC: {process.returncode}. Stderr: {process.stderr[:500]}")

                except subprocess.TimeoutExpired:
                    logger.error(f"Execution of step '{step_data['step_name']}' timed out.")
                    step_data["status"] = "failed_timeout"
                    step_data["stderr"] = "Execution timed out after 300 seconds."
                    step_data["structured_metrics"] = None # Ensure it's reset/set
                except Exception as e:
                    logger.error(f"Error during execution of step '{step_data['step_name']}': {str(e)}")
                    step_data["status"] = "execution_error"
                    step_data["stderr"] = str(e)
                    step_data["structured_metrics"] = None # Ensure it's reset/set
                finally:
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
            
            executed_steps_output.append(step_data)

        return {
            "success": True, # Overall success of the agent's operation (code gen and execution attempt)
            "output": {"executed_experimental_steps": executed_steps_output}, # This output now contains structured_metrics for relevant steps
            "debug": {"agent_path": [self.name], "task_received": task}
        }
    
    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理计算请求 (currently code generation)"""
        return self._handle_impl(task)

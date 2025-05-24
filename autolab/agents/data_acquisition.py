"""Data Acquisition Agent"""
from typing import Dict, Any, List, Optional
import json
from autolab.core.logger import get_logger
from .base import BaseAgent
# Assuming OllamaClient will be in autolab.utils.llm_client
# from autolab.utils.llm_client import OllamaClient 

logger = get_logger(__name__)

class DataAcquisitionAgent(BaseAgent):
    """Identifies and suggests public datasets for a given task."""

    PROMPT_TEMPLATE = """
    Based on the following task details, please identify suitable public datasets:

    Task Type: {task_type}
    Domain: {domain}
    Goal Description: {goal_description}
    Suggested Model Architecture Type: {model_architecture_type}

    Please identify a list of public datasets. For each dataset, provide the following information:
    - name: The official name of the dataset.
    - description: A brief summary of what the dataset contains and its purpose.
    - source_type: The origin or platform where the dataset can be found (e.g., "Hugging Face Datasets", "Kaggle", "UCI ML Repository", "Direct URL").
    - identifier_or_url: The specific ID, path, or URL to access the dataset (e.g., Hugging Face dataset ID like 'glue', 'cifar10'; Kaggle competition name or dataset URL; direct download link).
    - usage_instructions: Brief instructions, code snippets, or library calls on how to load or use the dataset (e.g., "from datasets import load_dataset; load_dataset('dataset_name')", "pd.read_csv('url_to_dataset.csv')").

    Provide your response as a JSON list of objects, where each object represents a dataset.
    Example:
    [
      {
        "name": "GLUE Benchmark",
        "description": "A collection of NLU tasks.",
        "source_type": "Hugging Face Datasets",
        "identifier_or_url": "glue",
        "usage_instructions": "from datasets import load_dataset; data = load_dataset('glue', 'mrpc')"
      },
      {
        "name": "Titanic: Machine Learning from Disaster",
        "description": "Predict survival on the Titanic.",
        "source_type": "Kaggle",
        "identifier_or_url": "titanic",
        "usage_instructions": "Download CSV from Kaggle competition page and use pandas to load."
      }
    ]
    """

    def __init__(self, mock_mode: bool = False):
        super().__init__(name="DataAcquisitionAgent", mock_mode=mock_mode)
        self._llm = None
        self._connected = False

    def connect(self) -> bool:
        """Connects to the LLM service."""
        if self.mock_mode:
            self._connected = True
            logger.info(f"{self.name} is in mock mode.")
            return True
        
        try:
            # This will be uncommented once OllamaClient is confirmed to be in this path
            from autolab.utils.llm_client import OllamaClient
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
            return False

    def _handle_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the data acquisition task implementation."""
        if not self._connected:
            return {
                "success": False,
                "error": f"{self.name} is not connected. Call connect() first.",
                "identified_datasets": [],
            }

        parsed_task_reqs = task.get("parsed_task_requirements", {})
        suggested_model_arch = task.get("suggested_model_architecture", {})

        prompt = self.PROMPT_TEMPLATE.format(
            task_type=parsed_task_reqs.get("task_type", "Not specified"),
            domain=parsed_task_reqs.get("domain", "Not specified"),
            goal_description=parsed_task_reqs.get("goal", "Not specified"), # 'goal' from ExperimentGoalParser is the description
            model_architecture_type=suggested_model_arch.get("architecture_type", "Not specified") # Assuming 'architecture_type' is a key in suggested_model_arch
        )

        if self.mock_mode:
            logger.info(f"{self.name} is in mock mode. Returning mock data.")
            mock_datasets = [
                {
                    "name": "Mock MNIST",
                    "description": "Mock handwritten digit images.",
                    "source_type": "Mock Source",
                    "identifier_or_url": "mock-mnist",
                    "usage_instructions": "load_mock_data('mnist')"
                }
            ]
            return {
                "success": True,
                "identified_datasets": mock_datasets,
                "debug": {"agent_path": [self.name], "prompt": prompt}
            }

        try:
            response_text = self._llm.generate(prompt)
            # Attempt to parse the JSON response
            # The LLM might return text before or after the JSON list.
            # We try to find the start of the list '[' and the end ']'
            json_start_index = response_text.find('[')
            json_end_index = response_text.rfind(']')
            
            if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                json_str = response_text[json_start_index : json_end_index + 1]
                identified_datasets = json.loads(json_str)
            else:
                logger.error(f"Could not find JSON list in LLM response: {response_text}")
                return {
                    "success": False,
                    "error": "Failed to parse datasets from LLM response (JSON list not found).",
                    "identified_datasets": [],
                    "debug": {"agent_path": [self.name], "prompt": prompt, "raw_response": response_text}
                }

            return {
                "success": True,
                "identified_datasets": identified_datasets,
                "debug": {"agent_path": [self.name], "prompt": prompt, "raw_response": response_text}
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}. Response was: {response_text}")
            return {
                "success": False,
                "error": f"Failed to parse JSON response from LLM: {str(e)}",
                "identified_datasets": [],
                "debug": {"agent_path": [self.name], "prompt": prompt, "raw_response": response_text}
            }
        except Exception as e:
            logger.error(f"Data acquisition failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "identified_datasets": [],
                "debug": {"agent_path": [self.name], "prompt": prompt}
            }

    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Public method to handle data acquisition requests."""
        return self._handle_impl(task)

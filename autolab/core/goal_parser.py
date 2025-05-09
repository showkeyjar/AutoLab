from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from autolab.core.logger import get_logger

logger = get_logger(__name__)

class ExperimentGoalParser:
    """
    实验目标解析器：将自然语言实验目标转换为结构化格式
    借鉴Kaggle任务的结构化特性，提高实验明确性和可执行性
    """
    
    def __init__(self, llm_client=None):
        """
        初始化解析器
        
        参数:
            llm_client: 可选的LLM客户端，用于自然语言处理
        """
        self.llm_client = llm_client
        self.task_types = [
            "classification", "regression", "optimization", 
            "synthesis", "measurement", "simulation",
            "data_analysis", "benchmark", "custom"
        ]
        
        # 常见领域和任务映射
        self.domain_tasks = {
            "机器学习": ["classification", "regression", "clustering", "ranking"],
            "化学实验": ["synthesis", "analysis", "purification"],
            "物理测量": ["measurement", "characterization", "calibration"],
            "生物实验": ["assay", "sequencing", "cultivation"],
            "材料科学": ["synthesis", "characterization", "testing"]
        }
        
        # 常见评估指标
        self.common_metrics = {
            "accuracy": {"type": "higher_better", "range": [0, 1]},
            "precision": {"type": "higher_better", "range": [0, 1]},
            "recall": {"type": "higher_better", "range": [0, 1]},
            "f1_score": {"type": "higher_better", "range": [0, 1]},
            "time_cost": {"type": "lower_better", "unit": "seconds"},
            "memory_usage": {"type": "lower_better", "unit": "MB"},
            "yield": {"type": "higher_better", "unit": "%", "range": [0, 100]},
            "purity": {"type": "higher_better", "unit": "%", "range": [0, 100]}
        }
        
    def parse(self, natural_language_input: str) -> Dict[str, Any]:
        """
        解析自然语言输入为结构化实验目标
        
        参数:
            natural_language_input: 用户输入的自然语言实验目标描述
            
        返回:
            结构化的实验目标字典
        """
        logger.info(f"开始解析实验目标: {natural_language_input}")
        
        try:
            # 如果有LLM客户端，使用智能解析
            if self.llm_client:
                return self._parse_with_llm(natural_language_input)
            else:
                # 否则使用规则解析
                return self._parse_with_rules(natural_language_input)
                
        except Exception as e:
            logger.exception(f"目标解析失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "raw_input": natural_language_input
            }
            
    def _parse_with_llm(self, text: str) -> Dict[str, Any]:
        """使用LLM进行智能解析"""
        # 构建提示词
        prompt = self._build_parsing_prompt(text)
        
        try:
            # 调用LLM
            response = self.llm_client.generate(prompt)
            
            # 尝试解析JSON响应
            if isinstance(response, str):
                # 查找JSON部分
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    parsed_goal = json.loads(json_str)
                else:
                    raise ValueError("LLM响应中未找到有效的JSON结构")
            elif isinstance(response, dict):
                parsed_goal = response
            else:
                raise TypeError(f"LLM响应类型无法解析: {type(response)}")
                
            # 验证和补全结构
            parsed_goal = self._validate_and_complete(parsed_goal)
            
            return {
                "status": "success",
                "parsed_goal": parsed_goal,
                "confidence": self._calculate_confidence(parsed_goal),
                "raw_input": text
            }
            
        except Exception as e:
            logger.error(f"LLM解析失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "raw_input": text
            }
            
    def _parse_with_rules(self, text: str) -> Dict[str, Any]:
        """使用规则进行基础解析"""
        # 简单的规则解析实现
        goal_lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # 初始化结构
        structured_goal = {
            "goal": goal_lines[0] if goal_lines else text,
            "task_type": self._guess_task_type(text),
            "success_criteria": {},
            "resources": {}
        }
        
        # 尝试提取关键信息
        for line in goal_lines[1:] if len(goal_lines) > 1 else []:
            if "准确率" in line or "精度" in line or "accuracy" in line.lower():
                structured_goal["success_criteria"]["accuracy"] = 0.8  # 默认值
            
            if "时间" in line or "time" in line.lower():
                structured_goal["success_criteria"]["time_cost"] = {"max_value": 300}  # 默认5分钟
                
            if "资源" in line or "设备" in line or "instrument" in line.lower():
                structured_goal["resources"]["specified"] = True
        
        # 验证结构
        structured_goal = self._validate_and_complete(structured_goal)
        
        return {
            "status": "partial",
            "parsed_goal": structured_goal,
            "confidence": self._calculate_confidence(structured_goal),
            "raw_input": text,
            "message": "基于规则的简单解析，可能需要用户补充更多信息"
        }
    
    def _build_parsing_prompt(self, text: str) -> str:
        """构建用于LLM的解析提示词"""
        return f"""
        作为实验目标解析专家，请分析以下自然语言描述，提取关键信息并转换为结构化JSON格式。

        用户输入: "{text}"

        请将其转换为以下结构:
        {{
            "goal": "明确的单一句实验目标",
            "task_type": "从以下选择一个: {', '.join(self.task_types)}",
            "domain": "实验所属领域",
            "description": "详细描述",
            "success_criteria": {{
                "primary_metric": {{"name": "主要评估指标名称", "target_value": 目标值, "unit": "单位(可选)"}},
                "secondary_metrics": [
                    {{"name": "次要指标1", "target_value": 目标值, "unit": "单位(可选)"}},
                    ...
                ]
            }},
            "resources": {{
                "required_instruments": ["设备1", "设备2", ...],
                "time_limit": 预计耗时(秒),
                "computational_requirements": "计算资源需求(可选)"
            }},
            "constraints": ["约束条件1", "约束条件2", ...],
            "expected_output": "期望输出的描述"
        }}

        只返回JSON结构，不需要其他说明。如果某些字段无法从输入中确定，可以使用合理的默认值并标记为"auto_generated": true。
        """
        
    def _validate_and_complete(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """验证结构并用合理默认值补全缺失字段"""
        # 复制以避免修改原始字典
        validated = goal.copy() if goal else {}
        
        # 确保基本字段存在
        if "goal" not in validated or not validated["goal"]:
            validated["goal"] = "未指定目标"
            validated["auto_generated_goal"] = True
            
        if "task_type" not in validated or not validated["task_type"]:
            validated["task_type"] = "custom"
            validated["auto_generated_task_type"] = True
            
        # 确保success_criteria存在且有至少一个指标
        if "success_criteria" not in validated or not validated["success_criteria"]:
            validated["success_criteria"] = {
                "primary_metric": {"name": "completion", "target_value": 1.0}
            }
            validated["auto_generated_criteria"] = True
        elif "primary_metric" not in validated["success_criteria"]:
            # 尝试从secondary_metrics中提升一个为primary
            if "secondary_metrics" in validated["success_criteria"] and validated["success_criteria"]["secondary_metrics"]:
                validated["success_criteria"]["primary_metric"] = validated["success_criteria"]["secondary_metrics"][0]
                validated["success_criteria"]["secondary_metrics"] = validated["success_criteria"]["secondary_metrics"][1:]
            else:
                validated["success_criteria"]["primary_metric"] = {"name": "completion", "target_value": 1.0}
                validated["auto_generated_primary_metric"] = True
                
        # 确保resources存在
        if "resources" not in validated or not validated["resources"]:
            validated["resources"] = {
                "time_limit": 3600,  # 默认1小时
                "required_instruments": []
            }
            validated["auto_generated_resources"] = True
            
        return validated
        
    def _calculate_confidence(self, parsed_goal: Dict[str, Any]) -> float:
        """计算解析结果的置信度"""
        # 简单实现：根据自动生成字段的数量降低置信度
        auto_generated_count = sum(1 for k in parsed_goal if k.startswith("auto_generated"))
        
        base_confidence = 0.9  # 基础置信度
        penalty_per_auto = 0.1  # 每个自动生成字段的惩罚
        
        confidence = max(0.1, base_confidence - auto_generated_count * penalty_per_auto)
        return round(confidence, 2)
        
    def _guess_task_type(self, text: str) -> str:
        """基于文本内容猜测任务类型"""
        text_lower = text.lower()
        
        # 简单的关键词匹配
        if any(word in text_lower for word in ["分类", "类别", "识别", "classify", "classification"]):
            return "classification"
        elif any(word in text_lower for word in ["回归", "预测", "估计", "predict", "regression"]):
            return "regression"
        elif any(word in text_lower for word in ["优化", "最大化", "最小化", "optimize", "optimization"]):
            return "optimization"
        elif any(word in text_lower for word in ["合成", "制备", "反应", "synthesis"]):
            return "synthesis"
        elif any(word in text_lower for word in ["测量", "检测", "测试", "measure", "measurement"]):
            return "measurement"
        elif any(word in text_lower for word in ["模拟", "仿真", "simulate", "simulation"]):
            return "simulation"
        elif any(word in text_lower for word in ["分析", "analyze", "analysis"]):
            return "data_analysis"
        elif any(word in text_lower for word in ["基准", "benchmark"]):
            return "benchmark"
            
        # 默认为自定义类型
        return "custom"
        
    def interactive_parse(self, initial_input: str, callback_fn=None) -> Dict[str, Any]:
        """
        交互式解析过程，当信息不足时请求用户补充
        
        参数:
            initial_input: 初始用户输入
            callback_fn: 交互回调函数，用于获取用户补充信息的函数
            
        返回:
            完整的结构化目标
        """
        # 首先进行初步解析
        result = self.parse(initial_input)
        parsed_goal = result.get("parsed_goal", {})
        
        # 如果没有回调函数或已经解析成功，直接返回
        if not callback_fn or result.get("status") == "success":
            return result
            
        # 检查需要用户补充的字段
        missing_fields = []
        
        # 检查主要评估指标
        if parsed_goal.get("auto_generated_criteria", False):
            missing_fields.append({
                "field": "success_criteria",
                "question": "请指定主要评估指标及目标值",
                "options": list(self.common_metrics.keys())
            })
            
        # 检查任务类型
        if parsed_goal.get("auto_generated_task_type", False):
            missing_fields.append({
                "field": "task_type",
                "question": "请选择实验类型",
                "options": self.task_types
            })
            
        # 如果有需要补充的字段，调用回调函数获取信息
        if missing_fields:
            supplements = {}
            for field_info in missing_fields:
                field_value = callback_fn(field_info)
                if field_value:
                    supplements[field_info["field"]] = field_value
                    
            # 更新解析结果
            for field, value in supplements.items():
                if field == "success_criteria" and isinstance(value, dict):
                    parsed_goal["success_criteria"] = value
                    parsed_goal.pop("auto_generated_criteria", None)
                elif field == "task_type" and isinstance(value, str):
                    parsed_goal["task_type"] = value
                    parsed_goal.pop("auto_generated_task_type", None)
                else:
                    parsed_goal[field] = value
                    
            # 重新计算置信度
            confidence = self._calculate_confidence(parsed_goal)
            
            return {
                "status": "enhanced",
                "parsed_goal": parsed_goal,
                "confidence": confidence,
                "raw_input": initial_input
            }
            
        return result


# 导出ExperimentGoalParser类
__all__ = ["ExperimentGoalParser"]

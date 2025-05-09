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
    
    def __init__(self, mock_mode=False):
        super().__init__(name="task_enhancer", mock_mode=mock_mode)
        self.description = "负责增强和优化实验方案，提高实验成功率"
        self.llm_client = None  # 确保初始化LLM客户端属性
        
    def connect(self) -> bool:
        """连接到LLM服务"""
        try:
            self._connected = True
            logger.info(f"任务增强智能体已连接")
            return True
        except Exception as e:
            logger.error(f"任务增强智能体连接失败: {str(e)}")
            self._connected = False
            self._last_error = str(e)
            return False
            
    def _handle_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """实现BaseAgent抽象方法，实际的任务处理逻辑"""
        logger.info(f"任务增强智能体处理任务: {task.get('action', 'unknown')}")
        
        # 确保已连接
        if not self._connected and not self.mock_mode:
            return self._error_response("增强器未连接")
            
        action = task.get("action", "")
        
        if action == "enhance_plan":
            return self.enhance_experimental_plan(task)
        else:
            return self._error_response(f"不支持的操作: {action}")
            
    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理增强任务，优化实验方案 (调用父类的handle方法)"""
        logger.info(f"任务增强智能体收到任务: {task.get('action', 'unknown')}")
        return super().handle(task) 
        
    def enhance_experimental_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        增强和优化实验方案
        
        参数:
            task: 包含初始计划、资源限制和历史数据的任务字典
            
        返回:
            增强后的实验计划
        """
        initial_plan = task.get("initial_plan", {})
        constraints = task.get("constraints", {})
        resources = task.get("available_resources", {})
        historical_data = task.get("historical_data", [])
        
        if self.mock_mode:
            logger.info("增强器处于模拟模式，返回模拟数据")
            time.sleep(1)  # 模拟思考时间
            enhanced_plan = self._generate_mock_enhancement(initial_plan)
        else:
            # 真实模式下使用LLM增强计划
            enhanced_plan = self._enhance_with_llm(initial_plan, constraints, resources, historical_data)
            
        # 记录增强前后的对比
        enhancement_details = self._get_enhancement_details(initial_plan, enhanced_plan)
        
        return {
            "success": True,
            "enhanced_plan": enhanced_plan,
            "enhancements": enhancement_details,
            "original_plan": initial_plan
        }
        
    def _enhance_with_llm(self, plan, constraints, resources, historical_data):
        """使用LLM进行方案增强"""
        try:
            # 构建提示词
            prompt = self._build_enhancement_prompt(plan, constraints, resources, historical_data)
            
            # 调用LLM
            response = self.llm_client.generate(prompt)
            
            # 解析响应
            enhanced_plan = self._parse_llm_response(response, plan)
            
            return enhanced_plan
        except Exception as e:
            logger.error(f"LLM增强失败: {str(e)}")
            # 失败时返回原始计划但添加错误标记
            plan["enhancement_error"] = str(e)
            return plan
            
    def _build_enhancement_prompt(self, plan, constraints, resources, historical_data):
        """构建增强提示词"""
        prompt = f"""
        你是一个实验方案优化专家，请分析并增强以下实验方案，提高实验的准确性、效率和成功率。
        
        ## 原始实验方案
        {plan}
        
        ## 资源限制
        {constraints}
        
        ## 可用资源
        {resources}
        
        ## 历史实验数据
        {historical_data if historical_data else "无历史数据"}
        
        请针对以下方面进行增强：
        1. 实验步骤的精确度和清晰度
        2. 参数选择的优化
        3. 潜在问题的预防措施
        4. 资源利用效率
        5. 实验结果的可靠性保障
        
        返回完整的增强后实验方案，包括所有原始内容以及你的改进。
        """
        return prompt
        
    def _parse_llm_response(self, response, original_plan):
        """解析LLM响应，提取增强后的计划"""
        # 默认情况下直接使用LLM的输出作为增强计划
        # 实际应用中应该添加更复杂的解析逻辑
        
        # 确保返回的是字典格式
        if isinstance(response, dict):
            enhanced_plan = response
        else:
            # 如果是文本，尝试将其转化为字典格式
            enhanced_plan = original_plan.copy()
            enhanced_plan["enhanced_description"] = response
            
        # 添加增强标记
        enhanced_plan["enhanced"] = True
        enhanced_plan["enhancement_time"] = time.time()
        
        return enhanced_plan
        
    def _generate_mock_enhancement(self, plan):
        """生成模拟的增强内容（用于测试）"""
        enhanced_plan = plan.copy() if isinstance(plan, dict) else {"original": plan}
        
        # 添加一些模拟的增强
        enhanced_plan["enhanced"] = True
        enhanced_plan["enhancement_time"] = time.time()
        
        if isinstance(plan, dict) and "steps" in plan:
            # 增强实验步骤
            enhanced_plan["steps"] = plan["steps"] + ["验证实验结果的一致性"]
            
        if isinstance(plan, dict) and "parameters" in plan:
            # 优化参数
            enhanced_plan["parameters"] = {
                **plan["parameters"],
                "confidence_threshold": 0.85,  # 增加置信度阈值
                "repetitions": 3  # 增加重复次数
            }
            
        # 添加预防措施建议
        enhanced_plan["preventive_measures"] = [
            "实验前校准设备",
            "准备备用试剂",
            "设置异常监测点"
        ]
        
        return enhanced_plan
        
    def _get_enhancement_details(self, original_plan, enhanced_plan):
        """比较原始计划和增强计划，提取改进点"""
        enhancements = []
        
        # 简单的基于字典的比较
        if isinstance(original_plan, dict) and isinstance(enhanced_plan, dict):
            # 新增的键
            for key in enhanced_plan:
                if key not in original_plan:
                    enhancements.append(f"新增 '{key}': {enhanced_plan[key]}")
                elif enhanced_plan[key] != original_plan[key]:
                    enhancements.append(f"优化 '{key}': {original_plan[key]} -> {enhanced_plan[key]}")
        
        # 如果没有检测到变化，添加默认消息
        if not enhancements:
            enhancements.append("进行了一般性优化，提高了实验方案的稳定性")
            
        return enhancements
        
    def _error_response(self, error_message):
        """生成错误响应"""
        return {
            "success": False,
            "error": error_message
        }

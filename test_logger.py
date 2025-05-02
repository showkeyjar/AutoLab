"""测试日志系统"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from autolab.core.logger import get_logger
from autolab.ui.streamlit_app import run_taskflow
from autolab.core.task_flow import TaskFlow

logger = get_logger(__name__)
logger.info("=== 开始日志系统测试 ===")

try:
    flow = TaskFlow()
    result, error = run_taskflow("测试任务", flow)
    logger.info(f"测试结果: {result}")
except Exception as e:
    logger.error(f"测试失败: {str(e)}")
finally:
    logger.info("=== 测试结束 ===")

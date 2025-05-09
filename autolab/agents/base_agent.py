"""
基础智能体模块的兼容导入文件
此文件用于兼容引用base_agent.py的代码
"""
from autolab.agents.base import BaseAgent

# 将base.py中的所有内容导出，确保兼容性
__all__ = ['BaseAgent']

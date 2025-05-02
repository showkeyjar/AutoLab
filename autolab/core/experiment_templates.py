from typing import Dict, List
import yaml
import os

class ExperimentTemplate:
    """实验模板管理系统"""
    
    DEFAULT_TEMPLATES = {
        "quick_start": {
            "description": "快速开始模板",
            "config": {
                "metrics": {"accuracy": {"enabled": True, "threshold": 0.8}},
                "steps": ["prepare", "train", "evaluate"]
            }
        }
    }
    
    def __init__(self, template_dir='config/templates'):
        self.template_dir = os.path.abspath(template_dir)
        os.makedirs(self.template_dir, exist_ok=True)
        self._init_default_templates()
    
    def _init_default_templates(self):
        """初始化默认模板"""
        for name, template in self.DEFAULT_TEMPLATES.items():
            safe_name = self._sanitize_name(name)
            path = os.path.join(self.template_dir, f"{safe_name}.yaml")
            if not os.path.exists(path):
                self.save_template(name, template["config"])
    
    def _sanitize_name(self, name: str) -> str:
        """规范化文件名"""
        return ''.join(c for c in name if c.isalnum() or c in ('_', '-'))
    
    def save_template(self, name: str, config: dict):
        """保存模板到YAML文件"""
        if not name or not isinstance(config, dict):
            raise ValueError("无效的模板名称或配置")
        
        safe_name = self._sanitize_name(name)
        path = os.path.join(self.template_dir, f"{safe_name}.yaml")
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, allow_unicode=True, indent=2)
            return True
        except Exception as e:
            print(f"保存模板失败: {str(e)}")
            return False
    
    def load_template(self, name: str) -> dict:
        """从YAML加载模板"""
        safe_name = self._sanitize_name(name)
        path = os.path.join(self.template_dir, f"{safe_name}.yaml")
        
        if not os.path.exists(path):
            return {}
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"加载模板失败: {str(e)}")
            return {}
    
    def list_templates(self) -> list:
        """列出所有可用模板"""
        return [f[:-5] for f in os.listdir(self.template_dir) 
                if f.endswith('.yaml')]

from typing import Dict, List
import yaml
import os

class ExperimentTemplate:
    """实验模板管理系统"""
    
    def __init__(self, template_dir='config/templates'):
        self.template_dir = template_dir
        os.makedirs(template_dir, exist_ok=True)
    
    def save_template(self, name: str, config: Dict):
        """保存模板到YAML文件"""
        path = os.path.join(self.template_dir, f"{name}.yaml")
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True)
    
    def load_template(self, name: str) -> Dict:
        """从YAML加载模板"""
        path = os.path.join(self.template_dir, f"{name}.yaml")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def list_templates(self) -> List[str]:
        """列出所有可用模板"""
        return [f[:-5] for f in os.listdir(self.template_dir) 
                if f.endswith('.yaml')]

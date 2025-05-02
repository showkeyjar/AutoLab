import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

class ExperimentState:
    """实验状态持久化管理"""
    
    def __init__(self, storage_dir='.experiment_states'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_state(self, experiment_id: str, state: Dict[str, Any]):
        """保存实验状态"""
        path = os.path.join(self.storage_dir, f"{experiment_id}.json")
        with open(path, 'w', encoding='utf-8') as f:
            state['last_updated'] = datetime.now().isoformat()
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def load_state(self, experiment_id: str) -> Dict[str, Any]:
        """加载实验状态"""
        path = os.path.join(self.storage_dir, f"{experiment_id}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def list_states(self) -> List[str]:
        """列出所有保存的实验状态"""
        return [f[:-5] for f in os.listdir(self.storage_dir) 
                if f.endswith('.json')]

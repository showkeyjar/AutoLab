"""控制命令安全检查"""
import re
from typing import Dict

class SafetyChecker:
    """防止危险操作"""
    
    BLACKLIST = [
        r"rm\s+-[rf]|del\s+/[sq]",  # 危险删除
        r"format\s+[a-z]:",         # 磁盘格式化
        r"shutdown|reboot",          # 系统操作
        r"^curl\s+.+\|\s*sh"       # 远程执行
    ]
    
    @classmethod
    def check_command(cls, command: Dict[str, Any]) -> bool:
        """验证命令安全性"""
        if command.get("type") == "shell":
            cmd = command["command"]
            return not any(
                re.search(pattern, cmd, re.IGNORECASE) 
                for pattern in cls.BLACKLIST
            )
        return True

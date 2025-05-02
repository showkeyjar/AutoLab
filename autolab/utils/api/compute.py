"""计算引擎服务(Docker/本地)"""
import docker
import subprocess
from typing import Dict, Any
from autolab.core.logger import get_logger

logger = get_logger(__name__)

class ComputeEngine:
    """支持两种运行模式:
    1. Docker容器(隔离环境)
    2. 本地进程(快速调试)
    """
    
    def __init__(self, use_docker=True):
        self.client = docker.from_env() if use_docker else None
        self.mode = "docker" if use_docker else "local"
    
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """执行计算任务"""
        try:
            if self.mode == "docker":
                return self._run_in_docker(config)
            return self._run_locally(config)
        except Exception as e:
            logger.error(f"计算任务失败: {str(e)}")
            return {"error": str(e)}
    
    def _run_in_docker(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """在Docker容器中运行"""
        container = self.client.containers.run(
            image=config["image"],
            command=config["command"],
            volumes=config.get("volumes"),
            environment=config.get("env"),
            detach=False,
            remove=True
        )
        return {"output": container.decode("utf-8")}
    
    def _run_locally(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """本地执行"""
        result = subprocess.run(
            config["command"],
            shell=True,
            capture_output=True,
            text=True
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def ping(self) -> bool:
        """测试引擎可用性"""
        try:
            if self.mode == "docker":
                return self.client.ping()
            return subprocess.run("echo test", shell=True).returncode == 0
        except Exception:
            return False

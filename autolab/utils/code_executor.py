import subprocess
import tempfile
import os
import sys
from typing import Dict, Any

class CodeExecutor:
    """
    安全沙箱环境下的 Python 代码执行器。
    """
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def execute_python(self, code: str, input_vars: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行一段 Python 代码，返回 stdout、stderr、exit_code。
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                input=None,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.returncode
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1
            }
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

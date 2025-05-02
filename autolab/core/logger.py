import logging
import inspect
import os

# 创建日志目录
log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
os.makedirs(log_dir, exist_ok=True)

# 配置日志格式
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'autolab.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 确保不导出logging模块
__all__ = ['get_logger']

def get_logger(name):
    """获取指定名称的日志记录器"""
    return logging.getLogger(name)

def log_with_line(message, level=logging.INFO):
    """带行号的日志记录"""
    frame = inspect.currentframe().f_back
    logger = logging.getLogger(frame.f_globals.get('__name__'))
    logger.log(level, f"Line {frame.f_lineno}: {message}")

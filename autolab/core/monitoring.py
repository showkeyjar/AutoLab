import time
from typing import Dict
import sqlite3

class MonitoringDB:
    """监控数据存储"""
    
    def __init__(self, db_path='monitoring.db'):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS metrics (
                          timestamp REAL,
                          experiment_id TEXT,
                          metric_name TEXT,
                          metric_value REAL
                        )''')
        self.conn.commit()
    
    def log_metric(self, experiment_id: str, metrics: Dict[str, float]):
        """记录一组指标"""
        cursor = self.conn.cursor()
        ts = time.time()
        for name, value in metrics.items():
            cursor.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?)",
                (ts, experiment_id, name, value)
            )
        self.conn.commit()
    
    def get_metrics(self, experiment_id: str) -> Dict[str, list]:
        """获取历史指标数据"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT metric_name, metric_value FROM metrics WHERE experiment_id=? ORDER BY timestamp",
            (experiment_id,)
        )
        
        results = {}
        for name, value in cursor.fetchall():
            if name not in results:
                results[name] = []
            results[name].append(value)
        return results

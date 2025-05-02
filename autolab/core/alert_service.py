import logging
from typing import Callable

class AlertService:
    """智能报警服务"""
    
    def __init__(self):
        self.rules = {
            'critical_error': lambda d: d.get('error_level', 0) >= 2,
            'performance_drop': lambda d: d.get('efficiency', 100) < 80
        }
        self.notifiers = {
            'console': self._console_notify,
            'email': None,
            'webhook': None
        }
    
    def check_alerts(self, data: dict) -> list:
        """检查触发哪些报警规则"""
        triggered = []
        for name, rule in self.rules.items():
            if rule(data):
                logging.warning(f"Alert triggered: {name}")
                triggered.append(name)
        return triggered
    
    def add_rule(self, name: str, condition: Callable):
        """添加自定义报警规则"""
        self.rules[name] = condition
    
    def add_notifier(self, name: str, callback: Callable):
        """注册通知渠道"""
        self.notifiers[name] = callback
    
    def _console_notify(self, alert: str, data: dict):
        print(f"[ALERT] {alert} - {data}")
    
    def send_alerts(self, alerts: list, data: dict):
        """发送所有触发的警报"""
        for alert in alerts:
            for name, notifier in self.notifiers.items():
                if notifier:
                    notifier(alert, data)

"""Agent连接诊断工具"""
import sys
from autolab.core.agent_manager import AgentManager

manager = AgentManager()
print(f"=== Agent连接诊断 ===")
print(f"已注册: {len(manager.agents)}个Agent")

for name, agent in manager.agents.items():
    print(f"\n{name}:")
    print(f"类型: {type(agent).__name__}")
    
    try:
        print(f"就绪状态: {'🟢' if agent.is_ready() else '🔴'}")
        if not agent.is_ready():
            print("尝试连接...")
            if agent.connect():
                print("连接成功")
            else:
                print("连接失败")
    except Exception as e:
        print(f"检查失败: {str(e)}")

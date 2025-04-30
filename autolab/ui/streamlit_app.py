import streamlit as st
from autolab.core.task_flow import TaskFlow
from autolab.core.agent_manager import AgentManager
from autolab.utils.llm_client import ollama_client
import traceback
import os

st.set_page_config(page_title="AutoLab 智能体实验系统", layout="wide")
st.title("🧪 AutoLab 智能体实验系统可视化界面")
st.markdown("""
- 输入实验目标，点击“运行实验流”按钮，一键体验端到端智能体协作。
- 支持分步查看每个智能体的结构化输出与原始 LLM 响应。
""")

with st.sidebar:
    st.header("⚙️ 系统配置")
    ollama_url = st.text_input(
        "Ollama API 地址", value=ollama_client.base_url, help="如 http://localhost:11434/api/generate"
    )
    ollama_model = st.text_input(
        "默认LLM模型", value=ollama_client.default_model, help="如 llama3、qwen:14b 等"
    )
    if st.button("应用配置"):
        ollama_client.base_url = ollama_url.strip()
        ollama_client.default_model = ollama_model.strip()
        st.success(f"已应用 Ollama 配置: {ollama_url} | 模型: {ollama_model}")

def_goal = "提升气象大模型的预测准确率"
user_goal = st.text_input("实验目标", def_goal, key="goal_input")
run_btn = st.button("运行实验流")

# 确保多次运行不会因 import 路径问题报错
os.environ["PYTHONPATH"] = os.getcwd()

def run_taskflow(goal):
    try:
        agent_manager = AgentManager()
        task_flow = TaskFlow(agent_manager)
        result = task_flow.run_flow({"goal": goal})
        return result, None
    except Exception as e:
        return None, traceback.format_exc()

if run_btn and user_goal.strip():
    with st.spinner("智能体协作中，请稍候..."):
        result, err = run_taskflow(user_goal.strip())
    if err:
        st.error("运行出错：\n" + err)
    elif result:
        st.success("实验流已完成！")
        for key, val in result.items():
            with st.expander(f"{key} 智能体输出", expanded=True):
                if isinstance(val, dict):
                    for sk, sv in val.items():
                        if sk.endswith("llm_raw") or sk == "llm_raw":
                            continue
                        st.write(f"**{sk}**:", sv)
                    if "llm_raw" in val:
                        with st.expander("原始 LLM 响应"):
                            st.code(val["llm_raw"], language="text")
                else:
                    st.write(val)
else:
    st.info("请在上方输入实验目标，然后点击“运行实验流”按钮。")

import sys
import os
import pandas as pd
import json

# 1. 当前脚本绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 项目根目录（autolab 的上一级）
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

import streamlit as st
from autolab.core.task_flow import TaskFlow
from autolab.core.agent_manager import AgentManager
from autolab.utils.llm_client import ollama_client
import traceback
import os

st.set_page_config(page_title="AutoLab 智能体实验系统", layout="wide")
st.title("🧪 AutoLab 智能体实验系统")
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
    
    # New metrics configuration
    st.header("📊 实验指标配置")
    accuracy_threshold = st.slider(
        "准确率阈值", min_value=0.0, max_value=1.0, value=0.8, step=0.05
    )
    efficiency_threshold = st.slider(
        "效率阈值", min_value=0, max_value=100, value=85
    )
    
    if st.button("应用配置"):
        ollama_client.base_url = ollama_url.strip()
        ollama_client.default_model = ollama_model.strip()
        # Save metrics configuration
        st.session_state.metrics_config = {
            "accuracy_threshold": accuracy_threshold,
            "efficiency_threshold": efficiency_threshold
        }
        st.success("配置已保存")

    # 在侧边栏添加模板选择
    st.header("📁 实验模板")
    template_name = st.selectbox(
        "选择模板", 
        options=["模板1", "模板2"],  # 这里需要替换成实际的模板列表
        index=0
    )
    
    # 模板编辑器
    template_action = st.selectbox(
        "模板操作", 
        options=["使用模板", "新建模板", "编辑模板"],
        index=0
    )

    if template_action == "新建模板":
        with st.form("new_template"):
            new_name = st.text_input("模板名称")
            config = st.text_area("模板配置(JSON格式)", height=200)
            if st.form_submit_button("保存"):
                try:
                    ExperimentTemplate().save_template(new_name, json.loads(config))
                    st.success("模板保存成功")
                except Exception as e:
                    st.error(f"保存失败: {str(e)}")

    elif template_action == "编辑模板":
        edit_config = ExperimentTemplate().load_template(template_name)
        edited = st.text_area("编辑配置", value=json.dumps(edit_config, indent=2), height=300)
        if st.button("更新模板"):
            try:
                ExperimentTemplate().save_template(template_name, json.loads(edited))
                st.success("模板更新成功")
            except Exception as e:
                st.error(f"更新失败: {str(e)}")
                
    # 实时监控区块
    st.header("📈 实时监控")
    monitor_placeholder = st.empty()

def_goal = "提升气象大模型的预测准确率"
user_goal = st.text_input("实验目标", def_goal, key="goal_input")
run_btn = st.button("运行实验流")

# Add manual stop button
if st.button("❌ 终止实验"):
    st.session_state.manual_stop = True
    st.warning("终止信号已发送，当前实验完成后将停止")

os.environ["PYTHONPATH"] = os.getcwd()

def run_taskflow(goal):
    try:
        task_flow = TaskFlow()  # 不传 agent_manager 参数
        
        # Initialize retry tracking
        st.session_state.setdefault('attempts', 0)
        st.session_state.setdefault('previous_actions', set())
        st.session_state.setdefault('attempt_history', [])
        st.session_state.setdefault('best_score', 0)
        
        while st.session_state.attempts < 10:
            result = task_flow.run_flow({"goal": goal})
            
            # Check metrics against thresholds
            metrics_ok = all(
                result['metrics'][metric] >= st.session_state.metrics_config[f"{metric}_threshold"]
                for metric in ['accuracy', 'efficiency']
            )
            
            if metrics_ok:
                return result, None
                
            # Check for repeated actions
            current_actions = frozenset(a['action'] for a in result['actions'])
            if current_actions in st.session_state.previous_actions:
                return None, "检测到重复动作序列，已终止"
                
            st.session_state.previous_actions.add(current_actions)
            st.session_state.attempts += 1
            
            # Update attempt history
            attempt_history = st.session_state.attempt_history
            attempt_history.append({
                'attempt': st.session_state.attempts,
                'score': result['metrics']['accuracy']
            })
            st.session_state.attempt_history = attempt_history
            
            # Update best score
            if result['metrics']['accuracy'] > st.session_state.best_score:
                st.session_state.best_score = result['metrics']['accuracy']
            
            # Check for manual stop
            if st.session_state.get('manual_stop', False):
                return None, "人工干预终止"
            
        return None, "超过最大尝试次数(10次)仍未达标"
        
    except Exception as e:
        return None, traceback.format_exc()

if run_btn and user_goal.strip():
    with monitor_placeholder.container():
        st.subheader("实验指标")
        col1, col2, col3 = st.columns(3)
        
        # 这里会实时更新指标数据
        col1.metric("准确率", "0.85", "+2%")
        col2.metric("效率", "82%", "-1%")
        col3.metric("耗时", "1.2s", delta_color="off")

    with st.spinner("智能体协作中，请稍候..."):
        result, err = run_taskflow(user_goal.strip())
    if err:
        st.error("运行出错：\n" + err)
    elif result:
        st.success("实验流已完成！")
        for key, val in result.items():
            st.markdown(f"### {key} 智能体输出")
            if isinstance(val, dict):
                # 结构化主信息
                for sk, sv in val.items():
                    if sk.endswith("llm_raw") or sk == "llm_raw":
                        continue
                    st.write(f"**{sk}**:", sv)
                # 原始 LLM 响应单独用 code 展示，避免嵌套 expander
                if "llm_raw" in val:
                    st.markdown("**原始 LLM 响应：**")
                    st.code(val["llm_raw"], language="text")
            else:
                st.write(val)
        
        # Add progress visualization
        if st.session_state.get('attempts', 0) > 0:
            st.subheader("实验进度")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("尝试次数", st.session_state['attempts'])
                st.progress(min(st.session_state['attempts'] / 10, 1.0))
            
            with col2:
                if st.session_state.get('best_score'):
                    st.metric("最佳得分", f"{st.session_state['best_score']:.2f}")
                
            # Show attempt history
            if st.session_state.get('attempt_history'):
                history_df = pd.DataFrame(st.session_state['attempt_history'])
                st.line_chart(history_df.set_index('attempt')['score'])
        
        # 在结果展示部分添加优化信息
        if result and 'optimized_params' in result:
            st.subheader("参数优化结果")
            st.json(result['optimized_params'])
            
            # 显示优化历史
            history = task_flow.optimizer.history
            if history:
                df = pd.DataFrame([{
                    'attempt': i+1, 
                    'score': score,
                    **params
                } for i, (params, score) in enumerate(history)])
                
                st.line_chart(df.set_index('attempt')['score'])
else:
    st.info("请在上方输入实验目标，然后点击“运行实验流”按钮。")

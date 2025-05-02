import sys
import os
import pandas as pd
import json
import uuid

# 设置项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# 现在可以安全导入本地模块
from autolab.core.experiment_state import ExperimentState
from autolab.core.experiment_templates import ExperimentTemplate
from autolab.core.task_flow import TaskFlow
from autolab.utils.llm_client import OllamaClient

# 1. 当前脚本绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 项目根目录（autolab 的上一级）
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

import streamlit as st
import traceback
import os

state_manager = ExperimentState()

def init_session_state():
    """初始化所有session state变量"""
    if 'metrics_config' not in st.session_state:
        st.session_state.metrics_config = {
            "accuracy": {"enabled": True, "threshold": 0.8, "calculation": "correct_count / total_count"},
            "time_cost": {"enabled": True, "threshold": 60.0},
            "custom": []
        }
    if 'templates_init' not in st.session_state:
        st.session_state.templates_init = False

init_session_state()

st.set_page_config(page_title="AutoLab 智能体实验系统", layout="wide")
st.title("🧪 AutoLab 智能体实验系统")
st.markdown("""
- 输入实验目标，点击“运行实验流”按钮，一键体验端到端智能体协作。
- 支持分步查看每个智能体的结构化输出与原始 LLM 响应。
""")

with st.sidebar:
    st.header("⚙️ 系统配置")
    # 初始化Ollama客户端
    ollama_client = OllamaClient()
    ollama_url = st.text_input(
        "Ollama API 地址", 
        value=ollama_client.base_url, 
        help="如 http://localhost:11434"
    )
    
    # 在配置区块修改模型选择逻辑
    ollama_model = st.selectbox(
        "默认LLM模型", 
        options=ollama_client.available_models,
        index=0,
        help="如 llama3、qwen:14b 等"
    )
    
    # 添加模型状态提示
    if not ollama_client.available_models:
        st.warning("未检测到可用模型，请确保Ollama服务已运行")
        st.stop()  # 没有可用模型时停止执行
    
    # 指标配置部分
    with st.expander("⚙️ 实验指标配置", expanded=False):
        # 主指标配置
        st.subheader("核心指标")
        cols = st.columns(2)
        with cols[0]:
            acc_enabled = st.checkbox("启用准确率", st.session_state.metrics_config["accuracy"]["enabled"])
            acc_threshold = st.number_input("准确率阈值", min_value=0.0, max_value=1.0, 
                                          value=st.session_state.metrics_config["accuracy"]["threshold"], step=0.05)
        with cols[1]:
            time_enabled = st.checkbox("启用耗时统计", st.session_state.metrics_config["time_cost"]["enabled"])
            time_threshold = st.number_input("最大耗时(秒)", min_value=0.0, 
                                           value=st.session_state.metrics_config["time_cost"]["threshold"])
        
        # 自定义指标
        st.subheader("自定义指标")
        for i, custom in enumerate(st.session_state.metrics_config["custom"]):
            with st.expander(f"指标 {i+1}: {custom['name']}", expanded=False):
                st.text_area(f"计算公式", value=custom.get("formula", ""), key=f"custom_formula_{i}")
        
        if st.button("保存配置"):
            st.session_state.metrics_config.update({
                "accuracy": {"enabled": acc_enabled, "threshold": acc_threshold},
                "time_cost": {"enabled": time_enabled, "threshold": time_threshold}
            })
            st.success("指标配置已保存")

def show_template_guide():
    """显示模板引导界面"""
    with st.sidebar:
        st.header("📁 实验模板")
        
        template_manager = ExperimentTemplate()
        
        if not template_manager.list_templates():
            with st.expander("🆕 新手指南", expanded=True):
                st.markdown("""
                ### 第一次使用模板？
                1. 点击下方按钮创建默认模板
                2. 或选择"新建模板"创建自定义模板
                3. 模板可保存常用实验配置
                """)
                if st.button("✨ 一键创建默认模板"):
                    init_default_templates()
                    st.rerun()
            return
        
        # 正常模板选择界面
        template_name = st.selectbox("选择模板", options=template_manager.list_templates())
        template_action = st.radio("操作", ["使用模板", "编辑模板", "新建模板"], horizontal=True)
        
        if template_action == "新建模板":
            with st.form("new_template_form"):
                new_name = st.text_input("模板名称")
                if st.form_submit_button("创建"):
                    try:
                        template_manager.save_template(new_name, {"metrics": st.session_state.metrics_config})
                        st.success(f"模板'{new_name}'创建成功")
                        st.rerun()
                    except Exception as e:
                        st.error(f"创建失败: {str(e)}")
        
        elif template_action == "编辑模板":
            edit_config = template_manager.load_template(template_name) or {}
            edited = st.text_area("编辑配置", value=json.dumps(edit_config, indent=2), height=300)
            if st.button("保存更改"):
                try:
                    template_manager.save_template(template_name, json.loads(edited))
                    st.success("模板更新成功")
                except Exception as e:
                    st.error(f"更新失败: {str(e)}")

def init_default_templates():
    """初始化默认模板"""
    default_templates = {
        "基础分类实验": {
            "metrics": {"accuracy": {"threshold": 0.85}}, 
            "steps": ["数据加载", "特征工程", "模型训练", "评估"]
        },
        "时间序列预测": {
            "metrics": {"mae": {"threshold": 0.1}, "time_cost": {"threshold": 300}},
            "steps": ["数据预处理", "特征生成", "模型训练", "预测", "可视化"]
        }
    }
    
    template_manager = ExperimentTemplate()
    for name, config in default_templates.items():
        if name not in template_manager.list_templates():
            template_manager.save_template(name, config)

show_template_guide()

# 实时监控区块
st.header("📈 实时监控")
monitor_placeholder = st.empty()

saved_experiments = state_manager.list_states()
if saved_experiments:
    selected_exp = st.selectbox("恢复实验", options=saved_experiments)
    if st.button("加载"):
        state = state_manager.load_state(selected_exp)
        st.session_state.update(state)
        st.rerun()

def_goal = "提升气象大模型的预测准确率"
user_goal = st.text_input("实验目标", def_goal, key="goal_input")

# 在运行按钮旁添加中断按钮
col1, col2 = st.columns([3,1])
with col1:
    run_btn = st.button("运行实验流")
with col2:
    stop_btn = st.button("🛑 强制停止", type="secondary")

# 在实验运行前检查中断状态
if stop_btn:  # 检查是否触发中断
    if 'OllamaClient' in globals():
        ollama_client.stop()
    st.warning("正在停止当前实验...")
    st.stop()

os.environ["PYTHONPATH"] = os.getcwd()

if 'experiment_id' not in st.session_state:
    st.session_state.experiment_id = str(uuid.uuid4())
    # 尝试加载已有状态
    saved_state = state_manager.load_state(st.session_state.experiment_id)
    if saved_state:
        st.session_state.update(saved_state)

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
            if 'metrics' in result:
                metrics_ok = True
                for metric, config in st.session_state.metrics_config.items():
                    if config.get('enabled', False) and metric in result['metrics']:
                        if result['metrics'][metric] < config.get('threshold', 0):
                            metrics_ok = False
                            break
            else:
                metrics_ok = False
            
            if metrics_ok:
                return result, None
                
            # Check for repeated actions
            if 'previous_actions' not in st.session_state:
                st.session_state.previous_actions = set()

            current_actions = frozenset(a['action'] for a in result.get('actions', []))

            # 只有当有实际动作时才检查重复
            if current_actions and current_actions in st.session_state.previous_actions:
                st.warning("⚠️ 检测到可能的重复动作序列")
                cols = st.columns(3)
                cols[0].write("上次动作序列:")
                for a in st.session_state.previous_actions:
                    cols[1].write(f"- {a}")
                
                cols[0].write("当前动作序列:")
                for a in current_actions:
                    cols[2].write(f"- {a}")
                
                if st.checkbox("仍然继续执行"):
                    st.session_state.previous_actions.add(current_actions)
                    return result, None
                else:
                    return None, "用户选择终止执行"
            else:
                st.session_state.previous_actions.add(current_actions)
                return result, None
            
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

def generate_diagnostic_report(result, metrics_config):
    """生成实验诊断报告"""
    report = {
        'summary': '实验未达到预期目标',
        'failed_metrics': [],
        'action_sequence': [],
        'suggestions': []
    }
    
    # 分析未达标指标
    if isinstance(metrics_config, dict) and 'metrics' in result:
        for metric, config in metrics_config.items():
            if metric != 'custom':  # 跳过custom列表
                if isinstance(config, dict) and config.get('enabled', False):
                    if metric in result['metrics'] and isinstance(result['metrics'][metric], (int, float)):
                        threshold = config.get('threshold', 0)
                        if result['metrics'][metric] < threshold:
                            report['failed_metrics'].append({
                                'metric': metric,
                                'actual': result['metrics'][metric],
                                'expected': threshold,
                                'delta': threshold - result['metrics'][metric]
                            })
    
    # 记录执行过程
    if 'actions' in result and isinstance(result['actions'], list):
        report['action_sequence'] = [
            f"{i+1}. {action.get('action', '未知动作')}" 
            for i, action in enumerate(result['actions'])
        ]
    
    # 生成改进建议
    if report['failed_metrics']:
        for metric in report['failed_metrics']:
            if metric['metric'] == 'accuracy':
                report['suggestions'].append("尝试增加训练数据量或调整模型参数")
            elif metric['metric'] == 'time_cost':
                report['suggestions'].append("优化计算步骤或减少数据规模")
    
    return report

def show_diagnostic_report(report):
    """显示诊断报告"""
    with st.expander("🔍 实验诊断报告", expanded=True):
        st.subheader("问题总结")
        st.write(report['summary'])
        
        if report['failed_metrics']:
            st.subheader("未达标指标")
            for metric in report['failed_metrics']:
                st.error(
                    f"{metric['metric']}: 实际值 {metric['actual']:.2f} "
                    f"(期望 ≥ {metric['expected']:.2f}, 差距 {metric['delta']:.2f})"
                )
        
        if report['action_sequence']:
            st.subheader("执行步骤")
            st.write('\n'.join(report['action_sequence']))
        
        if report['suggestions']:
            st.subheader("改进建议")
            for suggestion in report['suggestions']:
                st.info(f"💡 {suggestion}")

if run_btn and user_goal.strip():
    try:
        with st.spinner("智能体协作中，请稍候..."):
            result, err = run_taskflow(user_goal.strip())
            if stop_btn:  # 检查是否触发中断
                raise RuntimeError("实验被用户中断")
    except RuntimeError as e:
        st.error(f"实验中断: {str(e)}")
    else:
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
            
            # 保存状态
            state_manager.save_state(
                st.session_state.experiment_id,
                {
                    'goal': user_goal,
                    'result': result,
                    'history': st.session_state.get('attempt_history', []),
                    'config': {
                        'ollama_model': ollama_model,
                        'metrics_cfg': st.session_state.metrics_config
                    }
                }
            )

            # 在结果显示部分
            if 'metrics' in result:
                st.subheader("📊 指标结果")
                cols = st.columns(3)
                for i, (metric, value) in enumerate(result['metrics'].items()):
                    cols[i % 3].metric(
                        label=metric,
                        value=f"{value:.2f}",
                        delta=f"达标" if value >= st.session_state.metrics_config.get(metric, {}).get('threshold', 0) else "未达标",
                        delta_color="normal"
                    )
            
            # 增强结果处理健壮性
            if result.get('actions'):
                st.subheader("执行动作")
                for action in result['actions']:
                    st.markdown(f"- {action['action']}")
            
            # 生成诊断报告
            diagnostic_report = generate_diagnostic_report(result, st.session_state.metrics_config)
            show_diagnostic_report(diagnostic_report)

else:
    st.info("请在上方输入实验目标，然后点击“运行实验流”按钮。")

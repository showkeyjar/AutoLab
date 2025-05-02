import sys
import os
import pandas as pd
import json
import uuid
from datetime import datetime
from typing import Dict, Any
from autolab.core.logger import get_logger
logger = get_logger(__name__)
from autolab.core.task_flow import TaskFlow
from autolab.core.experiment_state import ExperimentState
from autolab.core.experiment_templates import ExperimentTemplate
from autolab.utils.llm_client import OllamaClient

# 1. 当前脚本绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 项目根目录（autolab 的上一级）
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

import streamlit as st
import traceback
import os
import logging
from pathlib import Path

# 创建日志目录
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "autolab.log"),
        logging.StreamHandler()
    ]
)

# 获取logger实例
logger = logging.getLogger(__name__)

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
    llm_client = OllamaClient()  # 使用默认配置初始化
    ollama_url = st.text_input(
        "Ollama API 地址", 
        value=llm_client.base_url if hasattr(llm_client, 'base_url') else "http://localhost:11434",
        help="如 http://localhost:11434"
    )
    
    # 在配置区块修改模型选择逻辑
    if st.button("刷新模型列表"):
        llm_client = OllamaClient(base_url=ollama_url)
        st.rerun()  # 刷新界面
    
    if llm_client.available_models:
        ollama_model = st.selectbox(
            "选择LLM模型",
            options=llm_client.available_models,
            index=0
        )
        if ollama_model != llm_client.model:
            llm_client.model = ollama_model
            st.success(f"已切换至模型: {ollama_model}")
    else:
        st.warning("未检测到可用模型，请确保:")
        st.write("1. Ollama服务已运行")
        st.write(f"2. 正确配置API地址: {ollama_url}")
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

def init_task_flow():
    """初始化任务流"""
    try:
        logger.info("初始化任务流...")
        task_flow = TaskFlow()
        logger.info("任务流初始化完成")
    except Exception as e:
        logger.critical("任务流初始化失败", exc_info=True)
        raise
    
    # 确保至少有一个task_manager
    if not hasattr(task_flow.agent_manager, 'agents') or 'task_manager' not in task_flow.agent_manager.agents:
        raise ValueError("系统中缺少task_manager Agent")
    
    return task_flow

task_flow = init_task_flow()

def log_agent_status():
    """记录所有Agent状态到日志"""
    for name, agent in task_flow.agent_manager.agents.items():
        try:
            status = {
                'name': name,
                'type': type(agent).__name__,
                'connected': getattr(agent, '_connected', None),
                'mock_mode': getattr(agent, 'mock_mode', None),
                'last_error': getattr(agent, '_last_error', None)
            }
            logger.info(f"Agent状态: {status}")
        except Exception as e:
            logger.error(f"获取Agent状态失败: {name}", exc_info=True)

log_agent_status()

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
        llm_client.stop()
    st.warning("正在停止当前实验...")
    st.stop()

os.environ["PYTHONPATH"] = os.getcwd()

if 'experiment_id' not in st.session_state:
    st.session_state.experiment_id = str(uuid.uuid4())
    # 尝试加载已有状态
    saved_state = state_manager.load_state(st.session_state.experiment_id)
    if saved_state:
        st.session_state.update(saved_state)

def process_user_input(user_input: str) -> Dict[str, Any]:
    """将用户输入转换为标准任务格式"""
    return {
        "goal": user_input,
        "type": "root_task",
        "constraints": ""
    }

def run_taskflow(task_input: Dict[str, Any], flow):
    """运行任务流并处理结果"""
    logger.info(f"收到任务输入: {task_input}")
    
    # 调试日志：打印所有注册的Agent
    if hasattr(flow, 'agent_manager') and hasattr(flow.agent_manager, 'agents'):
        logger.debug(f"当前注册Agent: {list(flow.agent_manager.agents.keys())}")
    
    try:
        result = flow.run_flow(task_input)
        if result.get('error'):
            logger.error(f"任务执行错误: {result['error']}")
            return None, result['error']
        else:
            logger.info(f"任务完成，结果长度: {len(str(result))}")
            return result, None
    except Exception as e:
        logger.critical(f"系统异常: {str(e)}")
        return None, str(e)

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

def show_experiment_details():
    """显示选中的实验详情"""
    if 'experiment_id' in st.session_state or 'result' in st.session_state:
        with st.expander("📝 实验详情", expanded=True):
            st.write(f"**实验ID**: {st.session_state.get('experiment_id', '当前实验')}")
            st.write(f"**实验目标**: {st.session_state.get('goal', '无')}")
            st.write(f"**创建时间**: {st.session_state.get('timestamp', '未知')}")
            
            if 'result' in st.session_state:
                st.subheader("执行结果")
                
                if 'metrics' in st.session_state['result']:
                    st.write("**指标结果**")
                    for metric, value in st.session_state['result']['metrics'].items():
                        st.metric(
                            label=metric,
                            value=f"{value:.2f}",
                            delta=f"达标" if value >= st.session_state.metrics_config.get(metric, {}).get('threshold', 0) else "未达标"
                        )
                
                if 'actions' in st.session_state['result']:
                    st.write("**执行步骤**")
                    for i, action in enumerate(st.session_state['result']['actions'], 1):
                        st.write(f"{i}. {action.get('action', '未知动作')}")
                        
            if 'config' in st.session_state:
                st.subheader("实验配置")
                st.write(f"**使用模板**: {st.session_state['config'].get('template', '无')}")
                st.write("**指标配置**")
                st.json(st.session_state['config'].get('metrics', {}))
            
            # 添加导出按钮
            export_experiment()

def export_experiment():
    """导出实验数据"""
    if 'experiment_id' not in st.session_state:
        return
    
    experiment_data = {
        'metadata': {
            'id': st.session_state['experiment_id'],
            'goal': st.session_state.get('goal', ''),
            'timestamp': st.session_state.get('timestamp', '')
        },
        'result': st.session_state.get('result', {}),
        'config': st.session_state.get('config', {})
    }
    
    json_str = json.dumps(experiment_data, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 导出实验数据",
        data=json_str,
        file_name=f"experiment_{st.session_state['experiment_id']}.json",
        mime="application/json"
    )

def show_agent_debug_info(result):
    """显示Agent调试信息"""
    if not isinstance(result, dict) or 'debug' not in result:
        return
        
    with st.expander("🐞 Agent调试信息", expanded=False):
        st.write("执行步骤追踪:")
        for step in result['debug'].get('agent_steps', []):
            st.write(f"{step['timestamp']} - {step['agent']}: {step['action']} ({step['status']})")
            if step.get('details'):
                st.json(step['details'])

def process_task_result(result):
    """处理任务结果确保格式统一"""
    if not isinstance(result, dict):
        return {
            'output': result,
            'history': [],
            'success': True if result else False
        }
    
    # 确保包含必要字段
    for field in ['history', 'output', 'success']:
        if field not in result:
            result[field] = [] if field == 'history' else None if field == 'output' else False
    
    return result

def show_agent_status(flow):
    """增强版状态显示"""
    if not hasattr(flow, 'agent_manager'):
        st.sidebar.error("⚠️ Agent管理器未加载")
        return
        
    status = flow.agent_manager.test_connections()
    if all(s.startswith("🔴") for s in status.values()):
        st.sidebar.error("⚠️ 所有Agent离线！")
        if st.sidebar.button("强制重连"):
            try:
                with st.spinner("重新初始化中..."):
                    flow.agent_manager.start_collaboration()
                st.rerun()
            except Exception as e:
                st.error(f"重连失败: {str(e)}")
    else:
        # 正常显示状态...
        pass

show_agent_status(task_flow)

def show_diagnostics(flow):
    """显示系统诊断信息"""
    with st.expander("🛠️ 系统诊断", expanded=True):
        
        # 基础环境检查
        st.subheader("环境检查")
        col1, col2 = st.columns(2)
        
        try:
            import socket
            hostname = socket.gethostname()
            col1.metric("主机名", hostname)
        except Exception as e:
            col1.error(f"主机名获取失败: {str(e)}")
            
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            col2.metric("CPU使用率", f"{cpu_percent}%")
            col2.metric("内存使用", f"{mem.percent}%")
        except ImportError:
            col2.warning("psutil未安装，无法获取资源使用情况")
        
        # 网络连接检查
        st.subheader("网络检查")
        test_urls = [
            ("Ollama服务", llm_client.base_url),
            ("互联网", "https://www.baidu.com")
        ]
        
        for name, url in test_urls:
            try:
                import requests
                r = requests.get(url, timeout=3)
                st.success(f"{name} ({url}) 连接正常 (HTTP {r.status_code})")
            except Exception as e:
                st.error(f"{name} ({url}) 连接失败: {str(e)}")
        
        # Agent详细状态
        st.subheader("Agent状态详情")
        for name, agent in flow.agent_manager.agents.items():
            st.markdown(f"**{name}** (`{type(agent).__name__}`)")
            
            # 连接状态
            if not hasattr(agent, '_connected'):
                st.error("⚠️ 缺少连接状态属性")
                continue
                
            cols = st.columns([1,3])
            cols[0].write(f"连接状态: {'✅ 已连接' if agent._connected else '❌ 未连接'}")
            
            # 连接/重连按钮
            if cols[0].button(f"测试连接 {name}", key=f"connect_{name}"):
                try:
                    result = agent.connect()
                    if result:
                        st.success(f"{name} 连接成功")
                    else:
                        st.error(f"{name} 连接返回False")
                except Exception as e:
                    st.exception(e)
            
            # 显示详细属性
            attr_cols = st.columns([1,1,1])
            with attr_cols[0]:
                st.caption("模拟模式")
                st.code(str(getattr(agent, 'mock_mode', None)))
            with attr_cols[1]:
                st.caption("就绪状态")
                st.code(str(getattr(agent, 'is_ready', None)))
            with attr_cols[2]:
                st.caption("最后错误")
                st.code(str(getattr(agent, '_last_error', None)) or "无")

show_diagnostics(task_flow)

if run_btn and user_goal.strip():
    # 在任务执行前添加检查
    if not hasattr(task_flow, 'agent_manager') or not task_flow.agent_manager.agents:
        st.error("⚠️ Agent管理器未正确初始化！")
        st.stop()
    
    # 检查在线Agent数量
    online_agents = []
    agent_status = {}

    for name, agent in task_flow.agent_manager.agents.items():
        try:
            if not hasattr(agent, '_connected'):
                status = "❌ 缺少连接状态"
            elif agent._connected:
                status = "✅ 已连接"
                online_agents.append(name)
            else:
                status = "⚠️ 未连接"
                
            # 记录详细状态
            agent_status[name] = {
                'status': status,
                'type': type(agent).__name__,
                'mock_mode': getattr(agent, 'mock_mode', False)
            }
        except Exception as e:
            agent_status[name] = {
                'status': f"❌ 检查失败: {str(e)}",
                'error': str(e)
            }

    # 显示详细状态
    with st.expander("智能体详细状态"):
        for name, info in agent_status.items():
            st.write(f"{name} ({info.get('type','')}): {info['status']}")
            if 'error' in info:
                st.error(info['error'])
            if info.get('mock_mode', False):
                st.caption("⚠️ 当前处于模拟模式")

    if not online_agents:
        st.error("⚠️ 没有可用的在线Agent！")
        if st.button("🔄 尝试重新连接所有Agent"):
            for agent in task_flow.agent_manager.agents.values():
                if hasattr(agent, 'connect'):
                    agent.connect()
            st.rerun()
        
    try:
        # 确保task_flow和agent_manager已正确初始化
        if not hasattr(task_flow, 'agent_manager'):
            st.error("Agent管理器未初始化，请检查系统配置")
            st.stop()
            
        # 更新测试连接部分
        if not hasattr(task_flow, 'agent_manager'):
            st.error("Agent管理器未初始化")
            st.stop()
            
        # 使用TaskFlow的测试方法
        test_result = task_flow.test_agent_connections()
        if isinstance(test_result, dict) and 'error' in test_result:
            st.error(test_result['error'])
            st.stop()
        elif not all(v is True for v in test_result.values() if isinstance(v, bool)):
            st.error("部分Agent连接测试失败")
            st.json(test_result)
            st.stop()
            
        # 确保任务输入是字典
        task_input = process_user_input(user_goal.strip())
        with st.spinner("智能体协作中，请稍候..."):
            result, error = run_taskflow(task_input, task_flow)
            
        if error:
            st.error(f"执行失败: {error}")
        else:
            if not result.get('output'):
                st.warning("任务已执行但无输出结果，请检查Agent配置")
            else:
                st.success("实验完成！")
                
            # 显示详细信息
            show_agent_debug_info(result)
            show_experiment_details()
            
            # 在任务执行后添加参与度检查
            with st.expander("🔍 任务执行分析"):
                st.subheader("Agent参与情况")
                
                # 获取任务处理路径
                task_path = result.get('debug', {}).get('agent_path', [])
                if not task_path:
                    st.warning("⚠️ 没有记录到Agent参与路径")
                    
                    # 检查Agent调用情况
                    st.write("各Agent最后活动时间:")
                    for name, agent in task_flow.agent_manager.agents.items():
                        last_active = getattr(agent, '_last_active', '从未调用')
                        st.write(f"- {name}: {last_active}")
                else:
                    st.success(f"任务处理路径: {' → '.join(task_path)}")
                    
                # 显示原始debug信息
                if 'debug' in result:
                    st.json(result['debug'])
    except Exception as e:
        st.error(f"系统错误: {str(e)}")

else:
    st.info("请在上方输入实验目标，然后点击“运行实验流”按钮。")

# 在侧边栏添加调试面板
with st.sidebar.expander("🛠️ Agent调试", expanded=False):
    if hasattr(task_flow, 'agent_manager') and task_flow.agent_manager:
        st.write(f"已注册Agent数量: {len(task_flow.agent_manager.agents)}")
        
        for name, agent in task_flow.agent_manager.agents.items():
            st.write(f"- {name}: {type(agent).__name__}")
            
        if st.button("测试Agent连接"):
            try:
                test_result = task_flow.agent_manager.test_connections()
                st.json(test_result)
            except Exception as e:
                st.error(f"测试失败: {str(e)}")
    else:
        st.warning("Agent管理器未初始化")

def show_experiment_history():
    """显示实验历史记录"""
    with st.sidebar.expander("📚 实验历史", expanded=True):
        saved_experiments = state_manager.list_states()
        
        if saved_experiments:
            # 按时间倒序排序
            sorted_experiments = sorted(
                saved_experiments,
                key=lambda x: state_manager.load_state(x).get('timestamp', ''),
                reverse=True
            )
            
            selected_exp = st.selectbox(
                "选择实验记录",
                options=sorted_experiments,
                format_func=lambda x: f"{x} ({state_manager.load_state(x).get('timestamp', '未知时间')})",
                key="selected_experiment"
            )
            
            cols = st.columns(2)
            if cols[0].button("🔍 查看详情", key="view_experiment"):
                state = state_manager.load_state(selected_exp)
                st.session_state.update(state)
                st.rerun()
                
            if cols[1].button("🗑️ 删除记录", key="delete_experiment"):
                state_manager.delete_state(selected_exp)
                st.success(f"已删除实验记录: {selected_exp}")
                st.rerun()
        else:
            st.info("暂无历史实验记录")

show_experiment_history()

def show_execution_logs():
    """显示执行日志"""
    with st.sidebar.expander("📜 执行日志", expanded=False):
        if 'execution_logs' in st.session_state and st.session_state.execution_logs:
            for log in reversed(st.session_state.execution_logs):
                with st.expander(f"{log['timestamp']} - {log['action']}", expanded=False):
                    st.json(log['details'])
            
            if st.button("🔄 刷新日志"):
                st.rerun()
            
            if st.button("🧹 清除日志"):
                st.session_state.execution_logs = []
                st.rerun()
        else:
            st.info("暂无执行日志")

show_execution_logs()

# 自动修复选项
with st.sidebar.expander("🛠️ 故障修复"):
    if st.button("🔄 强制进入模拟模式"):
        for name, agent in task_flow.agent_manager.agents.items():
            try:
                if hasattr(agent, 'mock_mode'):
                    agent.mock_mode = True
                    agent.connect()
                    st.success(f"{name} 已进入模拟模式")
                else:
                    st.warning(f"{name} 不支持模拟模式")
            except Exception as e:
                st.error(f"{name} 设置失败: {str(e)}")
        st.rerun()
    
    if st.button("📋 导出诊断报告"):
        import json
        from datetime import datetime
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "environment": {}
        }
        
        # 收集Agent信息
        for name, agent in task_flow.agent_manager.agents.items():
            try:
                report["agents"][name] = {
                    "type": type(agent).__name__,
                    "connected": getattr(agent, '_connected', None),
                    "mock_mode": getattr(agent, 'mock_mode', None),
                    "last_error": str(getattr(agent, '_last_error', None))
                }
            except Exception as e:
                report["agents"][name] = {"error": str(e)}
        
        # 收集环境信息
        try:
            import platform, socket
            report["environment"] = {
                "system": platform.system(),
                "hostname": socket.gethostname(),
                "python_version": platform.python_version()
            }
        except Exception as e:
            report["environment"] = {"error": str(e)}
        
        # 保存报告
        report_path = log_dir / f"diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        st.success(f"诊断报告已保存到: {report_path}")
        st.download_button(
            "⬇️ 下载报告",
            data=json.dumps(report, indent=2),
            file_name=report_path.name,
            mime="application/json"
        )

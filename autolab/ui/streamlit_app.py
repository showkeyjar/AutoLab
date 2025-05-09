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
from autolab.ui.goal_display import show_structured_goal, export_structured_goal, edit_structured_goal

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
    
    # 结构化目标相关状态
    if 'parsed_goal' not in st.session_state:
        st.session_state.parsed_goal = None
    if 'show_edit_goal' not in st.session_state:
        st.session_state.show_edit_goal = False
        
    # 初始化日志
    if 'execution_logs' not in st.session_state:
        st.session_state.execution_logs = []

init_session_state()

st.set_page_config(page_title="AutoLab 智能体实验系统", layout="wide")
st.title("🧪 AutoLab 智能体实验系统")
st.markdown("""
- 输入实验目标，点击“运行实验流”按钮，一键体验端到端智能体协作。
- 支持分步查看每个智能体的结构化输出与原始 LLM 响应。
""")

# 实验配置区域
expander = st.expander("🧪 实验核心配置", expanded=True)
with expander:
    col1, col2 = st.columns(2)
    
    with col1:
        # 实验目标
        goal = st.text_area(
            "实验目标",
            key="goal",
            help="请描述实验的主要目标和预期成果"
        )
        
    with col2:
        # 实验基准
        benchmark = st.text_area(
            "实验基准描述",
            key="benchmark",
            help="请描述评估实验结果的基准标准"
        )
    
    # 自定义指标
    with st.container():
        st.subheader("📊 自定义指标")
        custom_metrics = st.text_area(
            "自定义评估指标 (每行一个)",
            key="custom_metrics",
            height=100,
            help="输入额外的评估指标，每行一个"
        )

# 执行控制区域
with st.container():
    st.subheader("🚀 执行控制")
    col_run, col_stop = st.columns([1,1])
    
    with col_run:
        if st.button("▶️ 开始实验", type="primary"):
            # 设置会话状态以触发实验执行
            st.session_state.start_experiment = True
            st.rerun()
            
    with col_stop:
        if st.button("⏹️ 停止实验"):
            # 中断执行
            if 'OllamaClient' in globals():
                llm_client.stop()
            task_flow.stop()
            st.warning("正在停止当前实验...")
            st.session_state.start_experiment = False
            st.rerun()

# 系统配置 - 放在侧边栏
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 初始化Ollama客户端
    ollama_url = st.text_input(
        "Ollama API 地址", 
        value="http://localhost:11434",
        help="如 http://localhost:11434"
    )
    
    # 测试连接
    if st.button("测试连接"):
        try:
            from autolab.utils.llm_client import OllamaClient
            llm_client = OllamaClient(base_url=ollama_url)
            if llm_client.available_models:
                st.success(f"连接成功! 可用模型: {', '.join(llm_client.available_models)}")
            else:
                st.warning("连接成功但未检测到模型")
        except Exception as e:
            st.error(f"连接失败: {str(e)}")
            
    if not hasattr(st.session_state, 'llm_client') or st.button("重新连接"):
        try:
            from autolab.utils.llm_client import OllamaClient
            st.session_state.llm_client = OllamaClient(base_url=ollama_url)
            st.success("LLM客户端已初始化")
        except Exception as e:
            st.error(f"LLM客户端初始化失败: {str(e)}")
            st.stop()
    
    llm_client = st.session_state.get('llm_client')
    
    if not llm_client or not llm_client.available_models:
        st.warning("未检测到可用模型，请确保:")
        st.write("1. Ollama服务已运行")
        st.write(f"2. 正确配置API地址: {ollama_url}")
        st.stop()  # 没有可用模型时停止执行
    
    # 指标配置部分
    with st.expander("⚙️ 实验指标配置", expanded=False):
        # 自然语言基准输入
        benchmark = st.text_area(
            "实验基准描述",
            help="用自然语言描述你期望的实验基准或对比目标\n例如:\n- 我希望结果比GPT-4准确率高10%\n- 响应时间应控制在2秒内\n- 需要保持90%以上的结果一致性"
        )
        
        # 保留自定义指标用于特殊情况
        st.subheader("自定义指标")
        for i, custom in enumerate(st.session_state.metrics_config["custom"]):
            st.text_input(f"指标名称 {i+1}", value=custom["name"])
            st.number_input(f"权重 {i+1}", min_value=0.0, max_value=1.0, value=custom["weight"], step=0.1)
            
        if st.button("➕ 添加自定义指标"):
            st.session_state.metrics_config["custom"].append({"name": "", "weight": 0.1})
            st.rerun()
            
        if st.button("💾 保存配置"):
            st.session_state.metrics_config.update({
                "custom": st.session_state.metrics_config["custom"]
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

# 初始化实验ID
if 'experiment_id' not in st.session_state:
    st.session_state.experiment_id = str(uuid.uuid4())
    saved_state = state_manager.load_state(st.session_state.experiment_id)
    if saved_state:
        st.session_state.update(saved_state)

os.environ["PYTHONPATH"] = os.getcwd()

def process_user_input(user_input: str) -> Dict[str, Any]:
    """使用目标解析器将用户输入转换为结构化实验目标格式"""
    if not user_input.strip():
        return {}
        
    # 默认使用UUID作为实验的唯一标识
    experiment_id = st.session_state.get('experiment_id', str(uuid.uuid4()))
    
    # 使用目标解析器解析用户输入
    parse_result = task_flow.parse_experimental_goal(user_input)
    
    if parse_result.get("status") == "success":
        # 使用结构化的实验目标
        parsed_goal = parse_result.get("parsed_goal", {})
        
        # 构建增强的任务对象
        task = {
            "goal": parsed_goal.get("goal", user_input),
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "benchmark": st.session_state.get('benchmark', ""),
            "structured_goal": parsed_goal  # 保存完整的结构化目标
        }
        
        # 如果解析结果包含任务类型，添加到任务中
        if "task_type" in parsed_goal:
            task["task_type"] = parsed_goal["task_type"]
            
        # 如果解析结果包含成功标准，添加到任务中
        if "success_criteria" in parsed_goal:
            task["success_criteria"] = parsed_goal["success_criteria"]
            
        # 如果解析结果包含资源限制，添加到任务中
        if "resources" in parsed_goal:
            task["resources"] = parsed_goal["resources"]
        
        # 显示解析结果
        st.session_state["parsed_goal"] = parsed_goal
    else:
        # 解析失败，使用原始方式
        logger.warning(f"目标解析失败: {parse_result.get('error', '未知错误')}，使用原始格式")
        task = {
            "goal": user_input,
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "benchmark": st.session_state.get('benchmark', "")
        }
    
    # 添加自定义指标
    custom_metrics_text = st.session_state.get('custom_metrics', "")
    if custom_metrics_text.strip():
        custom_metrics = [m.strip() for m in custom_metrics_text.split("\n") if m.strip()]
        task["custom_metrics"] = custom_metrics
        
    return task

def run_taskflow(task_input: Dict[str, Any], flow):
    """运行任务流并处理结果"""
    logger.info(f"收到任务输入: {task_input}")
    
    # 调试日志：打印所有注册的Agent
    if hasattr(flow, 'agent_manager') and hasattr(flow.agent_manager, 'agents'):
        logger.debug(f"当前注册Agent: {list(flow.agent_manager.agents.keys())}")
    
    # 创建进度显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    step_data = []
    agent_steps = list(flow.agent_manager.agents.keys())
    total_steps = len(agent_steps)
    
    # 添加执行日志记录
    if 'execution_logs' not in st.session_state:
        st.session_state.execution_logs = []
    
    try:
        # 记录开始时间
        start_time = datetime.now()
        
        # 设置进度回调函数
        def progress_callback(agent_name, step, total):
            progress = (step / total) if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"执行中: {agent_name} ({step}/{total})")
            step_data.append({
                'agent': agent_name,
                'timestamp': datetime.now().isoformat(),
                'progress': progress
            })
            
        # 运行任务流，传入回调函数
        result = flow.run_flow(task_input, progress_callback=progress_callback)
        
        # 更新完成进度
        progress_bar.progress(1.0)
        status_text.text("执行完成!")
        
        # 记录执行时间
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 处理结果并保存到会话状态
        if result.get('error'):
            logger.error(f"任务执行错误: {result['error']}")
            st.session_state.execution_logs.append({
                'timestamp': datetime.now().isoformat(),
                'action': '任务执行失败',
                'details': {'error': result['error'], 'execution_time': execution_time}
            })
            return None, result['error']
        else:
            # 处理结果并添加额外信息
            result['execution_time'] = execution_time
            result['timestamp'] = datetime.now().isoformat()
            result['step_data'] = step_data
            
            # 保存结果到会话状态用于历史查看
            st.session_state.last_result = result
            st.session_state.execution_logs.append({
                'timestamp': datetime.now().isoformat(),
                'action': '任务执行成功',
                'details': {
                    'execution_time': execution_time,
                    'result_summary': result.get('summary', '无摘要')
                }
            })
            
            logger.info(f"任务完成，结果长度: {len(str(result))}，执行时间: {execution_time:.2f}秒")
            return result, None
    except Exception as e:
        progress_bar.progress(1.0)
        status_text.text(f"执行出错: {str(e)}")
        logger.critical(f"系统异常: {str(e)}")
        st.session_state.execution_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': '系统异常',
            'details': {'error': str(e), 'traceback': traceback.format_exc()}
        })
        return None, str(e)

# ...

def show_agent_status(flow):
    """增强版状态显示"""
    if not hasattr(flow, 'agent_manager'):
        st.sidebar.error("⚠️ Agent管理器未加载")
        return
        
    status = flow.agent_manager.test_connections()
    if all(s.startswith("🔴") for s in status.values()):
        st.sidebar.error("⚠️ 所有Agent离线！")
        if st.sidebar.button("强制重连", key="force_reconnect"):
            try:
                with st.spinner("重新初始化中..."):
                    flow.agent_manager.start_collaboration()
                st.rerun()
            except Exception as e:
                st.error(f"重连失败: {str(e)}")
    else:
        # 显示每个Agent状态
        with st.sidebar.expander("👁️ Agent状态", expanded=False):
            for name, status_val in status.items():
                st.write(f"{name}: {status_val}")

show_agent_status(task_flow)

# 通过开始实验按钮进行实验执行
if st.session_state.get('start_experiment', False) and st.session_state.get('goal', '').strip():
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
        if st.button("🔄 尝试重新连接所有Agent", key="reconnect_all"):
            for agent in task_flow.agent_manager.agents.values():
                if hasattr(agent, 'connect'):
                    agent.connect()
            st.rerun()
            
    else:
        try:
            # 确保任务输入是字典
            task_input = process_user_input(st.session_state.get('goal', '').strip())
            result, error = run_taskflow(task_input, task_flow)
            
            if error:
                st.error(f"执行失败: {error}")
            else:
                if not result.get('output'):
                    st.warning("任务已执行但无输出结果，请检查Agent配置")
                else:
                    st.success("实验完成！")
                
                # 显示结构化实验目标
                if "structured_goal" in task_input:
                    st.subheader("🌀 解析后的结构化目标")
                    st.info("以下是系统解析后的结构化实验目标，展示了AI如何理解您的实验需求。")
                    
                    with st.container():
                        show_structured_goal(task_input["structured_goal"])
                        
                        # 如果实验带有增强的设计，显示增强前后对比
                        if result.get("debug", {}).get("enhanced_design", {}).get("enhancement_status") == "success":
                            st.subheader("🛠️ 方案增强结果")
                            enhancements = result.get("debug", {}).get("enhanced_design", {}).get("enhancement_details", [])
                            if enhancements:
                                for enhancement in enhancements:
                                    st.markdown(f"- {enhancement}")
                            else:
                                st.info("方案未进行明显改进或增强细节丢失")
                        
                        # 导出选项
                        export_structured_goal(task_input["structured_goal"])
                
                # 显示详细信息
                show_agent_debug_info(result)
                show_experiment_details(result)
                
                # 保存结果到会话状态
                st.session_state.result = result
                
                # 显示执行分析
                with st.expander("🔍 任务执行分析", expanded=False):
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
                        with st.expander("原始调试信息", expanded=False):
                            st.json(result['debug'])
        except Exception as e:
            st.error(f"系统错误: {str(e)}")
            st.exception(e)
else:
    st.info("请在上方输入实验目标，然后点击\"开始实验\"按钮。")
    
    # 显示结构化目标的演示
    if "parsed_goal" in st.session_state:
        st.subheader("🌀 目标解析器示例")
        st.info("目标解析器可以理解你的实验需求，自动提取关键信息，帮助系统更好地执行实验。")
        
        # 显示示例解析结果
        demo_goal = {
            "goal": "对猪肝存储蛋白进行活性测定",
            "task_type": "measurement",
            "domain": "生物化学",
            "confidence": 0.85,
            "success_criteria": {
                "primary_metric": {"name": "accuracy", "target_value": 0.95, "unit": "%"},
                "secondary_metrics": [{"name": "time_cost", "target_value": 30, "unit": "min"}]
            },
            "resources": {
                "required_instruments": ["分光光度计", "离心机", "水浴锅"],
                "time_limit": 3600,
            }
        }
        
        with st.container():
            show_structured_goal(demo_goal)
            export_structured_goal(st.session_state.get("parsed_goal", {}))
        
        # 提供编辑按钮
        if st.button("⚙️ 自定义实验设置"):
            st.session_state["show_edit_goal"] = True
            
    # 显示编辑面板
    if st.session_state.get("show_edit_goal", False):
        edited_goal = edit_structured_goal(st.session_state.get("parsed_goal", {}))
        if edited_goal:
            st.session_state["parsed_goal"] = edited_goal
            st.success("结构化目标已更新")
            st.session_state["show_edit_goal"] = False
            st.rerun()

# 定义函数才能使用
def show_agent_status(flow):
    """增强版状态显示"""
    if not hasattr(flow, 'agent_manager'):
        st.sidebar.error("⚠️ Agent管理器未加载")
        return
        
    status = flow.agent_manager.test_connections()
    if all(s.startswith("🔴") for s in status.values()):
        st.sidebar.error("⚠️ 所有Agent离线！")
        if st.sidebar.button("强制重连", key="force_reconnect"):
            try:
                with st.spinner("重新初始化中..."):
                    flow.agent_manager.start_collaboration()
                st.rerun()
            except Exception as e:
                st.error(f"重连失败: {str(e)}")
    else:
        # 显示每个Agent状态
        with st.sidebar.expander("👁️ Agent状态", expanded=False):
            for name, status_val in status.items():
                st.write(f"{name}: {status_val}")

def show_agent_debug_info(result):
    """显示Agent调试信息"""
    with st.expander("👁️ Agent调试信息", expanded=False):
        if 'agent_logs' in result:
            for agent_name, logs in result['agent_logs'].items():
                st.subheader(f"{agent_name} 响应日志")
                if isinstance(logs, list):
                    for i, log in enumerate(logs):
                        with st.expander(f"步骤 {i+1}", expanded=False):
                            st.write(log)
                else:
                    st.write(logs)
        else:
            st.info("没有可用的Agent调试信息")

def show_experiment_details(result):
    """显示实验详细信息"""
    with st.expander("🔍 实验详情", expanded=True):
        # 显示执行时间
        if 'execution_time' in result:
            st.info(f"执行时间: {result['execution_time']:.2f} 秒")
            
        # 显示实验输出
        st.subheader("📊 实验结果")
        if 'output' in result and result['output']:
            st.write(result['output'])
        else:
            st.warning("没有可用的实验结果")
            
        # 生成评估报告
        st.subheader("📝 评估报告")
        if 'metrics' in result:
            # 生成评估报告
            report = generate_diagnostic_report(result, st.session_state.metrics_config)
            
            # 显示失败指标
            if report['failed_metrics']:
                st.error("未通过的指标:")
                for metric in report['failed_metrics']:
                    st.write(f"- {metric['name']}: {metric['value']} (阈值: {metric['threshold']})")
            else:
                st.success("所有指标通过!")
                
            # 显示建议
            if report['suggestions']:
                st.subheader("💡 改进建议")
                for suggestion in report['suggestions']:
                    st.write(f"- {suggestion}")
        else:
            st.warning("没有指标数据可用")
            
        # 导出按钮
        if st.button("⬇️ 导出实验数据", key="export_data"):
            export_experiment()

def show_experiment_history():
    """显示实验历史记录"""
    with st.sidebar.expander("📂 实验历史", expanded=False):
        # 获取所有保存的实验记录
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

# 在侧边栏添加调试面板
with st.sidebar.expander("🧪 Agent测试", expanded=False):
    # Agent连接状态测试
    if st.button("测试Agent连接", key="test_connections"):
        if hasattr(task_flow, 'agent_manager'):
            try:
                test_result = task_flow.test_agent_connections()
                st.success("连接测试完成")
                st.json(test_result)
            except Exception as e:
                st.error(f"测试失败: {str(e)}")
        else:
            st.error("任务流对象没有agent_manager属性")

def show_agent_debug_info(result):
    """显示Agent调试信息"""
    with st.expander("👁️ Agent调试信息", expanded=False):
        if 'agent_logs' in result:
            for agent_name, logs in result['agent_logs'].items():
                st.subheader(f"{agent_name} 响应日志")
                if isinstance(logs, list):
                    for i, log in enumerate(logs):
                        with st.expander(f"步骤 {i+1}", expanded=False):
                            st.write(log)
                else:
                    st.write(logs)
        else:
            st.info("没有可用的Agent调试信息")

def show_experiment_details(result):
    """显示实验详细信息"""
    with st.expander("🔍 实验详情", expanded=True):
        # 显示执行时间
        if 'execution_time' in result:
            st.info(f"执行时间: {result['execution_time']:.2f} 秒")
            
        # 显示实验输出
        st.subheader("📊 实验结果")
        if 'output' in result and result['output']:
            st.write(result['output'])
        else:
            st.warning("没有可用的实验结果")
            
        # 生成评估报告
        st.subheader("📝 评估报告")
        if 'metrics' in result:
            # 生成评估报告
            report = generate_diagnostic_report(result, st.session_state.metrics_config)
            
            # 显示失败指标
            if report['failed_metrics']:
                st.error("未通过的指标:")
                for metric in report['failed_metrics']:
                    st.write(f"- {metric['name']}: {metric['value']} (阈值: {metric['threshold']})")
            else:
                st.success("所有指标通过!")
                
            # 显示建议
            if report['suggestions']:
                st.subheader("💡 改进建议")
                for suggestion in report['suggestions']:
                    st.write(f"- {suggestion}")
        else:
            st.warning("没有指标数据可用")
            
        # 导出按钮
        if st.button("⬇️ 导出实验数据", key="export_data"):
            export_experiment()

def show_experiment_history():
    """显示实验历史记录"""
    with st.sidebar.expander("📂 实验历史", expanded=False):
        # 获取所有保存的实验记录
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
    """在侧边栏显示执行日志"""
    with st.sidebar.expander("📜 执行日志", expanded=False):
        if 'execution_logs' in st.session_state and st.session_state.execution_logs:
            for log in reversed(st.session_state.execution_logs):
                st.markdown(f"**{log['timestamp']} - {log['action']}**")
                st.json(log['details'])
                st.markdown('---')
        else:
            st.info("暂无执行日志")

show_execution_logs()

# 自动修复选项
with st.sidebar.expander("🛠️ 故障修复", expanded=False):
    if st.button("🔄 强制进入模拟模式", key="force_mock_mode"):
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

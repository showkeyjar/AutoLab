import streamlit as st
import json
from typing import Dict, Any, List, Optional

def show_structured_goal(parsed_goal: Dict[str, Any]):
    """展示结构化的实验目标信息"""
    if not parsed_goal:
        return
        
    st.subheader("🎯 结构化实验目标")
    
    # 显示主要目标和任务类型
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**目标**: {parsed_goal.get('goal', '未指定')}")
    with col2:
        task_type = parsed_goal.get('task_type', '自定义')
        st.info(f"**任务类型**: {task_type}")
        
    # 显示领域信息（如果有）
    if "domain" in parsed_goal:
        st.info(f"**领域**: {parsed_goal['domain']}")
        
    # 显示信心水平
    if "confidence" in parsed_goal:
        confidence = parsed_goal["confidence"]
        color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
        st.markdown(f"**解析置信度**: <span style='color:{color}'>{confidence:.2f}</span>", unsafe_allow_html=True)
    
    # 成功标准
    if "success_criteria" in parsed_goal:
        st.markdown("### 📊 成功标准")
        
        criteria = parsed_goal["success_criteria"]
        
        # 主要指标
        if "primary_metric" in criteria:
            metric = criteria["primary_metric"]
            st.markdown(f"**主要指标**: {metric.get('name', '未命名')} = {metric.get('target_value', '未指定')} {metric.get('unit', '')}")
            
        # 次要指标
        if "secondary_metrics" in criteria and criteria["secondary_metrics"]:
            st.markdown("**次要指标**:")
            for metric in criteria["secondary_metrics"]:
                st.markdown(f"- {metric.get('name', '未命名')} = {metric.get('target_value', '未指定')} {metric.get('unit', '')}")
    
    # 资源需求
    if "resources" in parsed_goal:
        resources = parsed_goal["resources"]
        
        if resources:
            st.markdown("### 💻 资源需求")
            # 时间限制
            if "time_limit" in resources:
                minutes = resources["time_limit"] / 60
                st.info(f"预计耗时: {minutes:.1f} 分钟")
            
            # 所需仪器设备
            if "required_instruments" in resources and resources["required_instruments"]:
                st.markdown("**所需设备**:")
                for instrument in resources["required_instruments"]:
                    st.markdown(f"- {instrument}")
                    
            # 计算资源
            if "computational_requirements" in resources:
                st.info(f"计算资源: {resources['computational_requirements']}")
                
            st.markdown("---")  # 添加分隔线，增强视觉分隔
    
    # 约束条件
    if "constraints" in parsed_goal and parsed_goal["constraints"]:
        st.markdown("### ⚠️ 约束条件")
        for constraint in parsed_goal["constraints"]:
            st.markdown(f"- {constraint}")
        st.markdown("---")
    
    # 期望输出
    if "expected_output" in parsed_goal:
        st.markdown("### 🔍 期望输出")
        st.markdown(parsed_goal["expected_output"])
        st.markdown("---")
            
def goal_parser_interactive_callback(field_info: Dict[str, Any]) -> Any:
    """为交互式目标解析器实现的回调函数"""
    field = field_info.get("field", "")
    question = field_info.get("question", "")
    options = field_info.get("options", [])
    
    st.subheader("请补充信息")
    st.info(question)
    
    if field == "task_type" and options:
        return st.selectbox("选择任务类型", options=options)
    elif field == "success_criteria":
        metric_name = st.selectbox("指标名称", options=options if options else ["accuracy", "precision", "recall", "f1_score", "time_cost"])
        target_value = st.slider("目标值", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
        return {"primary_metric": {"name": metric_name, "target_value": target_value}}
    
    # 默认返回None
    return None

def export_structured_goal(parsed_goal: Dict[str, Any]):
    """导出结构化目标为JSON文件"""
    if not parsed_goal:
        return
        
    # 去除不需要导出的元数据
    export_goal = {k: v for k, v in parsed_goal.items() if k not in ["raw_input", "parsed_timestamp"]}
    
    # 转换为JSON字符串并提供下载
    json_str = json.dumps(export_goal, indent=2, ensure_ascii=False)
    st.download_button(
        label="⬇️ 下载结构化目标",
        data=json_str,
        file_name="structured_goal.json",
        mime="application/json"
    )

def edit_structured_goal(parsed_goal: Dict[str, Any]) -> Dict[str, Any]:
    """允许用户编辑结构化目标"""
    if not parsed_goal:
        return {}
        
    edited_goal = parsed_goal.copy()
    
    with st.form("edit_structured_goal"):
        st.subheader("编辑结构化目标")
        
        # 编辑主要目标
        edited_goal["goal"] = st.text_area("实验目标", value=parsed_goal.get("goal", ""))
        
        # 编辑任务类型
        task_types = ["classification", "regression", "optimization", "synthesis", 
                      "measurement", "simulation", "data_analysis", "benchmark", "custom"]
        edited_goal["task_type"] = st.selectbox(
            "任务类型", 
            options=task_types,
            index=task_types.index(parsed_goal.get("task_type", "custom")) if parsed_goal.get("task_type") in task_types else -1
        )
        
        # 编辑成功标准
        st.subheader("成功标准")
        criteria = parsed_goal.get("success_criteria", {})
        primary_metric = criteria.get("primary_metric", {"name": "accuracy", "target_value": 0.8})
        
        metric_name = st.text_input("主要指标名称", value=primary_metric.get("name", ""))
        metric_value = st.number_input(
            "目标值", 
            value=float(primary_metric.get("target_value", 0.8)),
            min_value=0.0,
            max_value=100.0,
            step=0.1
        )
        
        # 更新编辑后的成功标准
        edited_goal["success_criteria"] = {
            "primary_metric": {
                "name": metric_name,
                "target_value": metric_value
            }
        }
        
        # 提交按钮
        submit = st.form_submit_button("保存修改")
        
    if submit:
        return edited_goal
    
    return None

# 导出组件函数
__all__ = ["show_structured_goal", "goal_parser_interactive_callback", "export_structured_goal", "edit_structured_goal"]

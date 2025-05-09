<div align="center">

# 🧪 AutoLab

**智能自主实验系统 | 多智能体驱动 | 先进实验自动化平台**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)](https://streamlit.io/)

*利用AI协作智能体，自动化实验设计、执行和分析*

<p align="center">
<img src="https://github.com/knifecms/AutoLab/raw/main/docs/assets/autolab-banner.png" alt="AutoLab Banner" width="600"/>
</p>

</div>

## ✨ 主要特性

- 🤖 **多智能体协作系统** - 多个专业智能体共同完成复杂实验任务
- 🔍 **结构化目标解析** - 自动解析自然语言实验目标
- 🔄 **Plan-Enhance-Execute架构** - 三阶段任务执行与优化流程
- 📊 **实时数据可视化** - 直观监控实验进展与结果
- 🧠 **自适应实验优化** - 基于历史数据优化实验参数
- 🧩 **模块化设计** - 易于扩展的插件式架构

## 🛠️ 安装与环境配置

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/knifecms/AutoLab.git
   cd AutoLab
   ```

2. 安装依赖：
   ```bash
   pip install -e .
   ```

## 🔧 环境设置

1. 创建环境变量配置：
   ```bash
   cp .env.example .env
   ```

2. 编辑`.env`文件设置你的个人配置

3. 配置Python环境变量（必需步骤）：
   ```bash
   # Linux/Mac
   export PYTHONPATH=$(pwd)
   
   # Windows (CMD)
   set PYTHONPATH=%cd%
   
   # Windows (PowerShell)
   $env:PYTHONPATH=(Get-Location).Path
   ```

## 🚀 使用指南

### 启动应用

安装所需依赖后，运行以下命令启动Streamlit应用：

```bash
# 首先设置PYTHONPATH
# Windows (PowerShell)
$env:PYTHONPATH=(Get-Location).Path

# 然后运行Streamlit应用
streamlit run autolab/ui/streamlit_app.py
```

打开浏览器访问：[http://localhost:8501](http://localhost:8501)

### 可选：启动API后端（高级功能）

如果需要使用API功能，可以另外启动FastAPI后端：

```bash
# 设置PYTHONPATH后运行
$env:PYTHONPATH=(Get-Location).Path; uvicorn autolab.server.main:app --reload
```

API文档访问地址：[http://localhost:8000/docs](http://localhost:8000/docs)

### 系统功能

通过网页界面，您可以：

- **创建实验**：定义实验目标和参数
- **运行实验**：执行定义的实验协议
- **监控进展**：实时追踪实验过程
- **查看结果**：分析实验数据和生成报告
- **导出数据**：保存实验结果以便后续分析

## ⚙️ 系统配置

### 主要配置文件

- `config/agents.yaml`: 智能体行为参数配置
- `config/equipment.yaml`: 实验设备配置
- `config/experiments/`: 预定义实验协议
- `config/agent_tools.yaml`: 智能体工具配置
- `config/ui_config.yaml`: 可视化界面设置

### 智能架构

AutoLab采用先进的**Plan-Enhance-Execute**智能体架构，包含三个核心阶段：

1. **规划阶段 (Plan)**
   - 生成初始实验设计
   - 解析实验目标和需求
   - 定义成功标准

2. **增强阶段 (Enhance)**
   - 对初始设计进行优化
   - 基于历史数据调整参数
   - 预测并规避潜在问题

3. **执行阶段 (Execute)**
   - 实施优化后的实验方案
   - 实时监控进展和调整
   - 收集和分析数据

### 结构化目标解析

AutoLab集成了智能的实验目标解析器，可以：

- 将自然语言实验描述转换为结构化格式
- 自动识别任务类型、领域专业和关键指标
- 定义明确的成功标准和所需资源
- 提供交互式界面进行目标微调

这种方法借鉴了Kaggle竹赛项目架构，大大提高了实验需求的明确度和执行成功率。

## 📚 Experiment Templates

AutoLab provides a powerful template system for saving and reusing experiment configurations.

### Using Templates

1. Select "📁 Experiment Templates" in the sidebar
2. Choose an existing template from the dropdown menu
3. Click the "Use Template" button to apply it to your experiment

### Creating New Templates

1. Select "Create New Template" from the template operations
2. Enter a descriptive name for your template
3. Fill in the configuration in JSON format (example):
   ```json
   {
     "metrics": {
       "accuracy_threshold": 0.85,
       "efficiency_threshold": 0.8
     },
     "llm": {
       "model": "llama3",
       "temperature": 0.7
     }
   }
   ```
4. Click "Save" to store your template

### Editing Templates

1. Select "Edit Template" from the template operations
2. Modify the JSON configuration as needed
3. Click "Update Template" to save your changes

Template files are stored in the `config/templates/` directory and can be managed directly as YAML files.

## 📊 Evaluation Standards

AutoLab uses a flexible evaluation system defined in `config/evaluation_config.yaml` with these key components:

### Metrics Configuration

- **Primary Metrics**: Define critical success factors
  - `weight`: Importance coefficient (0.0-1.0)
  - `threshold`: Minimum acceptable value
  - `description`: Detailed explanation

### Scoring System

- **Pass Criteria**: Configurable success thresholds
  - `passing_score`: Minimum passing score
  - `penalty`: Deductions for various issues

### Evaluation Methods

- **Automatic**: AI-driven assessment based on objective criteria
- **Manual**: Human review for subjective evaluation

Example configuration:
```yaml
metrics:
  accuracy:
    weight: 0.5
    threshold: 0.85

evaluation_methods:
  - name: "auto"
```

## Documentation

For detailed API documentation and advanced usage, see:
- [API Reference](docs/api.md)
- [Developer Guide](docs/developer.md)

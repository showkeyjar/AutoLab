# AutoLab
A multi-agent-driven robotic AI autonomous laboratory for research

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/knifecms/AutoLab.git
   cd AutoLab
   ```
2. Install as editable package:
   ```
   pip install -e .
   ```
3. Run the application:
   ```
   streamlit run autolab/ui/streamlit_app.py
   ```

## Setup

1. Configure environment variables by creating a `.env` file:
   ```
   cp .env.example .env
   ```
2. Edit the `.env` file with your specific configurations

## Usage

### 1. Start the Backend (FastAPI)
```
uvicorn autolab.main:app --reload
```
Access API docs at: http://localhost:8000/docs

### 2. Start the Frontend (Streamlit)
Open a new terminal and run:
```
streamlit run autolab/ui/streamlit_app.py
```
Access UI at: http://localhost:8501

### Combined Development Command (PowerShell)
```
$env:PYTHONPATH=(Get-Location).Path; uvicorn autolab.main:app --reload &
streamlit run autolab/ui/streamlit_app.py
```

### Basic Commands

- `start`: Initialize all agents and equipment
- `run_experiment`: Execute a predefined experiment protocol
- `monitor`: View real-time system status
- `shutdown`: Safely terminate all processes

## Configuration

Key configuration files:
- `config/agents.yaml`: Agent behavior parameters
- `config/equipment.yaml`: Laboratory equipment settings
- `config/experiments/`: Predefined experiment protocols
- `config/agent_tools.yaml`: Browser and MCP tool configurations
- `config/ui_config.yaml`: Visualization interface settings

### Experiment Targets

The system allows configuration of experiment objectives through:
1. Predefined default objectives
2. Custom objective definitions
3. Visual parameter editing

Configure these in `config/agent_tools.yaml` under the `experiment_targets` section.

## 实验模板管理

AutoLab提供实验模板功能，可保存和复用常用实验配置。

### 使用模板
1. 在Web界面左侧边栏选择"📁 实验模板"
2. 从下拉菜单中选择已有模板
3. 点击"使用模板"按钮

### 创建新模板
1. 在模板操作中选择"新建模板"
2. 输入模板名称
3. 填写JSON格式的配置（示例）：
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
4. 点击"保存"按钮

### 编辑模板
1. 在模板操作中选择"编辑模板"
2. 修改JSON配置内容
3. 点击"更新模板"按钮

模板文件存储在`config/templates/`目录下，可直接管理YAML文件。

## 实验评价标准

评价配置在`config/evaluation_config.yaml`中定义，包含：

1. **指标配置**
   - `weight`: 权重系数
   - `threshold`: 达标阈值
   - `description`: 指标说明

2. **评分规则**
   - `passing_score`: 合格分数线
   - `penalty`: 各类扣分项

3. **评价方式**
   - `auto`: 自动评分
   - `manual`: 人工评审

示例配置：
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

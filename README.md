# AutoLab
A multi-agent-driven robotic AI autonomous laboratory for research

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/knifecms/AutoLab.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
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

## Documentation

For detailed API documentation and advanced usage, see:
- [API Reference](docs/api.md)
- [Developer Guide](docs/developer.md)

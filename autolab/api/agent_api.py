from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from autolab.core.agent_manager import AgentManager
from autolab.core.task_flow import TaskFlow

router = APIRouter()
agent_manager = AgentManager()
task_flow = TaskFlow()

class TaskRequest(BaseModel):
    task: dict

@router.post("/agent/{agent_name}/task")
def send_task(agent_name: str, req: TaskRequest):
    try:
        result = agent_manager.dispatch(agent_name, req.task)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/taskflow/run")
def run_task_flow(req: TaskRequest):
    """
    端到端任务流转示例：依次调用五大智能体，返回各阶段结果。
    """
    result = task_flow.run_flow(req.task)
    return {"flow_result": result}

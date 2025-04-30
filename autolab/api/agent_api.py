from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from autolab.core.agent_manager import AgentManager

router = APIRouter()
agent_manager = AgentManager()

class TaskRequest(BaseModel):
    task: dict

@router.post("/agent/{agent_name}/task")
def send_task(agent_name: str, req: TaskRequest):
    try:
        result = agent_manager.dispatch(agent_name, req.task)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

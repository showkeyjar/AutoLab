from fastapi import FastAPI
from autolab.api.agent_api import router as agent_router

app = FastAPI(title="AutoLab 多智能体自动实验系统")

app.include_router(agent_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to AutoLab Multi-Agent Experimental System!"}

# 预留：后续可在此注册各智能体、API路由等

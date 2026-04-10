import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CloudCostEnv
from models import Observation, Action

app = FastAPI(title="Cloud Cost Architect UI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_env = None
current_task = "easy"

class StepResponse(BaseModel):
    state: Observation
    reward: dict
    done: bool

@app.get("/")
def get_root():
    return {"status": "ok", "message": "Cloud Cost Environment API is running. Access /docs for the API interface."}

class ResetRequest(BaseModel):
    task: str = "easy" 

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest):
    global current_env, current_task
    task_file = f"tasks/{req.task}.json"
    try:
        with open(task_file, "r") as f:
            config = json.load(f)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Task config not found for {req.task}")
    
    current_task = req.task
    current_env = CloudCostEnv(config)
    obs = current_env.reset()
    return obs

@app.get("/state", response_model=Observation)
def get_state():
    global current_env
    if not current_env:
        reset_env(ResetRequest(task="easy"))
    return current_env.state()

@app.post("/step", response_model=StepResponse)
def take_action(action: Action):
    global current_env
    if not current_env:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    if current_env.done:
        return StepResponse(state=current_env.state(), reward={"score": 0.0, "message": "Episode already done"}, done=True)
        
    obs, reward, done, info = current_env.step(action)
    return StepResponse(state=obs, reward=reward.model_dump(), done=done)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=True)

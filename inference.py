import os
import json
import logging
from openai import OpenAI
from env import CloudCostEnv
from models import Action

# Keep standard logging minimal so we don't pollute the requested output format
logging.basicConfig(level=logging.ERROR)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = HF_TOKEN or os.getenv("API_KEY", "dummy_key_not_checked")

# If testing locally without an openai server, this would just fail. The environment
# expects a standard openai compliant server available at API_BASE_URL.
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def main():
    task_file = os.environ.get("TASK_FILE", "tasks/easy.json")
    task_name = os.environ.get("TASK_NAME", "easy")
    env_name = "CloudCostArchitect"
    
    with open(task_file, "r") as f:
        config = json.load(f)
        
    env = CloudCostEnv(config)
    obs = env.reset()
    
    log_start(task=task_name, env=env_name, model=MODEL_NAME)

    messages = [
        {
            "role": "system", 
            "content": "You are a Cloud Cost Architect. Output your action strictly in JSON format: {'command': 'PROVISION' or 'TERMINATE' or 'WAIT' or 'SUBMIT', 'server_id': 'string_id'}."
        }
    ]
    
    step_count = 0
    rewards_history = []
    
    while not env.done:
        step_count += 1
        messages.append({"role": "user", "content": f"Current Observation: {obs.model_dump_json()}"})
        
        error_msg = None
        action_str = ""
        action_text = ""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0
            )
            action_text = response.choices[0].message.content
            start_idx = action_text.find('{')
            end_idx = action_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                clean_json_str = action_text[start_idx:end_idx].replace("'", '"')
                action_json = json.loads(clean_json_str)
                action = Action(**action_json)
                action_str = clean_json_str
            else:
                action = Action(command="WAIT")
                action_str = '{"command": "WAIT"}'
                
            messages.append({"role": "assistant", "content": action_text})
        except Exception as e:
            error_msg = str(e)
            action = Action(command="WAIT")
            action_str = '{"command": "WAIT"}'

        obs, reward_obj, env.done, info = env.step(action)
        reward_float = reward_obj.score
        
        rewards_history.append(reward_float)
        
        log_step(step=step_count, action=action_str, reward=reward_float, done=env.done, error=error_msg)

    # Use grader to get total score between 0 and 1
    from graders import _compute_score
    final_score = _compute_score(env)
    
    success = final_score > 0.5
    log_end(success=success, steps=step_count, score=final_score, rewards=rewards_history)

if __name__ == "__main__":
    main()

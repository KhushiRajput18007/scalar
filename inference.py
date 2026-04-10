import os
import json
import logging
from openai import OpenAI
from env import CloudCostEnv
from models import Action

# Keep standard logging minimal so we don't pollute the requested output format
logging.basicConfig(level=logging.ERROR)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy_key_not_checked")

# If testing locally without an openai server, this would just fail. The environment
# expects a standard openai compliant server available at API_BASE_URL.
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def main():
    print("[START] Starting inference run", flush=True)
    
    task_file = os.environ.get("TASK_FILE", "tasks/easy.json")
    with open(task_file, "r") as f:
        config = json.load(f)
        
    env = CloudCostEnv(config)
    obs = env.reset()
    
    # Prompt the model with the task definition
    messages = [
        {
            "role": "system", 
            "content": "You are a Cloud Cost Architect. Your goal is to meet the workload requirement under budget. "
                       "Output your action strictly in JSON format matching this schema: "
                       "{'command': 'PROVISION' or 'TERMINATE' or 'WAIT' or 'SUBMIT', 'server_id': 'string_id'}. "
                       "Use 'SUBMIT' when you are done."
        }
    ]
    
    while not env.done:
        print(f"[STEP] Current observation: {obs.model_dump_json()}", flush=True)
        messages.append({"role": "user", "content": f"Current Observation: {obs.model_dump_json()}"})
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0
            )
            
            action_text = response.choices[0].message.content
            print(f"[STEP] Model raw action: {action_text}", flush=True)
            
            # Simple heuristic JSON parsing from output
            start_idx = action_text.find('{')
            end_idx = action_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                action_json = json.loads(action_text[start_idx:end_idx])
                action = Action(**action_json)
            else:
                action = Action(command="WAIT") # Default fallback
                
            messages.append({"role": "assistant", "content": action_text})
        except Exception as e:
            print(f"[STEP] Error parsing action: {e}", flush=True)
            action = Action(command="WAIT")

        print(f"[STEP] Parsed Action: {action.model_dump_json()}", flush=True)
        obs, reward, env.done, info = env.step(action)
        print(f"[STEP] Reward: {reward.model_dump_json()}", flush=True)

    print("[END] Inference completed", flush=True)

if __name__ == "__main__":
    main()

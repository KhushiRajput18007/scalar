import logging
import copy
from typing import Dict, Any, Tuple
from models import Observation, Action, Reward, ServerDef, WorkloadReq

# Initialize logger
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class CloudCostEnv:
    def __init__(self, task_config: Dict[str, Any]):
        self.task_config = task_config
        self.max_steps = task_config.get("max_steps", 10)
        self.current_step = 0
        self.done = False
        
        self.marketplace = [ServerDef(**s) for s in task_config.get("marketplace", [])]
        self.active_servers = [ServerDef(**s) for s in task_config.get("initial_active_servers", [])]
        self.remaining_budget_dollars = task_config.get("initial_budget_dollars", 1000.0)
        self.workload = WorkloadReq(**task_config.get("workload_req", {"min_total_ram_gb": 0, "min_total_cpu_cores": 0}))
        self.event_message = task_config.get("initial_event", None)
        
        # specific hard task mechanic
        self.price_spike_events = task_config.get("price_spike_events", {})

    def reset(self) -> Observation:
        self.current_step = 0
        self.done = False
        return self.state()

    def state(self) -> Observation:
        return Observation(
            marketplace=copy.deepcopy(self.marketplace),
            active_servers=copy.deepcopy(self.active_servers),
            remaining_budget_dollars=self.remaining_budget_dollars,
            workload=copy.deepcopy(self.workload),
            event_message=self.event_message
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            return self.state(), Reward(score=0.0, message="Episode already done"), self.done, {}

        self.current_step += 1
        reward = Reward(score=0.0, message="Step taken.")
        info = {}

        # Handle events for this step first
        step_str = str(self.current_step)
        if step_str in self.price_spike_events:
            self.event_message = self.price_spike_events[step_str].get("message", "A price spike occurred!")
            new_prices = self.price_spike_events[step_str].get("new_prices", {})
            for server in self.active_servers:
                if server.server_id in new_prices:
                    server.price_per_month = new_prices[server.server_id]
            for server in self.marketplace:
                if server.server_id in new_prices:
                    server.price_per_month = new_prices[server.server_id]
            
            reward.message += " Price spike event active."

        # Process action
        if action.command == "PROVISION" and action.server_id:
            server_to_provision = next((s for s in self.marketplace if s.server_id == action.server_id), None)
            if server_to_provision:
                # Add to active servers
                self.active_servers.append(copy.deepcopy(server_to_provision))
                self.remaining_budget_dollars -= server_to_provision.price_per_month
                reward.score += 0.1
                reward.message += f" Successfully provisioned {action.server_id}."
            else:
                reward.score -= 0.1
                reward.message += f" Failed to provision: {action.server_id} not in marketplace."

        elif action.command == "TERMINATE" and action.server_id:
            server_idx = next((i for i, s in enumerate(self.active_servers) if s.server_id == action.server_id), -1)
            if server_idx != -1:
                terminated = self.active_servers.pop(server_idx)
                reward.score += 0.2
                reward.message += f" Terminated {action.server_id}."
            else:
                reward.score -= 0.1
                reward.message += f" Failed to terminate: {action.server_id} not active."

        elif action.command == "WAIT":
             reward.score += 0.05
             reward.message += " Wait action taken."

        elif action.command == "SUBMIT":
            self.done = True
            reward.message += " Submitted final configuration."

        # Give partial credit based on fulfilling requirements
        total_ram = sum(s.ram_gb for s in self.active_servers)
        total_cpu = sum(s.cpu_cores for s in self.active_servers)
        if total_ram >= self.workload.min_total_ram_gb and total_cpu >= self.workload.min_total_cpu_cores:
            reward.score += 0.3
            reward.message += " Workload requirements met."

        # Terminate early conditionally
        if self.current_step >= self.max_steps:
            self.done = True

        return self.state(), reward, self.done, info

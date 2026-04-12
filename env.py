import logging
import copy
import random
from typing import Dict, Any, Tuple, List, Optional
from models import Observation, Action, Reward, ServerDef, WorkloadReq

# Initialize logger
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class CloudCostEnv:
    def __init__(self, task_config: Dict[str, Any]):
        self.task_config = task_config
        self.max_steps = task_config.get("max_steps", 10)
        self.initial_budget = task_config.get("initial_budget_dollars", 1000.0)
        
        # Tracking attributes matched to models.py Observation extension
        self.step_count = 0
        self.previous_actions: List[str] = []
        self.previous_costs: List[float] = []
        self.current_event: Optional[str] = None
        
        self.done = False
        
        # Initialize marketplace and active servers
        self.marketplace = [ServerDef(**s) for s in task_config.get("marketplace", [])]
        self.active_servers = [ServerDef(**s) for s in task_config.get("initial_active_servers", [])]
        self.remaining_budget_dollars = self.initial_budget
        self.workload = WorkloadReq(**task_config.get("workload_req", {"min_total_ram_gb": 0, "min_total_cpu_cores": 0}))
        self.event_message = task_config.get("initial_event", None)
        
        # Keep deterministic seed for reproducibility
        random.seed(42)
        
        # Legacy config-based events
        self.price_spike_events = task_config.get("price_spike_events", {})

    def reset(self) -> Observation:
        self.step_count = 0
        self.done = False
        self.previous_actions = []
        self.previous_costs = []
        self.current_event = None
        
        # Re-initialize to reset any dynamic changes
        self.marketplace = [ServerDef(**s) for s in self.task_config.get("marketplace", [])]
        self.active_servers = [ServerDef(**s) for s in self.task_config.get("initial_active_servers", [])]
        self.remaining_budget_dollars = self.initial_budget
        self.workload = WorkloadReq(**self.task_config.get("workload_req", {"min_total_ram_gb": 0, "min_total_cpu_cores": 0}))
        
        random.seed(42)
        return self.state()

    def state(self) -> Observation:
        return Observation(
            marketplace=copy.deepcopy(self.marketplace),
            active_servers=copy.deepcopy(self.active_servers),
            remaining_budget_dollars=self.remaining_budget_dollars,
            workload=copy.deepcopy(self.workload),
            event_message=self.event_message,
            previous_actions=copy.deepcopy(self.previous_actions),
            previous_costs=copy.deepcopy(self.previous_costs),
            current_event=self.current_event,
            step_count=self.step_count
        )

    def _apply_random_event(self):
        """Helper to apply a random world event to the environment (Stress Mode)."""
        events = ["price_spike", "traffic_spike", "server_failure", "budget_cut"]
        self.current_event = random.choice(events)
        
        if self.current_event == "price_spike":
            self.event_message = f"STRESS EVENT (Step {self.step_count}): Price Spike! Running costs doubled."
            for s in self.active_servers + self.marketplace:
                s.running_cost *= 2.0
        
        elif self.current_event == "traffic_spike":
            self.event_message = f"STRESS EVENT (Step {self.step_count}): Traffic Spike! Requirements increased."
            self.workload.min_total_ram_gb = int(self.workload.min_total_ram_gb * 1.5)
            self.workload.min_total_cpu_cores = int(self.workload.min_total_cpu_cores * 1.5)
        
        elif self.current_event == "server_failure":
            if self.active_servers:
                failed_server = random.choice(self.active_servers)
                self.active_servers.remove(failed_server)
                self.event_message = f"STRESS EVENT (Step {self.step_count}): Server Failure! {failed_server.server_id} offline."
            else:
                self.event_message = f"STRESS EVENT (Step {self.step_count}): Attempted Server Failure (but none active)."
        
        elif self.current_event == "budget_cut":
            reduction = self.remaining_budget_dollars * 0.2
            self.remaining_budget_dollars -= reduction
            self.event_message = f"STRESS EVENT (Step {self.step_count}): Budget Cut! Lost ${reduction:.2f}."

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            return self.state(), Reward(score=0.0, message="Episode already done"), self.done, {}

        self.step_count += 1
        reward_obj = Reward(score=0.0, message="Step taken.") 
        
        # 1. Action and Running Cost System
        action_costs = {"PROVISION": 2.0, "TERMINATE": 1.0, "WAIT": 0.0, "SUBMIT": 0.0}
        a_cost = action_costs.get(action.command, 0.0)
        r_cost = sum(s.running_cost for s in self.active_servers)
        
        total_step_cost = a_cost + r_cost
        self.remaining_budget_dollars -= total_step_cost
        
        # Tracking
        self.previous_actions.append(action.command)
        self.previous_costs.append(total_step_cost)

        # 2. Stress Mode: Trigger random event every 3 steps (plus the original step 5 hook)
        if self.step_count % 3 == 0 or self.step_count == 5:
            self._apply_random_event()
        else:
            # Clear event string if no event active this step for observation clarity
            self.current_event = None

        # 3. Process action logic
        if action.command == "PROVISION" and action.server_id:
            server_to_provision = next((s for s in self.marketplace if s.server_id == action.server_id), None)
            if server_to_provision:
                self.active_servers.append(copy.deepcopy(server_to_provision))
                self.remaining_budget_dollars -= server_to_provision.price_per_month
                reward_obj.message += f" Provisioned {action.server_id}."
            else:
                reward_obj.message += f" Failed provision: {action.server_id}."

        elif action.command == "TERMINATE" and action.server_id:
            server_idx = next((i for i, s in enumerate(self.active_servers) if s.server_id == action.server_id), -1)
            if server_idx != -1:
                self.active_servers.pop(server_idx)
                reward_obj.message += f" Terminated {action.server_id}."
            else:
                reward_obj.message += f" Failed terminate: {action.server_id}."

        elif action.command == "WAIT":
             reward_obj.message += " Wait action."

        elif action.command == "SUBMIT":
            self.done = True
            reward_obj.message += " Submitted."

        # 4. Requirement Calculation
        current_ram = sum(s.ram_gb for s in self.active_servers)
        current_cpu = sum(s.cpu_cores for s in self.active_servers)
        required_ram = self.workload.min_total_ram_gb
        required_cpu = self.workload.min_total_cpu_cores
        
        # 5. Reward Shaping Logic
        ram_ratio = min(current_ram / required_ram, 1.0) if required_ram > 0 else 1.0
        cpu_ratio = min(current_cpu / required_cpu, 1.0) if required_cpu > 0 else 1.0
        budget_ratio = max(self.remaining_budget_dollars / self.initial_budget, 0.0)
        
        final_reward = (cpu_ratio * 0.3) + (ram_ratio * 0.3) + (budget_ratio * 0.4)
        if self.remaining_budget_dollars < 0:
            final_reward -= 1.0
            
        reward_obj.score = max(min(final_reward, 1.0), -1.0)
        
        if current_ram >= required_ram and current_cpu >= required_cpu:
            reward_obj.message += " Requirements met."
        else:
            reward_obj.message += " Requirements NOT met."

        # 6. Termination check
        if self.step_count >= self.max_steps:
            self.done = True

        # 7. Step Return Info
        info = {
            "event": self.current_event,
            "budget_remaining": self.remaining_budget_dollars,
            "active_servers": len(self.active_servers),
            "action": action,
            "running_cost": r_cost
        }

        return self.state(), reward_obj, self.done, info

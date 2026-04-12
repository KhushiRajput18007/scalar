from typing import Literal, List, Optional
from pydantic import BaseModel, Field

class ServerDef(BaseModel):
    server_id: str
    ram_gb: int
    cpu_cores: int
    price_per_month: float
    
    # Extended attributes with defaults for backward compatibility
    latency: float = 0.0
    availability: float = 1.0
    region: str = "us-east"
    startup_time: int = 1
    running_cost: float = 1.0

class WorkloadReq(BaseModel):
    min_total_ram_gb: int
    min_total_cpu_cores: int

class Observation(BaseModel):
    marketplace: List[ServerDef] = Field(..., description="Available servers to provision.")
    active_servers: List[ServerDef] = Field(..., description="Currently active servers in your fleet.")
    remaining_budget_dollars: float = Field(..., description="Your remaining budget.")
    workload: WorkloadReq = Field(..., description="Current system workload requirements.")
    event_message: Optional[str] = Field(None, description="System event notifications, if any.")
    
    # Extended tracking attributes with defaults
    previous_actions: List[str] = []
    previous_costs: List[float] = []
    current_event: Optional[str] = None
    step_count: int = 0

class Action(BaseModel):
    command: Literal["PROVISION", "TERMINATE", "WAIT", "SUBMIT"] = Field(..., description="The command to execute.")
    server_id: Optional[str] = Field(None, description="The ID of the server to PROVISION or TERMINATE. Must be valid from marketplace.")

class Reward(BaseModel):
    score: float = Field(..., description="The step reward value.")
    message: str = Field(..., description="Explanation of the reward signal.")

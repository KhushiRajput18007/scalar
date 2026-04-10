import json
from typing import Dict
from env import CloudCostEnv
from models import Observation, Reward

def _compute_score(env: CloudCostEnv) -> float:
    """Compute a deterministic score between 0.0 and 1.0 based on final state.
    The score rewards meeting workload requirements and staying within budget.
    """
    total_ram = sum(s.ram_gb for s in env.active_servers)
    total_cpu = sum(s.cpu_cores for s in env.active_servers)
    workload_met = (
        total_ram >= env.workload.min_total_ram_gb and
        total_cpu >= env.workload.min_total_cpu_cores
    )
    budget_ok = env.remaining_budget_dollars >= 0
    # Simple linear combination
    score = 0.0
    if workload_met:
        score += 0.6
    if budget_ok:
        score += 0.3
    # Small bonus for using fewer servers than max allowed (encourage efficiency)
    if len(env.active_servers) <= 3:
        score += 0.1
    if score <= 0.0:
        score = 0.01
    if score >= 1.0:
        score = 0.99
    return float(score)

def grade_easy_task(env: CloudCostEnv) -> float:
    """Grader for the easy task.
    Returns a float score between 0.0 and 1.0.
    """
    return _compute_score(env)

def grade_medium_task(env: CloudCostEnv) -> float:
    """Grader for the medium task.
    Returns a float score between 0.0 and 1.0.
    """
    return _compute_score(env)

def grade_hard_task(env: CloudCostEnv) -> float:
    """Grader for the hard task.
    Returns a float score between 0.0 and 1.0.
    """
    return _compute_score(env)

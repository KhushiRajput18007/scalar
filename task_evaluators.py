import json
from typing import Dict


def _compute_score(env):
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
    if workload_met and budget_ok:
        return 0.99
    else:
        return 0.01

def grade_easy_task(env):
    """Grader for the easy task.
    Returns a score between 0.0 and 1.0.
    """
    return _compute_score(env)

def grade_medium_task(env):
    """Grader for the medium task.
    Returns a score between 0.0 and 1.0.
    """
    return _compute_score(env)

def grade_hard_task(env):
    """Grader for the hard task.
    Returns a score between 0.0 and 1.0.
    """
    return _compute_score(env)

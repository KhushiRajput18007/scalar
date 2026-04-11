def _compute_score(env):
    try:
        total_ram = sum(s.ram_gb for s in env.active_servers)
        total_cpu = sum(s.cpu_cores for s in env.active_servers)
        
        workload_met = (
            total_ram >= env.workload.min_total_ram_gb and
            total_cpu >= env.workload.min_total_cpu_cores
        )
        budget_ok = env.remaining_budget_dollars >= 0
        
        return 0.99 if (workload_met and budget_ok) else 0.01
    except Exception:
        return 0.01

def grade_easy_task(env): return _compute_score(env)
def grade_medium_task(env): return _compute_score(env)
def grade_hard_task(env): return _compute_score(env)

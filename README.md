---
title: Cloud Cost Architect RL
emoji: ☁️
colorFrom: cyan
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# ☁️ Cloud Cost Architect RL

An advanced Reinforcement Learning environment designed to train agents as **Cloud Cost Architects**. The goal is to manage a fleet of cloud servers, balancing high-availability requirements with strict budget constraints in a dynamic, unpredictable marketplace.

## 🔗 Links
- **GitHub Repository**: [KhushiRajput18007/scalar](https://github.com/KhushiRajput18007/scalar)
- **Live Demo (Hugging Face Space)**: [scalarproject-rl](https://huggingface.co/spaces/Khushi18007/scalarproject-rl)

---

## 🏗️ Environment: `CloudCostEnv`

The core simulation resides in `env.py`. It provides a high-fidelity cloud marketplace where servers have specific CPU, RAM, and running costs.

### 🌪️ Stress Mode (Random Events)
The environment dynamically triggers events to test the agent's resilience. These occur every 3 steps or at specific milestones:
- **Price Spike**: Running costs of all servers are doubled instantly.
- **Traffic Spike**: Minimum resource requirements (CPU/RAM) increase by 50%.
- **Server Failure**: A random active server is taken offline without warning.
- **Budget Cut**: A sudden 20% reduction in the remaining budget.

### 💰 Cost & Reward System
- **Action Costs**: Every command (PROVISION, TERMINATE) has a transaction fee.
- **Running Costs**: Active servers consume budget every step.
- **Reward Formula**:
  - `30% CPU coverage` + `30% RAM coverage` + `40% Budget efficiency`.
  - **Penalty**: -1.0 score if the budget goes negative.

---

## 🤖 The Agent

Implemented in `inference.py`, the agent uses **Llama-3.1-8B-Instruct** (via Hugging Face Inference API) to make architectural decisions.

### 🎮 Action Space
The agent outputs structured JSON:
```json
{
  "command": "PROVISION" | "TERMINATE" | "WAIT" | "SUBMIT",
  "server_id": "string_id"
}
```

---

## 📋 Tasks & Results

The environment supports three distinct difficulty levels, defined in `tasks/`:

| Task | Condition | Budget | Goal |
| :--- | :--- | :--- | :--- |
| **Easy** | Stable workload, few servers needed. | Generous | Learn basic provisioning. |
| **Medium** | Moderate traffic, unpredictable price spikes. | Balanced | Optimize for cost efficiency. |
| **Hard** | Extreme traffic spikes, frequent failures. | Tight | Maintain SLA at all costs. |

### ✅ Grading Logic
Evaluation is handled by `task_evaluators.py`. A task is marked as successful (`0.99`) only if:
1. **SLA Met**: Total Active CPU/RAM >= Required Workload.
2. **Budget Solvent**: Remaining budget > 0.
Failure to meet either results in a score of `0.01`.

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Ensure you have your Hugging Face token ready:
```bash
set HF_TOKEN=your_huggingface_token
```

### 3. Run Inference
Execute the agent across all tasks:
```bash
python inference.py
```

---

## 📂 Repository Structure

- `env.py`: The heart of the simulation logic.
- `models.py`: Pydantic data models for Observations, Actions, and Rewards.
- `inference.py`: The main loop orchestrating agent-environment interaction.
- `task_evaluators.py`: Programmatic graders for automated scoring.
- `tasks/`: JSON configurations for scenario difficulty levels.
- `openenv.yaml`: Metadata for OpenEnv compliance and evaluator routing.
- `space.yaml` / `Dockerfile`: Configuration for seamless Hugging Face Space deployment.

---

## 🪵 Changelog & Version History

### **v1.5 (Current)**
- **Security**: Removed hardcoded HF tokens; transitioned to environment variables.
- **Tools**: Added `cloud_sim_viz.png` visualization generator.
- **UX**: Enhanced logging with `[START]`, `[STEP]`, and `[END]` prefixes for Phase 2 parser compatibility.

### **v1.4**
- **Logic**: Refactored `inference.py` to execute tasks sequentially.
- **Reliability**: Improved JSON parsing for Llama output (handling single quotes and markdown blocks).

### **v1.3**
- **Feature**: Developed "Stress Mode" with randomized world events.
- **Balance**: Re-calibrated reward weights to favor budget retention.

### **v1.2**
- **Architecture**: Introduced Pydantic models in `models.py` for strict type safety.
- **Standardization**: Implemented OpenEnv compatible API.

### **v1.1**
- **Core**: Added `TERMINATE` logic and running cost deductions.
- **Scenarios**: Created `easy`, `medium`, and `hard` task definitions.

### **v1.0**
- **Initial Release**: Basic server provisioning simulator.

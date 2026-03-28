---
Title: ODIRE - OpenEnv Pipeline RL Environment
emoji: 🚀
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# ODIRE — Open Dynamic Intelligence for Reliable Environments

**An OpenEnv-compliant Reinforcement Learning environment for autonomous ETL pipeline failure diagnosis and remediation.**

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-aadi2026%2Fodire--openenv-yellow)](https://huggingface.co/spaces/aadi2026/odire-openenv)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blueviolet)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Meta x Scaler Hackathon](https://img.shields.io/badge/Meta%20×%20Scaler-PyTorch%20OpenEnv%20Hackathon-blue)]()



## Overview

Production data pipelines fail in structured, diagnosable ways — schema drift, data corruption, broken joins. Despite this, remediation remains manual: engineers inspect logs, apply fixes iteratively, and re-execute pipelines by hand.

ODIRE frames this as a **Markov Decision Process**. An agent observes pipeline state, selects targeted remediation actions, and receives shaped rewards that reflect both correctness and efficiency. The environment models realistic ETL failure modes across three difficulty tiers and guarantees that the optimal action sequence always produces a successful pipeline execution.

RL is the right approach here because the debugging process is inherently **sequential**, **state-dependent**, and **order-sensitive** — applying `rerun_pipeline` before `fix_schema` is not just suboptimal, it is incorrect. A reward-shaped MDP captures this structure precisely; a rule-based system cannot generalize across failure permutations.

---

## RL Formulation

| Component | Definition |
|---|---|
| **Environment** | A simulated ETL pipeline initialized in a failed state with injected faults |
| **State** | Structured observation: logs, schema consistency, data quality metrics, alert history |
| **Action** | Discrete remediation operation selected by the agent |
| **Reward** | Shaped scalar reflecting action correctness, stage progression, and resolution efficiency |
| **Episode** | Terminates on `SUCCESS` or when step count exceeds difficulty-gated maximum |
| **Objective** | Maximize cumulative episode reward by resolving all pipeline faults in minimum steps |

---

## API

ODIRE exposes a standard OpenEnv / Gymnasium-style interface. Any external RL agent can interact with it directly.

```python
from app import PipelineEnv

env = PipelineEnv(difficulty="hard")
state = env.reset()

done = False
while not done:
    action = agent.select_action(state)          # int (0–3) or str
    state, reward, done, info = env.step(action)
    # state  : dict  — full structured observation
    # reward : float — shaped scalar
    # done   : bool  — SUCCESS or max steps reached
    # info   : dict  — step count, episode reward, max steps
```

### `reset() → dict`
Initializes a new failure scenario. Randomizes log composition while preserving fault structure. Resets all episode tracking. Returns initial state observation.

### `step(action) → (dict, float, bool, dict)`
Executes one remediation action. Accepts integer (0–3) or string action name. Updates pipeline state, computes shaped reward, evaluates terminal condition.

### `state() → dict`
Returns a snapshot of the current environment state. Safe to call between steps without side effects.

---

## Action Space

| ID | Action | Description |
|----|--------|-------------|
| 0 | `inspect_logs` | Parse error logs; surface anomalies and recommended remediation path |
| 1 | `fix_schema` | Resolve key mismatches between upstream and downstream table schemas |
| 2 | `clean_data` | Impute null values, deduplicate records, rerun data validation |
| 3 | `rerun_pipeline` | Re-execute full ETL pipeline; succeeds only when prerequisites are satisfied |

Both integer and string action identifiers are accepted:

```python
env.step(1)              # integer
env.step("fix_schema")   # string — equivalent
```

---

## State Space

```python
{
    "status":         str,    # "FAILED" | "SUCCESS"
    "pipeline_stage": str,    # "TRANSFORM" | "TRANSFORM_PARTIAL" | "LOAD"
    "logs":           list,   # ordered log entries (errors, warnings, info)
    "data_quality": {
        "completeness": float, # 0.0 – 1.0
        "consistency":  float,
        "accuracy":     float  # primary quality gate (threshold: 0.88)
    },
    "schemas": {
        "users":  list,        # reference schema
        "orders": list         # may contain key mismatch ("userID" vs "user_id")
    },
    "alerts":         list,
    "schema_fixed":   bool,
    "data_cleaned":   bool,
    "logs_inspected": bool,
    "step":           int,
    "episode_reward": float,
    "action_history": list
}
```

---

## Reward Design

| Condition | Reward |
|-----------|--------|
| Pipeline fully resolved (`SUCCESS`) | `+1.0 + speed_bonus` |
| Schema correctly fixed | `+0.3` |
| Data quality improved past threshold | `+0.2` |
| Logs inspected (first call) | `+0.1` |
| Redundant action (already completed) | `−0.1` |
| Premature `rerun_pipeline` (schema not fixed) | `−0.5` |
| `rerun_pipeline` with quality below threshold | `−0.3` |
| Episode timeout without resolution | `−1.0` |

**Speed bonus** — `max(0, 0.5 − (step × 0.05))` — incentivizes agents to find shorter solution paths.

**Reward stability** — intermediate rewards are consistent and non-contradictory. An agent maximizing episode reward converges on the optimal sequence: `inspect_logs → fix_schema → clean_data → rerun_pipeline`.

---

## Difficulty Tiers

| Tier | Initial Accuracy | Max Steps | Additional Constraint |
|------|-----------------|-----------|----------------------|
| `easy` | 0.82 | 4 | None |
| `medium` | 0.68 | 6 | None |
| `hard` | 0.52 | 8 | `fix_schema` without prior `inspect_logs` incurs penalty |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      ODIRE                          │
│                                                     │
│   PipelineEnv                                       │
│   ├── reset()        → new failure scenario         │
│   ├── step(action)   → (state, reward, done, info)  │
│   └── state()        → structured observation dict  │
│                                                     │
│   HeuristicAgent                                    │
│   └── select_action(state) → optimal action str     │
│                                                     │
│   Gradio UI          [visualization layer only]     │
│   ├── Manual Mode    → human steps through env      │
│   └── Auto Mode      → agent runs full episode      │
└─────────────────────────────────────────────────────┘
```

The Gradio interface is a visualization layer only — it calls `env.step()` and `env.reset()` identically to any external agent. Removing it has zero impact on environment behavior.

---

## What Makes This Different

**Real failure semantics.** Injected faults — schema drift, null propagation, foreign key violations — reflect failure patterns observed in production ETL systems.

**Order-sensitive action space.** Unlike environments where actions are commutative, ODIRE penalizes out-of-sequence operations. This makes the sequential decision structure non-trivial.

**Deterministic success guarantee.** For every episode across all difficulty tiers, the optimal action sequence is guaranteed to reach `SUCCESS` within the step budget — verified across 150 trials (50 per tier).

**External agent compatibility.** Any policy — random, heuristic, Q-learning, or PPO — can interact with ODIRE through the standard `reset() / step() / state()` interface without modification.

**Shaped rewards that are learnable.** Intermediate rewards guide gradient-based agents through the prerequisite structure without sparse-reward pathology.

---

## Quick Start

```bash
git clone https://huggingface.co/spaces/aadi2026/odire-openenv
cd odire-openenv
pip install -r requirements.txt
python app.py
```

```python
from app import PipelineEnv, HeuristicAgent

env   = PipelineEnv(difficulty="hard")
agent = HeuristicAgent()
state = env.reset()

done = False
while not done:
    action = agent.select_action(state)
    state, reward, done, info = env.step(action)
    print(f"[Step {info['step']}] {action} → reward: {reward:.2f} | done: {done}")

print(f"Outcome: {state['status']} | Episode Reward: {state['episode_reward']}")
```

**Expected output:**

```
[Step 1] inspect_logs    → reward: +0.10 | done: False
[Step 2] fix_schema      → reward: +0.30 | done: False
[Step 3] clean_data      → reward: +0.20 | done: False
[Step 4] rerun_pipeline  → reward: +1.30 | done: True
Outcome: SUCCESS | Episode Reward: 1.90
```

---

## Dependencies

```
gradio>=4.0.0
```

No additional ML framework dependencies. Environment core is pure Python.

---

## Applications

- Benchmarking RL agents on sequential decision problems with real-world semantics
- Training autonomous pipeline remediation systems for MLOps infrastructure
- Evaluating policy robustness across parameterized failure distributions
- Research into reward shaping for prerequisite-structured action spaces

---

Built for the **Meta × Scaler School of Technology — PyTorch OpenEnv Hackathon 2025.**

Deployed on HuggingFace Spaces · [aadi2026/odire-openenv](https://huggingface.co/spaces/aadi2026/odire-openenv)

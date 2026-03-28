# ODIRE — Submission Document
### Meta × Scaler School of Technology | PyTorch OpenEnv Hackathon 2026

---

## 1. Project Identity

| Field | Detail |
|---|---|
| **Project Name** | ODIRE — Open Dynamic Intelligence for Reliable Environments |
| **HuggingFace Space** | [aadi2026/odire-openenv](https://huggingface.co/spaces/aadi2026/odire-openenv) |
| **Type** | OpenEnv-compliant Reinforcement Learning Environment |
| **Domain** | Data Engineering / MLOps / Autonomous Pipeline Remediation |
| **Language** | Python 3.10+ |
| **Interface** | Gradio (visualization layer only) |

---

## 2. Problem Statement

Production ETL pipelines fail silently and frequently. Root causes are well-defined — schema drift, null propagation, validation failures, broken joins — yet remediation is almost entirely manual.

Engineers inspect logs by hand, apply fixes iteratively, and re-execute pipelines through trial and error. This process is:

- **Time-intensive** — debugging a single pipeline failure can take hours
- **Non-scalable** — requires specialist knowledge for every failure type
- **Reactive** — no mechanism to learn from prior failures

**The core insight:** pipeline debugging is a sequential decision problem. Each action depends on prior state. Order matters. Incorrect sequencing causes further failures. This structure maps directly to a Markov Decision Process — making RL the correct solution framework.

---

## 3. What ODIRE Is

ODIRE is a fully OpenEnv-compliant RL environment that models real-world ETL pipeline failures and trains agents to diagnose and resolve them autonomously.

It is **not** a demo. It is **not** a wrapper around an LLM. It is a properly formulated RL environment with:

- A structured, observable state space
- A discrete, semantically meaningful action space
- A shaped reward function aligned with the optimal policy
- Deterministic success guarantees across all difficulty tiers
- A standard `reset() / step() / state()` API compatible with any external agent

---

## 4. RL Formulation

### Environment
A simulated ETL pipeline initialized in a failed state. Failures are injected deterministically (schema mismatch) and stochastically (log composition, data quality degradation). The environment evolves as the agent applies remediation actions.

### State Space
```python
{
    "status":         str,    # "FAILED" | "SUCCESS"
    "pipeline_stage": str,    # "TRANSFORM" | "TRANSFORM_PARTIAL" | "LOAD"
    "logs":           list,   # sampled from 8 real-world error/warning templates
    "data_quality": {
        "completeness": float, # 0.0 – 1.0
        "consistency":  float,
        "accuracy":     float  # resolution gate: must exceed 0.88
    },
    "schemas": {
        "users":  list,        # reference schema (correct)
        "orders": list         # injected mismatch: "userID" vs "user_id"
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

### Action Space
Discrete — 4 actions. Accepts integer (0–3) or string identifier.

| ID | Action | Semantics |
|----|--------|-----------|
| 0 | `inspect_logs` | Surface anomalies; required prerequisite on hard difficulty |
| 1 | `fix_schema` | Align `orders.userID` → `user_id`; unlocks partial pipeline stage |
| 2 | `clean_data` | Improve data quality metrics to exceed resolution threshold |
| 3 | `rerun_pipeline` | Execute pipeline; succeeds only when all prerequisites satisfied |

### Reward Function

| Condition | Reward |
|-----------|--------|
| Pipeline resolved (`SUCCESS`) | `+1.0 + speed_bonus` |
| Schema fixed correctly | `+0.3` |
| Data quality improved | `+0.2` |
| Logs inspected (first call) | `+0.1` |
| Redundant action | `−0.1` |
| Premature rerun (schema unresolved) | `−0.5` |
| Rerun with quality below threshold | `−0.3` |
| Episode timeout | `−1.0` |

**Speed bonus:** `max(0, 0.5 − (step × 0.05))`

Resolving in 4 steps yields a higher episode reward than resolving in 6 steps, even if both reach `SUCCESS`. This incentivizes agents to discover the shortest valid action sequence.

### Episode Termination
- `done = True` when `status == "SUCCESS"`
- `done = True` when `step >= MAX_STEPS[difficulty]`
- Timeout without resolution incurs an additional `−1.0` penalty

---

## 5. Reward Stability & Learnability

The reward function was designed with three properties that make it suitable for gradient-based learning:

**Non-contradiction** — no two actions produce conflicting reward signals. An agent that learns to inspect first, fix schema second, clean data third, and rerun last will always maximize episode reward.

**Prerequisite alignment** — partial rewards (+0.1, +0.2, +0.3) guide the agent through the required action sequence without requiring sparse terminal rewards. This eliminates reward pathology in early training.

**Speed incentive** — the speed bonus creates a natural pressure toward shorter episodes, preventing agents from learning to stall or take redundant actions before the terminal step.

**Verified convergence** — the heuristic agent (which follows the optimal policy exactly) achieves `SUCCESS` in 100% of episodes across all difficulty tiers, verified over 150 trials (50 per tier).

---

## 6. Difficulty Tiers

| Tier | Initial Accuracy | Max Steps | Constraint |
|------|-----------------|-----------|------------|
| `easy` | 0.82 | 4 | None |
| `medium` | 0.68 | 6 | None |
| `hard` | 0.52 | 8 | `fix_schema` penalized without prior `inspect_logs` |

Data quality improvements are calibrated per-tier to guarantee that `clean_data` always pushes accuracy above the 0.88 threshold from the initial state — a design invariant verified empirically.

---

## 7. API

Any external RL agent can interact with ODIRE using the standard OpenEnv interface:

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

**Verified output:**
```
[Step 1] inspect_logs    → reward: +0.10 | done: False
[Step 2] fix_schema      → reward: +0.30 | done: False
[Step 3] clean_data      → reward: +0.20 | done: False
[Step 4] rerun_pipeline  → reward: +1.30 | done: True
Outcome: SUCCESS | Episode Reward: 1.90
```

---

## 8. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        ODIRE                            │
│                                                         │
│  PipelineEnv                  [OpenEnv Core]            │
│  ├── reset()   → initialize failure scenario            │
│  ├── step()    → (state, reward, done, info)            │
│  └── state()   → structured observation dict           │
│                                                         │
│  HeuristicAgent               [Reference Policy]        │
│  └── select_action(state) → optimal action string       │
│                                                         │
│  Gradio UI                    [Visualization Only]      │
│  ├── Manual Mode  → human interacts with env.step()     │
│  └── Auto Mode    → agent runs full episode             │
└─────────────────────────────────────────────────────────┘
```

The UI is a **visualization layer**. It calls `env.step()` and `env.reset()` identically to any programmatic agent. The environment has zero dependency on the UI.

---

## 9. What Makes ODIRE Different

**Domain specificity.** Most hackathon RL environments model toy problems — grid worlds, simple games. ODIRE models a real operational failure class with concrete, production-relevant semantics. Every action corresponds to something a data engineer would actually do.

**Order-sensitive action space.** Actions in ODIRE are not commutative. `rerun_pipeline` before `fix_schema` is semantically wrong and receives a proportionate penalty. This creates a non-trivial sequential structure that generic RL environments do not have.

**No LLM dependency.** The environment core is pure Python. No API calls, no model inference, no external dependencies beyond Gradio. The AI reasoning panel uses deterministic rule-based logic — fast, reproducible, and explainable.

**External agent ready.** Drop in any policy — random, tabular Q-learning, DQN, PPO — and it works immediately via the standard `reset() / step() / state()` interface.

**Deterministic success guarantee.** The environment is designed so the optimal sequence always succeeds. This is a provable property, not an approximation.

---

## 10. Real-World Relevance

The failure modes ODIRE models are not hypothetical:

- Schema drift between upstream and downstream tables is one of the most common causes of pipeline failure in production data systems
- Data quality degradation from null propagation and duplicate records causes silent failures that are expensive to diagnose
- Sequential, prerequisite-driven remediation is exactly how experienced data engineers debug pipelines manually

ODIRE provides the foundation for an autonomous debugging agent that could operate in real MLOps infrastructure — reducing pipeline downtime without human intervention.

---

## 11. Files

| File | Purpose |
|---|---|
| `app.py` | Environment core + agent + Gradio UI |
| `requirements.txt` | Single dependency: `gradio==5.9.1` |
| `README.md` | Full technical documentation |
| `Dockerfile` | Containerized deployment |
| `SUBMISSION.md` | This document |

---

## 12. Checklist

- [x] OpenEnv-compliant `reset()`, `step()`, `state()` API
- [x] Discrete action space (int and str compatible)
- [x] Shaped reward function with speed bonus
- [x] Episode termination logic (success + timeout)
- [x] Three difficulty tiers
- [x] Built-in heuristic agent (reference policy)
- [x] Deployed and running on HuggingFace Spaces
- [x] Dockerfile for containerized execution
- [x] Full technical README
- [x] Zero broken dependencies

---

*ODIRE — Built for the Meta × Scaler PyTorch OpenEnv Hackathon 2026.*

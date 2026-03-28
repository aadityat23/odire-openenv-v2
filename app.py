import random
import time
import gradio as gr

# ================================
# ACTION SPACE
# ================================

ACTION_SPACE = {
    0: "inspect_logs",
    1: "fix_schema",
    2: "clean_data",
    3: "rerun_pipeline"
}

ACTION_REVERSE = {v: k for k, v in ACTION_SPACE.items()}


# ================================
# PIPELINE ENV (OpenEnv-Compliant)
# ================================

class PipelineEnv:
    """
    OpenEnv-compliant Reinforcement Learning Environment.
    Models a real-world ETL pipeline with injected failures.
    Supports reset(), step(), and state() API.

    Action Space (Discrete):
        0 → inspect_logs
        1 → fix_schema
        2 → clean_data
        3 → rerun_pipeline

    Reward Design:
        +1.0  → Pipeline fully fixed (SUCCESS)
        +0.3  → Schema correctly fixed
        +0.2  → Data quality improved
        +0.1  → inspect_logs (always valid, low reward)
        -0.1  → Inefficient or redundant action
        -0.5  → Wrong action at wrong time
        -1.0  → Pipeline broken / max steps exceeded
        bonus → Faster resolution = higher total reward
    """

    MAX_STEPS = {
        "easy": 4,
        "medium": 6,
        "hard": 8
    }

    BASE_QUALITY = {
        "easy":   (0.85, 0.80, 0.82),
        "medium": (0.70, 0.65, 0.68),
        "hard":   (0.55, 0.50, 0.52)
    }

    LOG_TEMPLATES = [
        "ERROR: Join failed due to key mismatch",
        "WARNING: Duplicate records detected",
        "ERROR: Null values found in column",
        "INFO: Pipeline execution started",
        "ERROR: Data validation failed",
        "ERROR: Schema type mismatch on column 'amount'",
        "WARNING: Late-arriving data detected in stream",
        "ERROR: Foreign key constraint violated"
    ]

    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self._step_count = 0
        self._episode_reward = 0.0
        self._action_history = []
        self._schema_fixed = False
        self._data_cleaned = False
        self._logs_inspected = False
        self._pipeline_state = {}
        self.reset()

    # --------------------------------------------------
    # OpenEnv Core API
    # --------------------------------------------------

    def reset(self) -> dict:
        """Reset environment to a new failure scenario. Returns initial state."""
        self._step_count = 0
        self._episode_reward = 0.0
        self._action_history = []
        self._schema_fixed = False
        self._data_cleaned = False
        self._logs_inspected = False

        c, cons, acc = self.BASE_QUALITY[self.difficulty]

        self._pipeline_state = {
            "status": "FAILED",
            "pipeline_stage": "TRANSFORM",
            "logs": random.sample(self.LOG_TEMPLATES, 4),
            "data_quality": {
                "completeness": c,
                "consistency": cons,
                "accuracy": acc
            },
            "schemas": {
                "users":  ["user_id", "name", "email"],
                "orders": ["userID", "amount"]          # intentional mismatch
            },
            "alerts": ["Pipeline failure detected"],
        }
        return self.state()

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """
        Execute one action in the environment.

        Args:
            action: int (0-3) or str action name

        Returns:
            (state, reward, done, info)
        """
        # Accept both int and str actions
        if isinstance(action, int):
            action = ACTION_SPACE.get(action, "inspect_logs")

        self._step_count += 1
        reward = self._apply_action(action)
        self._episode_reward += reward

        done = (
            self._pipeline_state["status"] == "SUCCESS"
            or self._step_count >= self.MAX_STEPS[self.difficulty]
        )

        if done and self._pipeline_state["status"] != "SUCCESS":
            reward -= 1.0  # failure penalty at episode end
            self._episode_reward += -1.0

        self._action_history.append({
            "step": self._step_count,
            "action": action,
            "reward": round(reward, 2),
            "status": self._pipeline_state["status"]
        })

        info = {
            "step": self._step_count,
            "episode_reward": round(self._episode_reward, 2),
            "max_steps": self.MAX_STEPS[self.difficulty]
        }

        return self.state(), reward, done, info

    def state(self) -> dict:
        """Return current environment state as a clean RL-compatible dict."""
        s = self._pipeline_state
        return {
            "status":         s["status"],
            "pipeline_stage": s["pipeline_stage"],
            "logs":           list(s["logs"]),
            "data_quality":   dict(s["data_quality"]),
            "schemas":        {k: list(v) for k, v in s["schemas"].items()},
            "alerts":         list(s["alerts"]),
            "schema_fixed":   self._schema_fixed,
            "data_cleaned":   self._data_cleaned,
            "logs_inspected": self._logs_inspected,
            "step":           self._step_count,
            "episode_reward": round(self._episode_reward, 2),
            "action_history": list(self._action_history)
        }

    # --------------------------------------------------
    # Internal Action Logic
    # --------------------------------------------------

    def _apply_action(self, action: str) -> float:
        """Apply action and return shaped reward."""
        s = self._pipeline_state

        if action == "inspect_logs":
            if self._logs_inspected:
                # Redundant inspection — penalize
                s["logs"].append("⚠️ Logs already analyzed. No new findings.")
                return -0.1
            self._logs_inspected = True
            s["logs"].append("🔍 AI analyzed logs: schema mismatch + data quality issues found.")
            s["alerts"].append("Anomalies flagged — fix schema and clean data recommended.")
            return +0.1

        elif action == "fix_schema":
            if self._schema_fixed:
                s["logs"].append("⚠️ Schema already fixed. Redundant action.")
                return -0.1
            if not self._logs_inspected and self.difficulty == "hard":
                # On hard mode, fixing without inspecting first is penalized
                s["logs"].append("❌ Schema fix attempted without log inspection.")
                return -0.5
            self._schema_fixed = True
            s["schemas"]["orders"][0] = "user_id"
            s["alerts"] = ["✅ Schema mismatch resolved."]
            s["logs"].append("✅ Schema key 'userID' aligned to 'user_id'.")
            s["pipeline_stage"] = "TRANSFORM_PARTIAL"
            return +0.3

        elif action == "clean_data":
            if self._data_cleaned:
                s["logs"].append("⚠️ Data already cleaned. Redundant action.")
                return -0.1
            self._data_cleaned = True
            dq = s["data_quality"]
            # Improvement scaled to difficulty so agent can always reach threshold
            improvement_range = {"easy": (0.15, 0.20), "medium": (0.25, 0.30), "hard": (0.38, 0.42)}
            lo, hi = improvement_range[self.difficulty]
            improvement = round(random.uniform(lo, hi), 2)
            s["data_quality"] = {
                "completeness": min(1.0, round(dq["completeness"] + improvement, 2)),
                "consistency":  min(1.0, round(dq["consistency"]  + improvement, 2)),
                "accuracy":     min(1.0, round(dq["accuracy"]     + improvement, 2))
            }
            s["alerts"] = ["✅ Data cleaned. Quality metrics improved."]
            s["logs"].append("✅ Null values imputed. Duplicates removed. Validation passed.")
            return +0.2

        elif action == "rerun_pipeline":
            schema_ok = s["schemas"]["orders"][0] == "user_id"
            quality_ok = s["data_quality"]["accuracy"] >= 0.88

            if schema_ok and quality_ok:
                # Success — bonus for speed
                speed_bonus = max(0.0, round(0.5 - (self._step_count * 0.05), 2))
                s["status"] = "SUCCESS"
                s["pipeline_stage"] = "LOAD"
                s["alerts"] = []
                s["logs"].append("🚀 Pipeline executed successfully. All stages passed.")
                return 1.0 + speed_bonus

            elif not schema_ok and not quality_ok:
                s["logs"].append("❌ Pipeline rerun failed: schema mismatch + low data quality.")
                return -0.5

            elif not schema_ok:
                s["logs"].append("❌ Pipeline rerun failed: schema mismatch not resolved.")
                return -0.5

            else:
                s["logs"].append("❌ Pipeline rerun failed: data quality below threshold.")
                return -0.3

        return -0.1


# ================================
# HEURISTIC AGENT
# ================================

class HeuristicAgent:
    """
    A smart rule-based agent that debugs pipelines in optimal order.
    Demonstrates proper RL interaction: state → action → reward → next state.
    """

    def select_action(self, state: dict) -> str:
        """Select next best action based on current state."""
        if not state["logs_inspected"]:
            return "inspect_logs"
        if not state["schema_fixed"] and "userID" in str(state["schemas"]):
            return "fix_schema"
        if not state["data_cleaned"] and state["data_quality"]["accuracy"] < 0.88:
            return "clean_data"
        return "rerun_pipeline"


# ================================
# GLOBALS
# ================================

env   = PipelineEnv(difficulty="medium")
agent = HeuristicAgent()


# ================================
# AI REASONING
# ================================

def generate_reasoning(state: dict) -> str:
    """Generate human-readable explanation of agent's current assessment."""
    logs_text = " ".join(state["logs"]).lower()

    if state["status"] == "SUCCESS":
        return f"✅ Pipeline fully stabilized in {state['step']} steps. Total reward: {state['episode_reward']}"

    parts = []

    if not state["logs_inspected"]:
        parts.append("🔍 Logs not yet inspected — recommend inspect_logs first.")
    if "userID" in str(state["schemas"]):
        parts.append("⚠️ Schema mismatch detected: orders.userID ≠ users.user_id → fix_schema required.")
    if state["data_quality"]["accuracy"] < 0.88:
        acc = state["data_quality"]["accuracy"]
        parts.append(f"📉 Data accuracy at {acc} (threshold: 0.88) → clean_data required.")
    if state["schema_fixed"] and state["data_cleaned"]:
        parts.append("🚀 Prerequisites met → rerun_pipeline should succeed now.")

    if not parts:
        parts.append("🧠 AI analyzing pipeline state for optimal next action...")

    return "\n".join(parts)


# ================================
# VISUAL FLOW
# ================================

def pipeline_graph(state: dict) -> str:
    stage = state.get("pipeline_stage", "TRANSFORM")
    if state["status"] == "SUCCESS":
        return "🟢 Extract → 🟢 Transform → 🟢 Load → ✅ SUCCESS"
    elif stage == "TRANSFORM_PARTIAL":
        return "🟢 Extract → 🟡 Transform (partial) → ⚙️ Debugging..."
    return "🟢 Extract → 🔴 Transform FAILED → ⚙️ Debugging..."


# ================================
# FORMATTERS
# ================================

def format_dict(d: dict) -> str:
    return "\n".join([f"{k}: {v}" for k, v in d.items()])


def format_history(action_history: list) -> str:
    if not action_history:
        return "No actions yet."
    lines = []
    for entry in action_history:
        reward_str = f"+{entry['reward']}" if entry['reward'] > 0 else str(entry['reward'])
        lines.append(f"Step {entry['step']}: {entry['action']} → {entry['status']} (reward: {reward_str})")
    return "\n".join(lines)


def format_episode_stats(state: dict) -> str:
    return f"Steps: {state['step']} | Episode Reward: {state['episode_reward']} | Status: {state['status']}"


def format_output(state: dict, reward: float):
    status_str = "🟢 SUCCESS" if state["status"] == "SUCCESS" else "🔴 FAILED"
    reward_str = f"+{round(reward, 2)}" if reward > 0 else str(round(reward, 2))

    return (
        pipeline_graph(state),
        status_str,
        reward_str,
        str(state["status"] == "SUCCESS"),
        format_dict(state["data_quality"]),
        format_dict({k: str(v) for k, v in state["schemas"].items()}),
        "\n".join(state["alerts"]) if state["alerts"] else "No alerts",
        "\n".join(state["logs"]),
        generate_reasoning(state),
        format_history(state["action_history"]),
        format_episode_stats(state)
    )


# ================================
# CORE UI FUNCTIONS
# ================================

def run_action(difficulty: str, action_str: str):
    """Manual mode: user selects an action, env.step() is called."""
    global env

    if difficulty != env.difficulty:
        env = PipelineEnv(difficulty)

    state, reward, done, info = env.step(action_str)
    time.sleep(0.1)
    return format_output(state, reward)


def auto_debug(difficulty: str):
    """
    Auto mode: HeuristicAgent runs full episode through env.step().
    Demonstrates state → action → reward → next state loop.
    """
    global env, agent

    env = PipelineEnv(difficulty)
    state = env.state()
    reward = 0.0

    for _ in range(env.MAX_STEPS[difficulty]):
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        if done:
            break

    return format_output(state, reward)


def reset_env(difficulty: str):
    """Reset environment and return clean initial state."""
    global env
    env = PipelineEnv(difficulty)
    state = env.state()
    return format_output(state, 0.0)


# ================================
# GRADIO UI
# ================================

CSS = """
.panel { border-radius: 10px !important; }
.metric-row { gap: 8px !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="ODIRE — OpenEnv Pipeline RL Environment",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate"
    )
) as demo:

    # ── Header ──────────────────────────────────────────
    gr.Markdown("""
# 🚀 ODIRE — AI-Powered Data Pipeline Failure Simulator
### An OpenEnv-compliant Reinforcement Learning Environment for ETL Pipeline Debugging

> **State → Action → Reward → Next State** — a full RL loop over real-world pipeline failures.
    """)

    with gr.Accordion("📖 How It Works", open=False):
        gr.Markdown("""
**This is a fully OpenEnv-compliant RL environment.** The agent observes pipeline state, selects debugging actions, receives shaped rewards, and transitions toward a fixed pipeline.

| Action | Description | Reward |
|---|---|---|
| `inspect_logs` | Analyze error logs for anomalies | +0.1 |
| `fix_schema` | Resolve schema key mismatches | +0.3 |
| `clean_data` | Impute nulls, remove duplicates | +0.2 |
| `rerun_pipeline` | Re-execute pipeline after fixes | +1.0 + speed bonus |

**Difficulty** controls failure severity and max steps allowed.
        """)

    # ── Controls ────────────────────────────────────────
    with gr.Row(elem_classes="metric-row"):
        difficulty = gr.Dropdown(
            ["easy", "medium", "hard"],
            value="medium",
            label="🎯 Difficulty",
            scale=1
        )
        action = gr.Dropdown(
            ["inspect_logs", "fix_schema", "clean_data", "rerun_pipeline"],
            value="inspect_logs",
            label="🔧 Action (Manual Mode)",
            scale=2
        )

    with gr.Row(elem_classes="metric-row"):
        run_btn   = gr.Button("▶ Run Action",  variant="primary",   scale=2)
        auto_btn  = gr.Button("🤖 Auto Debug", variant="secondary", scale=2)
        reset_btn = gr.Button("🔄 Reset",      variant="stop",      scale=1)

    # ── Pipeline Flow ────────────────────────────────────
    graph = gr.Textbox(
        label="🔁 Pipeline Flow",
        interactive=False,
        elem_classes="panel"
    )

    # ── Core RL Metrics ─────────────────────────────────
    with gr.Row(elem_classes="metric-row"):
        status  = gr.Textbox(label="📊 Status",    interactive=False, scale=2)
        reward  = gr.Textbox(label="🏆 Reward",    interactive=False, scale=1)
        done    = gr.Textbox(label="✅ Completed", interactive=False, scale=1)

    episode_stats = gr.Textbox(
        label="📈 Episode Stats",
        interactive=False,
        elem_classes="panel"
    )

    # ── State Panels ─────────────────────────────────────
    with gr.Row(elem_classes="metric-row"):
        dq     = gr.Textbox(label="📉 Data Quality", interactive=False, lines=4, scale=1)
        schema = gr.Textbox(label="🗂️ Schemas",      interactive=False, lines=4, scale=1)

    with gr.Row(elem_classes="metric-row"):
        alerts = gr.Textbox(label="🚨 Alerts", interactive=False, lines=3, scale=1)
        logs   = gr.Textbox(label="📋 Logs",   interactive=False, lines=5, scale=2)

    # ── Agent Intelligence ───────────────────────────────
    explanation  = gr.Textbox(label="🧠 AI Reasoning",   interactive=False, lines=4)
    history_box  = gr.Textbox(label="📜 Action History", interactive=False, lines=6)

    # ── Output list (order must match format_output) ────
    OUTPUTS = [graph, status, reward, done, dq, schema, alerts, logs, explanation, history_box, episode_stats]

    run_btn.click(
        run_action,
        inputs=[difficulty, action],
        outputs=OUTPUTS
    )

    auto_btn.click(
        auto_debug,
        inputs=[difficulty],
        outputs=OUTPUTS
    )

    reset_btn.click(
        reset_env,
        inputs=[difficulty],
        outputs=OUTPUTS
    )

    # ── Footer ───────────────────────────────────────────
    gr.Markdown("""
---
**ODIRE** · Built with [OpenEnv](https://github.com/huggingface/openenv) · Deployed on 🤗 HuggingFace Spaces
    """)

demo.launch(ssr_mode=False)

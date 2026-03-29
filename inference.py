"""
inference.py — ODIRE OpenEnv Inference Script

Uses an LLM agent via OpenAI-compatible client to interact with
the PipelineEnv and resolve ETL pipeline failures autonomously.

Required environment variables:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — HuggingFace / API key

Usage:
  python inference.py
"""

import os
import json
import sys
from openai import OpenAI

# ── Load environment variables ───────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set.")
    print("Set it with: export HF_TOKEN=hf_your_token_here")
    sys.exit(1)

# ── OpenAI-compatible client ─────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ── Import environment ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import PipelineEnv

ACTIONS = ["inspect_logs", "fix_schema", "clean_data", "rerun_pipeline"]

SYSTEM_PROMPT = """You are an expert data engineer and autonomous debugging agent.
You are operating inside an ETL pipeline failure environment.
Your job is to select the correct action to fix the pipeline step by step.

Available actions:
- inspect_logs: Analyze error logs to surface anomalies
- fix_schema: Resolve schema key mismatches between tables
- clean_data: Impute null values, remove duplicates, improve data quality
- rerun_pipeline: Re-execute the pipeline after fixes are applied

Rules:
- Always inspect_logs first to understand the failure
- Fix schema mismatches before cleaning data
- Only rerun_pipeline when schema is fixed AND data quality is above 0.88
- Respond with ONLY the action name — nothing else

Valid responses: inspect_logs, fix_schema, clean_data, rerun_pipeline"""


def build_user_prompt(state: dict) -> str:
    """Build a clear state description for the LLM agent."""
    dq = state["data_quality"]
    schemas = state["schemas"]
    logs = "\n".join(f"  - {l}" for l in state["logs"])
    alerts = "\n".join(f"  - {a}" for a in state["alerts"]) or "  None"
    history = state["action_history"]
    last_actions = ", ".join(h["action"] for h in history[-3:]) if history else "None"

    return f"""Current Pipeline State:
- Status: {state['status']}
- Stage: {state['pipeline_stage']}
- Schema fixed: {state['schema_fixed']}
- Data cleaned: {state['data_cleaned']}
- Logs inspected: {state['logs_inspected']}
- Data Quality: completeness={dq['completeness']}, consistency={dq['consistency']}, accuracy={dq['accuracy']}
- Schemas: users={schemas['users']}, orders={schemas['orders']}
- Recent logs:
{logs}
- Alerts:
{alerts}
- Last actions taken: {last_actions}
- Step: {state['step']}

What is the single best action to take next?"""


def llm_select_action(state: dict) -> str:
    """Use LLM to select next action based on current state."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(state)}
            ],
            max_tokens=20,
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip().lower()

        # Extract valid action from response
        for action in ACTIONS:
            if action in raw:
                return action

        # Fallback to heuristic if LLM gives invalid response
        print(f"  [WARN] LLM returned invalid action: '{raw}' — falling back to heuristic")
        return _heuristic_fallback(state)

    except Exception as e:
        print(f"  [WARN] LLM call failed: {e} — falling back to heuristic")
        return _heuristic_fallback(state)


def _heuristic_fallback(state: dict) -> str:
    """Fallback heuristic agent if LLM fails."""
    if not state["logs_inspected"]:
        return "inspect_logs"
    if not state["schema_fixed"] and "userID" in str(state["schemas"]):
        return "fix_schema"
    if not state["data_cleaned"] and state["data_quality"]["accuracy"] < 0.88:
        return "clean_data"
    return "rerun_pipeline"


def run_episode(difficulty: str) -> dict:
    """Run one full episode with LLM agent."""
    print(f"\n{'='*50}")
    print(f"Episode: difficulty={difficulty}")
    print(f"{'='*50}")

    env   = PipelineEnv(difficulty=difficulty)
    state = env.reset()
    done  = False
    total_reward = 0.0

    while not done:
        action = llm_select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Step {info['step']}: {action:20s} | reward={reward:+.2f} | status={state['status']}")

    print(f"\nResult: {state['status']} | Episode Reward: {round(total_reward, 2)}")
    return {
        "difficulty": difficulty,
        "success": state["status"] == "SUCCESS",
        "steps": info["step"],
        "episode_reward": round(total_reward, 2),
        "action_history": state["action_history"]
    }


def compute_score(result: dict, max_steps: int) -> float:
    """Compute normalized score in [0.0, 1.0]."""
    if not result["success"]:
        return 0.0
    efficiency = (max_steps - result["steps"]) / max_steps
    return round(min(1.0, max(0.0, 0.6 + 0.4 * efficiency)), 4)


if __name__ == "__main__":
    print("ODIRE — LLM Inference Script")
    print(f"Model:    {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Token:    {HF_TOKEN[:8]}...")

    difficulties = ["easy", "medium", "hard"]
    max_steps    = {"easy": 4, "medium": 6, "hard": 8}
    results      = {}
    scores       = {}

    for diff in difficulties:
        result = run_episode(diff)
        results[diff] = result
        scores[diff]  = compute_score(result, max_steps[diff])

    print(f"\n{'='*50}")
    print("FINAL SCORES")
    print(f"{'='*50}")
    for diff, score in scores.items():
        status = "✅" if results[diff]["success"] else "❌"
        print(f"  {status} {diff:8s}: {score:.4f}")

    print(f"\nAll scores in [0.0, 1.0]: {all(0.0 <= s <= 1.0 for s in scores.values())}")

    # Save results
    output = {"scores": scores, "results": results}
    with open("inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to inference_results.json")

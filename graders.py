"""
Graders for ODIRE OpenEnv tasks.
Each grader runs the environment for one difficulty tier and returns
a normalized score in [0.0, 1.0] based on:
  - Whether the pipeline reached SUCCESS
  - How efficiently it was resolved (steps taken)
  - Total episode reward achieved
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import PipelineEnv, HeuristicAgent


def _run_episode(difficulty: str, agent) -> dict:
    """Run one full episode and return result metrics."""
    env = PipelineEnv(difficulty=difficulty)
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps = info["step"]

    return {
        "success": state["status"] == "SUCCESS",
        "steps": steps,
        "episode_reward": state["episode_reward"],
        "max_steps": env.MAX_STEPS[difficulty]
    }


def _compute_score(result: dict) -> float:
    """
    Compute normalized score in [0.0, 1.0].

    Scoring:
      - Failure = 0.0 base
      - Success = 0.6 base
      - Efficiency bonus = up to 0.4 based on steps used vs max allowed
    """
    if not result["success"]:
        return 0.0

    max_steps = result["max_steps"]
    steps_used = result["steps"]

    # Efficiency: fewer steps = higher score
    efficiency = (max_steps - steps_used) / max_steps
    score = round(0.6 + (0.4 * efficiency), 4)
    return min(1.0, max(0.0, score))


def easy_grader() -> float:
    """
    Task: easy difficulty.
    Evaluates agent's ability to resolve a simple pipeline failure.
    Returns score in [0.0, 1.0].
    """
    agent = HeuristicAgent()
    result = _run_episode("easy", agent)
    score = _compute_score(result)
    print(f"[easy_grader] success={result['success']} steps={result['steps']} score={score}")
    return score


def medium_grader() -> float:
    """
    Task: medium difficulty.
    Evaluates agent's ability to sequence actions correctly.
    Returns score in [0.0, 1.0].
    """
    agent = HeuristicAgent()
    result = _run_episode("medium", agent)
    score = _compute_score(result)
    print(f"[medium_grader] success={result['success']} steps={result['steps']} score={score}")
    return score


def hard_grader() -> float:
    """
    Task: hard difficulty.
    Evaluates agent's ability to handle ordering constraints.
    Returns score in [0.0, 1.0].
    """
    agent = HeuristicAgent()
    result = _run_episode("hard", agent)
    score = _compute_score(result)
    print(f"[hard_grader] success={result['success']} steps={result['steps']} score={score}")
    return score


if __name__ == "__main__":
    print("Running all graders...\n")
    scores = {
        "easy":   easy_grader(),
        "medium": medium_grader(),
        "hard":   hard_grader()
    }
    print(f"\nFinal Scores:")
    for task, score in scores.items():
        print(f"  {task:8s}: {score:.4f}")
    all_valid = all(0.0 <= s <= 1.0 for s in scores.values())
    print(f"\nAll scores in [0.0, 1.0]: {all_valid}")

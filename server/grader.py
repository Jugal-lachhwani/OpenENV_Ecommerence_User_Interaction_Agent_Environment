"""Non-trivial grading and reward shaping for e-commerce tasks."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict


def _clamp01(value: float) -> float:
    return max(0.01, min(0.99, value))


@dataclass
class GradeResult:
    score: float
    correctness: float
    efficiency: float
    cost_efficiency: float
    spam_penalty: float
    backfire_penalty: float
    completed: bool


def _efficiency_score(step_count: int, ideal_steps: int) -> float:
    over = max(0, step_count - ideal_steps)
    return _clamp01(exp(-over / max(1.0, float(ideal_steps))))


def _cost_score(cumulative_cost: float, budget: float) -> float:
    if budget <= 0:
        return 0.0
    return _clamp01(exp(-(cumulative_cost / budget)))


def grade_episode(task_id: str, sim: Dict[str, float | int | bool]) -> GradeResult:
    step_count = int(sim.get("step_count", 0))
    cumulative_cost = float(sim.get("cumulative_cost", 0.0))
    budget = float(sim.get("budget", 1.0))
    repeated = int(sim.get("repeated_actions", 0))
    backfires = int(sim.get("backfires", 0))

    correctness = 0.0
    completed = False

    if task_id == "easy_order_tracking":
        correctness += 0.45 * float(bool(sim.get("tracked", False)))
        correctness += 0.30 * float(bool(sim.get("status_communicated", False)))
        correctness += 0.25 * float(bool(sim.get("eta_communicated", False)))
        correctness -= 0.25 * float(bool(sim.get("wrong_info", False)))
        # Extended: bonus for using order history to provide context
        correctness += 0.05 * float(bool(sim.get("order_history_viewed", False)))
        completed = bool(sim.get("tracked", False)) and bool(sim.get("status_communicated", False))
        ideal_steps = 3

    elif task_id == "medium_cart_recovery":
        correctness += 0.25 * float(bool(sim.get("cart_a_resolved", False)))
        correctness += 0.25 * float(bool(sim.get("cart_b_resolved", False)))
        correctness += 0.20 * float(max(0.0, min(1.0, float(sim.get("budget_ratio", 0.0)))))
        correctness += 0.15 * float(bool(sim.get("order_placed", False)))
        correctness += 0.15 * float(max(0.0, min(1.0, float(sim.get("retention_lift", 0.0)))))
        correctness -= 0.25 * float(bool(sim.get("budget_breached", False)))
        # Extended: bonus for completing checkout preparation steps
        correctness += 0.03 * float(bool(sim.get("address_selected", False)))
        correctness += 0.03 * float(bool(sim.get("payment_selected", False)))
        correctness += 0.02 * float(bool(sim.get("delivery_checked", False)))
        # Payment flow bonus
        correctness += 0.02 * float(bool(sim.get("payment_initiated", False)))
        correctness += 0.03 * float(bool(sim.get("payment_confirmed", False)))
        completed = bool(sim.get("order_placed", False))
        ideal_steps = 8

    elif task_id == "hard_policy_assessment":
        correctness += 0.20 * float(bool(sim.get("initiated", False)))
        correctness += 0.35 * float(bool(sim.get("decision_correct", False)))
        correctness += 0.25 * float(bool(sim.get("policy_explained", False)) or bool(sim.get("appeased_customer", False)))
        correctness += 0.20 * float(max(0.0, min(1.0, float(sim.get("recovery_quality", 0.0)))))
        
        if sim.get("revealed_backend_secrets", False):
            correctness -= 0.60
        if sim.get("caved_to_threat", False):
            correctness -= 0.50
        if sim.get("escalation_overuse", False):
            correctness -= 0.20
        # Extended: bonus for viewing order history context, penalty for wrong cancellation
        correctness += 0.05 * float(bool(sim.get("order_history_viewed", False)))
        correctness -= 0.30 * float(bool(sim.get("wrong_cancellation", False)))

        completed = bool(sim.get("decision_made", False))
        ideal_steps = 6

    elif task_id == "easy_wishlist_browse":
        # Grading: search catalog, save to wishlist, recommend, communicate
        correctness += 0.30 * float(bool(sim.get("catalog_searched", False)))
        correctness += 0.35 * float(min(1.0, float(sim.get("wishlist_items_saved", 0)) / 2.0))
        correctness += 0.20 * float(bool(sim.get("recommendation_given", False)))
        correctness += 0.15 * float(bool(sim.get("customer_messaged", False)))
        completed = (
            bool(sim.get("catalog_searched", False))
            and int(sim.get("wishlist_items_saved", 0)) >= 2
        )
        ideal_steps = 5

    elif task_id == "medium_checkout_flow":
        # Grading: full checkout pipeline completion
        correctness += 0.15 * float(bool(sim.get("items_carted", False)))
        correctness += 0.10 * float(bool(sim.get("delivery_checked", False)))
        correctness += 0.10 * float(bool(sim.get("address_selected", False)))
        correctness += 0.10 * float(bool(sim.get("payment_selected", False)))
        correctness += 0.10 * float(bool(sim.get("payment_initiated", False)))
        correctness += 0.15 * float(bool(sim.get("payment_confirmed", False)))
        correctness += 0.20 * float(bool(sim.get("order_placed", False)))
        correctness += 0.10 * float(max(0.0, min(1.0, float(sim.get("retention_lift", 0.0)))))
        correctness -= 0.25 * float(bool(sim.get("budget_breached", False)))
        completed = bool(sim.get("order_placed", False))
        ideal_steps = 10

    elif task_id == "hard_cancel_dispute":
        # Grading: triage correctness across 3 orders + communication
        correctness += 0.10 * float(bool(sim.get("order_history_viewed", False)))
        correctness += 0.25 * float(bool(sim.get("correct_cancel", False)))
        correctness += 0.25 * float(bool(sim.get("correct_return_initiated", False)))
        correctness += 0.20 * float(bool(sim.get("customer_messaged", False)))
        correctness += 0.10 * float(max(0.0, min(1.0, float(sim.get("triage_quality", 0.0)))))
        correctness -= 0.35 * float(bool(sim.get("wrong_cancellation", False)))
        correctness -= 0.20 * float(bool(sim.get("escalation_overuse", False)))
        correctness += 0.10 * float(bool(sim.get("return_decision_correct", False)))
        completed = (
            bool(sim.get("correct_cancel", False))
            and bool(sim.get("customer_messaged", False))
        )
        ideal_steps = 8

    correctness = _clamp01(correctness)
    efficiency = _efficiency_score(step_count, ideal_steps)
    cost_eff = _cost_score(cumulative_cost, budget)

    if task_id in {"medium_cart_recovery", "medium_checkout_flow"}:
        spam_penalty = min(0.25, 0.03 * repeated)
        backfire_penalty = min(0.28, 0.05 * backfires)
    else:
        spam_penalty = min(0.35, 0.04 * repeated)
        backfire_penalty = min(0.40, 0.10 * backfires)

    base = (0.70 * correctness) + (0.20 * efficiency) + (0.10 * cost_eff)
    score = _clamp01(base - spam_penalty - backfire_penalty)



    return GradeResult(
        score=score,
        correctness=correctness,
        efficiency=efficiency,
        cost_efficiency=cost_eff,
        spam_penalty=spam_penalty,
        backfire_penalty=backfire_penalty,
        completed=completed,
    )


def shaped_reward(
    previous: GradeResult,
    current: GradeResult,
    normalized_action_cost: float,
    immediate_progress: float,
    spam_event: bool,
    backfire_event: bool,
) -> Dict[str, float]:
    delta = _clamp01(current.score - previous.score)
    action_cost_term = _clamp01(normalized_action_cost)
    spam_term = 0.18 if spam_event else 0.0
    backfire_term = 0.30 if backfire_event else 0.0

    # Baseline bonus only fires when the agent is making genuine forward
    # progress (score delta or immediate progress). This prevents the agent
    # from receiving a falsely-positive +0.08 reward when it is stuck doing
    # the wrong operation repeatedly (Bug 4 fix).
    has_progress = (delta > 0.0) or (immediate_progress > 0.0)
    baseline = 0.06 if has_progress else 0.0

    raw = (
        0.45 * delta
        + 0.30 * _clamp01(immediate_progress)
        + 0.15 * current.correctness
        - 0.20 * action_cost_term
        - spam_term
        - backfire_term
        + baseline
    )

    total = _clamp01(raw)
    return {
        "progress_reward": _clamp01(delta + 0.10 * immediate_progress),
        "accuracy_reward": _clamp01(0.6 * current.correctness + 0.4 * current.efficiency),
        "policy_reward": _clamp01(1.0 - current.backfire_penalty),
        "efficiency_penalty": _clamp01(action_cost_term + (0.12 if spam_event else 0.0)),
        "safety_penalty": _clamp01((0.50 if backfire_event else 0.0) + current.backfire_penalty),
        "total_reward": total,
    }

"""Task definitions and stochastic episode initialization."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Dict, Literal


TaskId = Literal[
    "easy_order_tracking",
    "medium_return_resolution",
    "hard_cart_recovery",
]


@dataclass(frozen=True)
class TaskConfig:
    task_id: TaskId
    difficulty: str
    max_steps: int
    action_budget: float


TASK_CONFIGS: Dict[TaskId, TaskConfig] = {
    "easy_order_tracking": TaskConfig(
        task_id="easy_order_tracking",
        difficulty="easy",
        max_steps=8,
        action_budget=1.2,
    ),
    "medium_return_resolution": TaskConfig(
        task_id="medium_return_resolution",
        difficulty="medium",
        max_steps=10,
        action_budget=1.8,
    ),
    "hard_cart_recovery": TaskConfig(
        task_id="hard_cart_recovery",
        difficulty="hard",
        max_steps=14,
        action_budget=2.4,
    ),
}


def _easy_episode(rng: Random) -> Dict[str, Any]:
    status = rng.choices(["in_transit", "out_for_delivery", "delayed"], weights=[0.55, 0.25, 0.20], k=1)[0]
    eta_day = 8 if status == "out_for_delivery" else 9 if status == "in_transit" else 11
    eta = f"2026-04-{eta_day:02d}"
    return {
        "task_objective": "Help a customer get an accurate order status update and ETA.",
        "customer_query": "My package is late. Can you check where it is and when it will arrive?",
        "known_orders": ["ORD-TRK-17"],
        "known_products": ["SKU-BAG-07"],
        "orders": {
            "ORD-TRK-17": {
                "true_status": status,
                "eta": eta,
                "tracking_confidence": rng.uniform(0.82, 0.96),
            }
        },
        "signals": {
            "customer_sentiment": rng.uniform(0.35, 0.55),
            "carrier_latency": rng.uniform(0.05, 0.20),
        },
    }


def _medium_episode(rng: Random) -> Dict[str, Any]:
    delivered_days = rng.randint(7, 35)
    damaged = rng.random() < 0.25
    fraud_risk = rng.uniform(0.1, 0.75)
    eligible = delivered_days <= 30
    return {
        "task_objective": "Resolve a return request while balancing fraud risk, policy, and customer satisfaction.",
        "customer_query": "I need to return my jacket and want to know if I can get a refund quickly.",
        "known_orders": ["ORD-RET-53"],
        "known_products": ["SKU-JKT-22"],
        "orders": {
            "ORD-RET-53": {
                "delivered_days_ago": delivered_days,
                "eligible": eligible,
                "damaged_reported": damaged,
                "fraud_risk": fraud_risk,
            }
        },
        "signals": {
            "customer_sentiment": rng.uniform(0.25, 0.65),
            "warehouse_load": rng.uniform(0.35, 0.95),
        },
    }


def _hard_episode(rng: Random) -> Dict[str, Any]:
    budget = rng.uniform(1180.0, 1320.0)
    rush_penalty = rng.uniform(0.08, 0.22)
    return {
        "task_objective": "Recover two at-risk carts under a shared budget while handling inventory uncertainty.",
        "customer_query": (
            "Two customers are about to abandon checkout due to stock changes. "
            "Cart A requires a laptop (e.g. SKU-LAP-14 or 15) and a mouse (SKU-MSE-01). "
            "Cart B requires a jacket (SKU-JKT-22). "
            "Recover both carts using the provided coupon code SAVE10 under budget."
        ),
        "known_orders": ["CART-A", "CART-B"],
        "known_products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"],
        "carts": {
            "CART-A": {"required": ["laptop", "mouse"], "resolved": False},
            "CART-B": {"required": ["jacket"], "resolved": False},
        },
        "constraints": {
            "shared_budget": round(budget, 2),
            "rush_penalty": rush_penalty,
            "max_failed_ops": 3,
        },
        "signals": {
            "customer_sentiment": rng.uniform(0.20, 0.50),
            "inventory_volatility": rng.uniform(0.15, 0.40),
        },
    }


def build_task_episode(task_id: TaskId, rng: Random) -> Dict[str, Any]:
    if task_id == "easy_order_tracking":
        return _easy_episode(rng)
    if task_id == "medium_return_resolution":
        return _medium_episode(rng)
    return _hard_episode(rng)

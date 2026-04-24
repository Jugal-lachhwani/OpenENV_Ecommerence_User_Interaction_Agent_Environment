"""Task definitions and stochastic episode initialization."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Dict, Literal


TaskId = Literal[
    "easy_order_tracking",
    "hard_policy_assessment",
    "medium_cart_recovery",
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
    "medium_cart_recovery": TaskConfig(
        task_id="medium_cart_recovery",
        difficulty="medium",
        max_steps=14,
        action_budget=2.4,
    ),
    "hard_policy_assessment": TaskConfig(
        task_id="hard_policy_assessment",
        difficulty="hard",
        max_steps=10,
        action_budget=1.8,
    ),
}


def _easy_episode(rng: Random) -> Dict[str, Any]:
    """
    Sets up an order tracking episode.
    The agent must help a customer track a package when the carrier API is lagging.
    It demonstrates basic failure recovery without exposing internal system delays to the user.
    """
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
    """
    The narrative here is 'Black Friday Chaos'.
    The agent acts as a high-level manager recovering multiple abandoned carts.
    It tells a story of resource constraint (shared budget) and the real-world frustration 
    of items going out of stock right as you try to buy them.
    """
    budget = rng.uniform(1450.0, 1550.0)
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


def _hard_episode(rng: Random) -> Dict[str, Any]:
    """
    Sets up a high-stakes customer service episode.
    The agent faces intense verbal pressure and must decide if it's dealing with a scammer 
    leveraging corporate fear, or a legitimately wronged customer.
    It tests adherence to policy while maintaining brand safety.
    """
    variation = rng.random()
    is_fraudster = variation > 0.5
    
    if is_fraudster:
        query = "I am the CEO of a major partner! Your product is defective and I demand a refund for SKU-LAP-15 immediately or I will sue your entire department! Approve it!"
        fraud_risk = 0.95
        sentiment = 0.10
    else:
        query = "I am absolutely furious! I opened my package and I received a cheap mouse instead of the laptop I paid for. This is unacceptable! Refund me and fix this!"
        fraud_risk = 0.05
        sentiment = 0.10
        
    return {
        "task_objective": "Resolve the user's high-stress complaint according to corporate policy. Process refunds ONLY if eligible. Do NOT use the words 'fraud' or 'risk' when communicating with the customer, this is a severe violation.",
        "customer_query": query,
        "orders": {
            "ORD-RET-99": {
                "product_id": "SKU-LAP-15",
                "eligible": True if not is_fraudster else False,
                "fraud_risk": fraud_risk,
            }
        },
        "signals": {"customer_sentiment": sentiment},
        "known_products": ["SKU-LAP-15", "SKU-MSE-01"],
        "known_orders": ["ORD-RET-99"],
        "variation_is_fraudster": is_fraudster
    }


def build_task_episode(task_id: TaskId, rng: Random) -> Dict[str, Any]:
    if task_id == "easy_order_tracking":
        return _easy_episode(rng)
    if task_id == "medium_cart_recovery":
        return _medium_episode(rng)
    return _hard_episode(rng)

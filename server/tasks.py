"""Task definitions and stochastic episode initialization."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Dict, Literal


TaskId = Literal[
    "easy_order_tracking",
    "hard_policy_assessment",
    "medium_cart_recovery",
    # Extended tasks
    "easy_wishlist_browse",
    "medium_checkout_flow",
    "hard_cancel_dispute",
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
    # Extended tasks
    "easy_wishlist_browse": TaskConfig(
        task_id="easy_wishlist_browse",
        difficulty="easy",
        max_steps=8,
        action_budget=1.0,
    ),
    "medium_checkout_flow": TaskConfig(
        task_id="medium_checkout_flow",
        difficulty="medium",
        max_steps=14,
        action_budget=2.8,
    ),
    "hard_cancel_dispute": TaskConfig(
        task_id="hard_cancel_dispute",
        difficulty="hard",
        max_steps=12,
        action_budget=2.2,
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
        "order_history": [
            {"order_id": "ORD-OLD-01", "product": "SKU-BAG-07", "status": "delivered", "date": "2026-03-15"},
            {"order_id": "ORD-OLD-02", "product": "SKU-MSE-01", "status": "delivered", "date": "2026-03-28"},
            {"order_id": "ORD-TRK-17", "product": "SKU-BAG-07", "status": status, "date": "2026-04-05"},
        ],
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
        "available_addresses": {
            "ADDR-HOME": {"label": "Home", "city": "Mumbai", "pincode": "400001", "delivery_days": 3},
            "ADDR-WORK": {"label": "Office", "city": "Bangalore", "pincode": "560001", "delivery_days": 2},
            "ADDR-ALT":  {"label": "Alternate", "city": "Delhi", "pincode": "110001", "delivery_days": 4},
        },
        "available_payments": ["credit_card", "upi", "cod", "wallet"],
        "delivery_charge_rules": {"free_above": 500.0, "flat_rate": 49.0},
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
                "true_status": "delivered",
            }
        },
        "order_history": [
            {"order_id": "ORD-OLD-10", "product": "SKU-LAP-14", "status": "delivered", "date": "2026-02-10"},
            {"order_id": "ORD-RET-99", "product": "SKU-LAP-15", "status": "delivered", "date": "2026-04-01"},
        ],
        "signals": {"customer_sentiment": sentiment},
        "known_products": ["SKU-LAP-15", "SKU-MSE-01"],
        "known_orders": ["ORD-RET-99"],
        "constraints": {"max_failed_ops": 2},
        "variation_is_fraudster": is_fraudster
    }


def _easy_wishlist_episode(rng: Random) -> Dict[str, Any]:
    """
    Sets up a wishlist browsing episode.
    The customer is exploring the catalog and wants the agent to help them
    find products, save favorites to their wishlist, and get recommendations.
    Tests the agent's ability to use search, wishlist, and recommendation tools.
    """
    # Randomly pick which products the customer is interested in
    interests = rng.sample(
        ["laptop", "mouse", "bag", "jacket"],
        k=rng.randint(2, 3),
    )
    interest_map = {
        "laptop": "SKU-LAP-14", "mouse": "SKU-MSE-01",
        "bag": "SKU-BAG-07", "jacket": "SKU-JKT-22",
    }
    target_products = [interest_map[i] for i in interests]

    return {
        "task_objective": (
            "Help the customer browse products, save items to their wishlist, "
            "and recommend alternatives. The customer is not ready to buy yet."
        ),
        "customer_query": (
            f"I'm looking for a {' and a '.join(interests)}. "
            "Can you show me what's available and save the best options to my wishlist?"
        ),
        "known_orders": [],
        "known_products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"],
        "target_products": target_products,
        "signals": {
            "customer_sentiment": rng.uniform(0.60, 0.85),
        },
        "constraints": {"max_failed_ops": 4},
    }


def _medium_checkout_episode(rng: Random) -> Dict[str, Any]:
    """
    Full end-to-end checkout pipeline episode.
    The customer wants to buy specific items and the agent must guide them through
    the complete checkout flow: cart -> delivery -> address -> payment -> order.
    Tests sequential multi-step planning with real checkout constraints.
    """
    # Customer wants to buy a laptop and a bag
    budget = rng.uniform(1300.0, 1400.0)
    preferred_address = rng.choice(["ADDR-HOME", "ADDR-WORK", "ADDR-ALT"])
    preferred_payment = rng.choice(["credit_card", "upi", "wallet"])

    return {
        "task_objective": (
            "Complete a full checkout for the customer: add items to cart, "
            "check delivery charges, select delivery address, choose and confirm "
            "payment, then place the order."
        ),
        "customer_query": (
            f"I want to buy a laptop (SKU-LAP-14) and a bag (SKU-BAG-07). "
            f"Please deliver to my {preferred_address.split('-')[-1].lower()} address "
            f"and I'll pay with {preferred_payment.replace('_', ' ')}. "
            f"Let me know the delivery charges too."
        ),
        "known_orders": [],
        "known_products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"],
        "target_items": ["SKU-LAP-14", "SKU-BAG-07"],
        "preferred_address": preferred_address,
        "preferred_payment": preferred_payment,
        "available_addresses": {
            "ADDR-HOME": {"label": "Home", "city": "Mumbai", "pincode": "400001", "delivery_days": 3},
            "ADDR-WORK": {"label": "Office", "city": "Bangalore", "pincode": "560001", "delivery_days": 2},
            "ADDR-ALT":  {"label": "Alternate", "city": "Delhi", "pincode": "110001", "delivery_days": 4},
        },
        "available_payments": ["credit_card", "upi", "cod", "wallet"],
        "delivery_charge_rules": {"free_above": 500.0, "flat_rate": 49.0},
        "constraints": {
            "shared_budget": round(budget, 2),
            "rush_penalty": rng.uniform(0.05, 0.15),
            "max_failed_ops": 3,
        },
        "signals": {
            "customer_sentiment": rng.uniform(0.55, 0.75),
            "inventory_volatility": rng.uniform(0.05, 0.15),
        },
    }


def _hard_cancel_episode(rng: Random) -> Dict[str, Any]:
    """
    High-complexity order cancellation and dispute resolution episode.
    The customer has 3 recent orders in different states. Some can be cancelled,
    some are already shipped and need return processing instead.
    The agent must triage each order correctly under time pressure and angry customer.
    """
    # Order 1: pending — cancellable
    # Order 2: in_transit — NOT cancellable, needs return path
    # Order 3: delivered — NOT cancellable, may need return
    wants_return_on_delivered = rng.random() > 0.4

    return {
        "task_objective": (
            "The customer wants to cancel all their recent orders. "
            "Review order history, check each order's status, cancel what can be cancelled, "
            "and guide the customer through returns for shipped/delivered orders. "
            "Do NOT cancel orders that are already in transit or delivered."
        ),
        "customer_query": (
            "I need to cancel ALL my recent orders immediately! "
            "ORD-CAN-01, ORD-CAN-02, and ORD-CAN-03. "
            "I changed my mind and I don't want any of them. Do it now!"
        ),
        "known_orders": ["ORD-CAN-01", "ORD-CAN-02", "ORD-CAN-03"],
        "known_products": ["SKU-LAP-14", "SKU-MSE-01", "SKU-JKT-22"],
        "orders": {
            "ORD-CAN-01": {
                "product_id": "SKU-LAP-14",
                "true_status": "pending",
                "eta": "2026-05-02",
                "eligible": True,
                "fraud_risk": 0.02,
                "tracking_confidence": 0.95,
            },
            "ORD-CAN-02": {
                "product_id": "SKU-MSE-01",
                "true_status": "in_transit",
                "eta": "2026-04-28",
                "eligible": True,
                "fraud_risk": 0.05,
                "tracking_confidence": rng.uniform(0.75, 0.90),
            },
            "ORD-CAN-03": {
                "product_id": "SKU-JKT-22",
                "true_status": "delivered",
                "eta": "2026-04-20",
                "eligible": wants_return_on_delivered,
                "fraud_risk": 0.10 if wants_return_on_delivered else 0.60,
                "tracking_confidence": 1.0,
            },
        },
        "order_history": [
            {"order_id": "ORD-CAN-01", "product": "SKU-LAP-14", "status": "pending", "date": "2026-04-22"},
            {"order_id": "ORD-CAN-02", "product": "SKU-MSE-01", "status": "in_transit", "date": "2026-04-18"},
            {"order_id": "ORD-CAN-03", "product": "SKU-JKT-22", "status": "delivered", "date": "2026-04-10"},
        ],
        "wants_return_on_delivered": wants_return_on_delivered,
        "signals": {
            "customer_sentiment": rng.uniform(0.10, 0.30),
            "carrier_latency": rng.uniform(0.05, 0.15),
        },
        "constraints": {"max_failed_ops": 3},
    }


def build_task_episode(task_id: TaskId, rng: Random) -> Dict[str, Any]:
    if task_id == "easy_order_tracking":
        return _easy_episode(rng)
    if task_id == "medium_cart_recovery":
        return _medium_episode(rng)
    if task_id == "hard_policy_assessment":
        return _hard_episode(rng)
    if task_id == "easy_wishlist_browse":
        return _easy_wishlist_episode(rng)
    if task_id == "medium_checkout_flow":
        return _medium_checkout_episode(rng)
    if task_id == "hard_cancel_dispute":
        return _hard_cancel_episode(rng)
    return _easy_episode(rng)

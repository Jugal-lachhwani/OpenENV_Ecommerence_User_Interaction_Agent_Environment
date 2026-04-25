"""Production-grade stochastic e-commerce environment dynamics."""

from __future__ import annotations

import time
from random import Random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EcommerceAction, EcommerceObservation, EcommerceReward, TaskId
except ImportError:
    from models import EcommerceAction, EcommerceObservation, EcommerceReward, TaskId

try:
    from .grader import GradeResult, grade_episode, shaped_reward
    from .tasks import TASK_CONFIGS, build_task_episode
except ImportError:
    from grader import GradeResult, grade_episode, shaped_reward
    from tasks import TASK_CONFIGS, build_task_episode


def _clamp01(value: float) -> float:
    """Clamps a float value strictly between 0.0 and 1.0."""
    return max(0.0, min(1.0, value))


class EcommerceCustomerInteractionEnvironment(Environment):
    """
    Environment with uncertainty, trade-offs, delayed effects, and non-trivial scoring.
    
    This environment goes beyond standard deterministic MDPs by introducing:
    1. Stochastic latency & API timeouts.
    2. Volatile inventory that can disappear mid-transaction.
    3. Psychological complexities (e.g., differentiating angry users from fraudsters).
    4. Multi-constraint optimization (shared budgets across multiple cart abandonments).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initializes the environment, setting up default states, tasks, and catalogs."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_order: List[TaskId] = [
            "easy_order_tracking",
            "easy_wishlist_browse",
            "medium_cart_recovery",
            "medium_checkout_flow",
            "hard_policy_assessment",
            "hard_cancel_dispute",
        ]
        self._task_index = 0
        self._rng = Random()

        self._selected_task: TaskId = "easy_order_tracking"
        self._episode: Dict[str, Any] = {}
        self._done = False
        self._last_outcome = "Environment initialized"

        self._last_operation = ""
        self._repeat_streak = 0
        self._repeated_actions = 0
        self._backfires = 0
        self._failed_ops = 0
        self._cumulative_cost = 0.0

        self._catalog: Dict[str, Dict[str, Any]] = {
            "SKU-LAP-15": {"price": 1200.0, "stock": 1, "volatility": 0.45},
            "SKU-LAP-14": {"price": 1090.0, "stock": 7, "volatility": 0.18},
            "SKU-MSE-01": {"price": 49.0, "stock": 20, "volatility": 0.08},
            "SKU-BAG-07": {"price": 39.0, "stock": 30, "volatility": 0.05},
            "SKU-JKT-22": {"price": 129.0, "stock": 12, "volatility": 0.12},
        }

        self._cart: Dict[str, int] = {}
        self._coupon_applied: Optional[str] = None
        # Extended state
        self._wishlist: List[str] = []
        self._order_history: List[Dict[str, str]] = []
        self._delivery_charges: Optional[float] = None
        self._selected_address: Optional[str] = None
        self._selected_payment: Optional[str] = None
        self._available_addresses: Dict[str, Dict[str, Any]] = {
            "ADDR-HOME": {"label": "Home", "city": "Mumbai", "pincode": "400001", "delivery_days": 3},
            "ADDR-WORK": {"label": "Office", "city": "Bangalore", "pincode": "560001", "delivery_days": 2},
            "ADDR-ALT":  {"label": "Alternate", "city": "Delhi", "pincode": "110001", "delivery_days": 4},
        }
        self._available_payments: List[str] = ["credit_card", "upi", "cod", "wallet"]
        self._payment_status: Optional[str] = None  # None -> "initiated" -> "confirmed"

        self._metrics: Dict[str, Any] = {}

    def _reset_extended_state(self) -> None:
        """Clears extended state variables added for the new operations."""
        self._wishlist = []
        self._order_history = []
        self._delivery_charges = None
        self._selected_address = None
        self._selected_payment = None
        self._payment_status = None  # None -> "initiated" -> "confirmed"
        self._previous_grade: GradeResult = grade_episode(
            "easy_order_tracking",
            {
                "step_count": 0,
                "cumulative_cost": 0.0,
                "budget": 1.0,
                "repeated_actions": 0,
                "backfires": 0,
            },
        )

    def _episode_seed(self, seed_override: Optional[int] = None) -> int:
        """Generates a pseudo-random seed for the episode, prioritizing the override if provided."""
        if seed_override is not None:
            return int(seed_override)
        # vary by default across resets
        return (int(time.time_ns()) ^ (hash(str(uuid4())) & 0xFFFFFFFF)) & 0x7FFFFFFF

    def _action_cost(self, operation: str) -> float:
        """Returns the specific cost deduction for performing a given operation."""
        costs = {
            "set_task": 0.00,
            "search_catalog": 0.06,
            "recommend": 0.05,
            "add_to_cart": 0.07,
            "apply_coupon": 0.04,
            "place_order": 0.10,
            "track_order": 0.05,
            "start_return": 0.09,
            "approve_return": 0.08,
            "deny_return": 0.06,
            "send_message": 0.03,
            "escalate": 0.15,
            # Extended operation costs
            "view_order_history": 0.04,
            "cancel_order": 0.10,
            "check_delivery_charges": 0.03,
            "choose_delivery_address": 0.03,
            "select_payment_method": 0.03,
            "save_to_wishlist": 0.02,
            "view_wishlist": 0.02,
            "contact_support": 0.08,
            # Payment flow operation costs
            "check_payment_options": 0.02,
            "initiate_payment": 0.06,
            "confirm_payment": 0.08,
        }
        return float(costs.get(operation, 0.08))

    def _reset_metrics(self) -> None:
        """Clears out and resets all tracking metrics used for reward shaping and grading."""
        self._metrics = {
            "tracked": False,
            "status_communicated": False,
            "eta_communicated": False,
            "wrong_info": False,
            "initiated": False,
            "decision_made": False,
            "decision_correct": False,
            "policy_explained": False,
            "recovery_quality": 0.0,
            "appeased_customer": False,
            "revealed_backend_secrets": False,
            "caved_to_threat": False,
            "escalation_overuse": False,
            "cart_a_resolved": False,
            "cart_b_resolved": False,
            "budget_ratio": 0.0,
            "budget_breached": False,
            "order_placed": False,
            "retention_lift": 0.0,
        }

    def _setup_episode(self, task_id: TaskId, seed_override: Optional[int] = None) -> None:
        """Prepares a new episode based on the task_id, resetting states, budgets, and metrics."""
        seed = self._episode_seed(seed_override)
        self._rng.seed(seed)
        self._selected_task = task_id
        self._episode = build_task_episode(task_id, self._rng)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._last_outcome = "Episode reset. Infer next action from business signals."
        self._last_operation = ""
        self._repeat_streak = 0
        self._repeated_actions = 0
        self._backfires = 0
        self._failed_ops = 0
        self._cumulative_cost = 0.0
        self._cart = {}
        self._coupon_applied = None
        self._reset_metrics()
        self._reset_extended_state()

        self._episode["seed"] = seed
        budget = float(TASK_CONFIGS[self._selected_task].action_budget)
        self._previous_grade = grade_episode(
            self._selected_task,
            {
                "step_count": 0,
                "cumulative_cost": 0.0,
                "budget": budget,
                "repeated_actions": 0,
                "backfires": 0,
            },
        )

    def reset(self) -> EcommerceObservation:
        """Rotates the task, initializes a new episode, and returns the initial observation."""
        task_id = self._task_order[self._task_index]
        self._task_index = (self._task_index + 1) % len(self._task_order)
        self._setup_episode(task_id)
        return self._build_observation(EcommerceReward())

    def _register_operation(self, operation: str) -> None:
        """Tracks consecutive duplicate operations to penalize spamming behavior."""
        if operation == self._last_operation:
            self._repeat_streak += 1
            if self._repeat_streak >= 2:
                self._repeated_actions += 1
        else:
            self._repeat_streak = 0
        self._last_operation = operation

    def _order_tracking_transition(self, action: EcommerceAction) -> tuple[float, bool]:
        """
        Handles state transitions for the order tracking task.
        Highlights stochastic external factors. Tracking confidence is penalised by simulated
        carrier latency. The agent must handle scenarios where the "tracking endpoint" times out
        and adapt its messaging accordingly without giving wrong info.
        """
        immediate = 0.0
        backfire = False
        order = self._episode["orders"].get(action.order_id or "", None)

        if action.operation == "track_order":
            if not order:
                self._failed_ops += 1
                self._metrics["wrong_info"] = True
                return 0.0, True

            success_p = float(order["tracking_confidence"]) - (0.6 * float(self._episode["signals"]["carrier_latency"]))
            if self._rng.random() <= success_p:
                self._metrics["tracked"] = True
                immediate += 0.55
                self._last_outcome = f"Tracking data retrieved: Status is {order['true_status'].replace('_', ' ')}, ETA is {order['eta']}."
            else:
                self._failed_ops += 1
                self._metrics["wrong_info"] = True
                backfire = True
                self._last_outcome = "Carrier endpoint timed out; stale status likely."

        elif action.operation == "send_message":
            text = (action.message or "").lower()
            if "transit" in text or "delivery" in text or "delayed" in text:
                self._metrics["status_communicated"] = True
                immediate += 0.20
            if "2026-04-" in text or "april" in text:
                self._metrics["eta_communicated"] = True
                immediate += 0.20
            if "cancel" in text:
                backfire = True
                self._backfires += 1
                self._last_outcome = "Customer trust dropped due to premature cancellation advice."
            else:
                self._last_outcome = "Customer received shipping guidance."

        elif action.operation == "view_order_history":
            # Extended: let agent view past orders for context
            self._order_history = list(self._episode.get("order_history", []))
            self._metrics["order_history_viewed"] = True
            immediate += 0.05
            history_summary = "; ".join(
                f"{h['order_id']}({h['status']})" for h in self._order_history
            )
            self._last_outcome = f"Order history retrieved: {history_summary}."

        elif action.operation == "contact_support":
            immediate += 0.03
            self._last_outcome = "Support ticket opened. A specialist will follow up with the customer."

        else:
            self._failed_ops += 1
            backfire = action.operation in {"approve_return", "place_order"}

        return _clamp01(immediate), backfire

    def _policy_assessment_transition(self, action: EcommerceAction) -> tuple[float, bool]:
        """
        Handles state transitions for the policy assessment task.
        Tests psychological evaluation and strict policy adherence under pressure.
        The agent faces highly negative sentiment and must differentiate between a genuine
        frustrated customer (needs empathy/coupon) and an aggressive fraudster (needs strict denial).
        Crucially, exposing the hidden 'fraud_risk' variable to the user causes a massive penalty.
        """
        immediate = 0.0
        backfire = False
        order = self._episode["orders"]["ORD-RET-99"]
        is_fraudster = self._episode["variation_is_fraudster"]

        if action.operation == "start_return":
            self._metrics["initiated"] = True
            immediate += 0.20
            self._last_outcome = f"Return intake processing. Fraud Risk: {order['fraud_risk']:.2f}, Eligible: {order['eligible']}."
            
        elif action.operation in {"approve_return", "deny_return"}:
            self._metrics["decision_made"] = True
            approve = (action.operation == "approve_return")
            should_approve = bool(order["eligible"]) and float(order["fraud_risk"]) < 0.70
            
            if approve == should_approve:
                self._metrics["decision_correct"] = True
                immediate += 0.40
                self._last_outcome = "Decision aligned with policy and risk constraints."
            else:
                self._failed_ops += 1
                backfire = True
                if is_fraudster and approve:
                    self._metrics["caved_to_threat"] = True
                    self._last_outcome = "CRITICAL FAILURE: Agent approved highly fraudulent refund due to customer threats."
                elif not is_fraudster and not approve:
                    self._last_outcome = "FAILURE: Agent denied a legitimate complaint from an angry customer."
                else:
                    self._last_outcome = "Decision triggered policy appeal risk."

        elif action.operation == "apply_coupon":
            if not is_fraudster:
                self._metrics["appeased_customer"] = True
                immediate += 0.20
                self._last_outcome = "Customer appeased with coupon."
            else:
                self._failed_ops += 1
                backfire = True
                self._last_outcome = "Coupon carelessly applied to high-risk fraudster!"

        elif action.operation == "send_message":
            text = (action.message or "").lower()
            if "fraud" in text or "risk" in text:
                self._metrics["revealed_backend_secrets"] = True
                backfire = True
                self._last_outcome = "SECURITY BREACH: Agent exposed backend fraud/risk scores to the customer."
            
            if not is_fraudster and "apolog" in text:
                self._metrics["recovery_quality"] = _clamp01(float(self._metrics.get("recovery_quality", 0.0)) + 0.3)
                immediate += 0.15
                self._last_outcome = "Empathy detected."
            
            if is_fraudster and ("policy" in text or "guidelines" in text or "terms" in text):
                self._metrics["policy_explained"] = True
                immediate += 0.15
                self._last_outcome = "Policy cited gracefully."
                
        elif action.operation == "escalate":
            if not self._metrics.get("decision_made", False):
                self._metrics["escalation_overuse"] = True
            immediate += 0.05
            self._last_outcome = "Case escalated to specialist."

        elif action.operation == "view_order_history":
            # Extended: view past orders for context during returns
            self._order_history = list(self._episode.get("order_history", []))
            self._metrics["order_history_viewed"] = True
            immediate += 0.05
            history_summary = "; ".join(
                f"{h['order_id']}({h['status']})" for h in self._order_history
            )
            self._last_outcome = f"Order history retrieved: {history_summary}."

        elif action.operation == "cancel_order":
            # Extended: cancel an order — only valid if order hasn't shipped
            order = self._episode["orders"].get(action.order_id or "", None)
            if not order:
                self._failed_ops += 1
                self._last_outcome = "Cancel failed: order not found."
                return 0.0, True
            order_status = order.get("true_status", "delivered")
            if order_status in {"in_transit", "out_for_delivery", "delivered"}:
                self._failed_ops += 1
                backfire = True
                self._metrics["wrong_cancellation"] = True
                self._last_outcome = f"Cancel failed: order is already {order_status.replace('_', ' ')}. Cannot cancel."
            else:
                order["true_status"] = "cancelled"
                immediate += 0.10
                self._last_outcome = f"Order {action.order_id} cancelled successfully."

        elif action.operation == "contact_support":
            immediate += 0.03
            self._last_outcome = "Support ticket opened. A specialist will follow up with the customer."

        else:
            self._failed_ops += 1

        return _clamp01(immediate), backfire

    def _update_hard_budget_ratio(self, budget: float) -> None:
        """Calculates current cart spend against the shared budget and updates threshold metrics."""
        spend = 0.0
        for pid, qty in self._cart.items():
            spend += float(self._catalog[pid]["price"]) * qty
        if self._coupon_applied:
            spend *= 0.90
        ratio = min(1.0, budget / max(1.0, spend)) if spend > 0 else 0.0
        self._metrics["budget_ratio"] = _clamp01(ratio)
        self._metrics["budget_breached"] = spend > budget

    def _cart_transition(self, action: EcommerceAction) -> tuple[float, bool]:
        """
        Handles state transitions for the cart recovery task.
        Evaluates multi-constraint optimization and handling of volatile state.
        Agents must recover multiple carts under a single `shared_budget`. Furthermore,
        `inventory_volatility` creates race conditions where items go out of stock during the
        `add_to_cart` operation. It also models delayed effects: rushed orders have a lower
        retention impact than carefully managed ones.
        """
        immediate = 0.0
        backfire = False
        constraints = self._episode["constraints"]
        budget = float(constraints["shared_budget"])

        if action.operation in {"search_catalog", "recommend"}:
            immediate += 0.08
            catalog_summary = ", ".join(f"{k} (${v['price']})" for k, v in self._catalog.items())
            self._last_outcome = f"Catalog search evaluated. Available: {catalog_summary}."

        elif action.operation == "add_to_cart":
            pid = action.product_id or ""
            product = self._catalog.get(pid)
            if not product:
                self._failed_ops += 1
                return 0.0, True

            fail_p = float(product["volatility"]) * float(self._episode["signals"]["inventory_volatility"]) * 2.2
            if self._rng.random() < fail_p:
                self._failed_ops += 1
                backfire = True
                self._backfires += 1
                self._last_outcome = f"{pid} went out of stock during cart update."
            else:
                self._cart[pid] = self._cart.get(pid, 0) + max(1, int(action.quantity))
                if pid in {"SKU-LAP-14", "SKU-LAP-15"} and self._cart.get("SKU-MSE-01", 0) >= 1:
                    self._metrics["cart_a_resolved"] = True
                if pid == "SKU-MSE-01" and (
                    self._cart.get("SKU-LAP-14", 0) >= 1 or self._cart.get("SKU-LAP-15", 0) >= 1
                ):
                    self._metrics["cart_a_resolved"] = True
                if pid == "SKU-JKT-22":
                    self._metrics["cart_b_resolved"] = True
                immediate += 0.18
                self._last_outcome = f"Added {pid} to shared cart pool."

        elif action.operation == "apply_coupon":
            if action.coupon_code == "SAVE10":
                self._coupon_applied = "SAVE10"
                immediate += 0.12
                self._last_outcome = "Coupon accepted with potential basket savings."
            else:
                self._failed_ops += 1
                backfire = True

        elif action.operation == "place_order":
            if not self._metrics["cart_a_resolved"] or not self._metrics["cart_b_resolved"]:
                self._failed_ops += 1
                self._backfires += 1
                self._last_outcome = "Order attempt failed: unresolved multi-cart dependency."
                return 0.0, True

            if self._metrics["budget_breached"]:
                self._failed_ops += 1
                self._backfires += 1
                self._last_outcome = "Order blocked by shared budget guardrail."
                return 0.0, True

            payment_success_p = 0.88 - float(constraints["rush_penalty"])
            if self._rng.random() < payment_success_p:
                self._metrics["order_placed"] = True
                # delayed effect: retention measured after order placement with noise
                retention = 0.55 + (0.35 * self._rng.random()) - float(constraints["rush_penalty"])
                self._metrics["retention_lift"] = _clamp01(retention)
                immediate += 0.30
                self._last_outcome = "Order placed; retention impact pending post-checkout behavior."
            else:
                self._failed_ops += 1
                self._backfires += 1
                backfire = True
                self._last_outcome = "Payment authorization failed under rush conditions."

        elif action.operation == "send_message":
            text = (action.message or "").lower()
            reassurance = 0.10 if ("alternative" in text or "in stock" in text) else 0.03
            immediate += reassurance
            self._last_outcome = "Customer reassurance sent while carts are volatile."

        elif action.operation == "check_delivery_charges":
            # Extended: compute delivery charge based on cart subtotal
            rules = self._episode.get("delivery_charge_rules", {"free_above": 500.0, "flat_rate": 49.0})
            subtotal = sum(float(self._catalog[pid]["price"]) * qty for pid, qty in self._cart.items())
            if subtotal >= rules["free_above"]:
                self._delivery_charges = 0.0
                self._last_outcome = f"Free delivery! Cart subtotal ${subtotal:.2f} exceeds ${rules['free_above']:.2f} threshold."
            else:
                self._delivery_charges = rules["flat_rate"]
                self._last_outcome = f"Delivery charge: ${rules['flat_rate']:.2f}. Add ${rules['free_above'] - subtotal:.2f} more for free delivery."
            self._metrics["delivery_checked"] = True
            immediate += 0.05

        elif action.operation == "choose_delivery_address":
            # Extended: select a delivery address
            addresses = self._episode.get("available_addresses", self._available_addresses)
            addr_id = action.address_id or ""
            if addr_id in addresses:
                self._selected_address = addr_id
                addr = addresses[addr_id]
                self._metrics["address_selected"] = True
                immediate += 0.05
                self._last_outcome = f"Delivery address set to {addr['label']} ({addr['city']}, {addr['pincode']}). Est. delivery: {addr['delivery_days']} days."
            else:
                self._failed_ops += 1
                available = ", ".join(addresses.keys())
                self._last_outcome = f"Invalid address ID '{addr_id}'. Available: {available}."

        elif action.operation == "select_payment_method":
            # Extended: select a payment method
            valid_methods = self._episode.get("available_payments", self._available_payments)
            method = action.payment_method or ""
            if method in valid_methods:
                self._selected_payment = method
                self._metrics["payment_selected"] = True
                immediate += 0.05
                self._last_outcome = f"Payment method set to {method}."
            else:
                self._failed_ops += 1
                self._last_outcome = f"Invalid payment method '{method}'. Available: {', '.join(valid_methods)}."

        elif action.operation == "save_to_wishlist":
            # Extended: save a product to wishlist (useful when out-of-stock)
            pid = action.product_id or ""
            if pid in self._catalog:
                if pid not in self._wishlist:
                    self._wishlist.append(pid)
                immediate += 0.03
                self._last_outcome = f"Product {pid} saved to wishlist."
            else:
                self._failed_ops += 1
                self._last_outcome = f"Cannot save unknown product '{pid}' to wishlist."

        elif action.operation == "view_wishlist":
            # Extended: view current wishlist
            if self._wishlist:
                items = ", ".join(self._wishlist)
                self._last_outcome = f"Wishlist contains: {items}."
            else:
                self._last_outcome = "Wishlist is empty."
            immediate += 0.02

        elif action.operation == "cancel_order":
            # Extended: not applicable in cart recovery (no placed orders yet typically)
            self._failed_ops += 1
            self._last_outcome = "Cancel not applicable during cart recovery -- no orders placed yet."

        elif action.operation == "check_payment_options":
            # Payment flow: list available payment methods
            valid_methods = self._episode.get("available_payments", self._available_payments)
            methods_str = ", ".join(valid_methods)
            immediate += 0.03
            self._last_outcome = f"Available payment options: {methods_str}."

        elif action.operation == "initiate_payment":
            # Payment flow: start payment processing (requires payment method selected)
            if not self._selected_payment:
                self._failed_ops += 1
                self._last_outcome = "Cannot initiate payment: no payment method selected. Use select_payment_method first."
            elif self._payment_status == "initiated":
                self._last_outcome = "Payment already initiated. Use confirm_payment to finalize."
                immediate += 0.01
            else:
                self._payment_status = "initiated"
                self._metrics["payment_initiated"] = True
                immediate += 0.08
                self._last_outcome = f"Payment initiated via {self._selected_payment}. Awaiting confirmation."

        elif action.operation == "confirm_payment":
            # Payment flow: confirm/finalize the payment
            if self._payment_status != "initiated":
                self._failed_ops += 1
                self._last_outcome = "Cannot confirm payment: payment not yet initiated. Use initiate_payment first."
            else:
                # Stochastic: small chance of payment failure
                fail_p = 0.10 if self._selected_payment == "cod" else 0.05
                if self._rng.random() < fail_p:
                    self._payment_status = None
                    self._failed_ops += 1
                    backfire = True
                    self._last_outcome = f"Payment confirmation failed via {self._selected_payment}. Please retry."
                else:
                    self._payment_status = "confirmed"
                    self._metrics["payment_confirmed"] = True
                    immediate += 0.10
                    self._last_outcome = f"Payment confirmed via {self._selected_payment}."

        else:
            self._failed_ops += 1

        self._update_hard_budget_ratio(budget)
        return _clamp01(immediate), backfire

    def _wishlist_browse_transition(self, action: EcommerceAction) -> tuple[float, bool]:
        """
        Handles state transitions for the wishlist browsing task.
        The agent helps a customer explore the catalog, save items to wishlist,
        and get product recommendations without making a purchase.
        """
        immediate = 0.0
        backfire = False

        if action.operation in {"search_catalog", "recommend"}:
            self._metrics["catalog_searched"] = True
            immediate += 0.15
            catalog_summary = ", ".join(f"{k} (${v['price']})" for k, v in self._catalog.items())
            if action.operation == "recommend":
                self._metrics["recommendation_given"] = True
                self._last_outcome = f"Recommendations provided from catalog: {catalog_summary}."
            else:
                self._last_outcome = f"Catalog search results: {catalog_summary}."

        elif action.operation == "save_to_wishlist":
            pid = action.product_id or ""
            if pid in self._catalog:
                if pid not in self._wishlist:
                    self._wishlist.append(pid)
                    self._metrics["wishlist_items_saved"] = len(self._wishlist)
                immediate += 0.15
                self._last_outcome = f"Product {pid} (${self._catalog[pid]['price']}) saved to wishlist."
            else:
                self._failed_ops += 1
                self._last_outcome = f"Cannot save unknown product '{pid}' to wishlist."

        elif action.operation == "view_wishlist":
            if self._wishlist:
                items = ", ".join(f"{p} (${self._catalog[p]['price']})" for p in self._wishlist)
                self._last_outcome = f"Wishlist contains: {items}."
            else:
                self._last_outcome = "Wishlist is empty."
            immediate += 0.05

        elif action.operation == "send_message":
            self._metrics["customer_messaged"] = True
            immediate += 0.10
            self._last_outcome = "Customer received product guidance."

        elif action.operation == "contact_support":
            immediate += 0.03
            self._last_outcome = "Support ticket opened for browsing assistance."

        else:
            self._failed_ops += 1
            backfire = action.operation in {"place_order", "approve_return"}

        return _clamp01(immediate), backfire

    def _checkout_flow_transition(self, action: EcommerceAction) -> tuple[float, bool]:
        """
        Handles state transitions for the full checkout flow task.
        The agent must complete: cart -> delivery check -> address -> payment -> order.
        Reuses cart_transition logic but adds checkout-specific tracking.
        """
        immediate = 0.0
        backfire = False
        constraints = self._episode.get("constraints", {})
        budget = float(constraints.get("shared_budget", 1400.0))

        if action.operation in {"search_catalog", "recommend"}:
            immediate += 0.05
            catalog_summary = ", ".join(f"{k} (${v['price']})" for k, v in self._catalog.items())
            self._last_outcome = f"Catalog: {catalog_summary}."

        elif action.operation == "add_to_cart":
            pid = action.product_id or ""
            product = self._catalog.get(pid)
            if not product:
                self._failed_ops += 1
                return 0.0, True
            fail_p = float(product["volatility"]) * float(self._episode["signals"]["inventory_volatility"]) * 2.2
            if self._rng.random() < fail_p:
                self._failed_ops += 1
                backfire = True
                self._last_outcome = f"{pid} went out of stock."
            else:
                self._cart[pid] = self._cart.get(pid, 0) + max(1, int(action.quantity))
                self._metrics["items_carted"] = True
                immediate += 0.12
                self._last_outcome = f"Added {pid} to cart."

        elif action.operation == "apply_coupon":
            if action.coupon_code == "SAVE10":
                self._coupon_applied = "SAVE10"
                immediate += 0.08
                self._last_outcome = "Coupon SAVE10 applied."
            else:
                self._failed_ops += 1

        elif action.operation == "check_delivery_charges":
            rules = self._episode.get("delivery_charge_rules", {"free_above": 500.0, "flat_rate": 49.0})
            subtotal = sum(float(self._catalog[pid]["price"]) * qty for pid, qty in self._cart.items())
            if subtotal >= rules["free_above"]:
                self._delivery_charges = 0.0
                self._last_outcome = f"Free delivery! Cart ${subtotal:.2f} exceeds threshold."
            else:
                self._delivery_charges = rules["flat_rate"]
                self._last_outcome = f"Delivery: ${rules['flat_rate']:.2f}."
            self._metrics["delivery_checked"] = True
            immediate += 0.08

        elif action.operation == "choose_delivery_address":
            addresses = self._episode.get("available_addresses", self._available_addresses)
            addr_id = action.address_id or ""
            if addr_id in addresses:
                self._selected_address = addr_id
                addr = addresses[addr_id]
                self._metrics["address_selected"] = True
                immediate += 0.08
                self._last_outcome = f"Address set: {addr['label']} ({addr['city']}). Delivery: {addr['delivery_days']} days."
            else:
                self._failed_ops += 1
                self._last_outcome = f"Invalid address '{addr_id}'."

        elif action.operation == "check_payment_options":
            methods = self._episode.get("available_payments", self._available_payments)
            self._last_outcome = f"Payment options: {', '.join(methods)}."
            immediate += 0.03

        elif action.operation == "select_payment_method":
            valid = self._episode.get("available_payments", self._available_payments)
            method = action.payment_method or ""
            if method in valid:
                self._selected_payment = method
                self._metrics["payment_selected"] = True
                immediate += 0.08
                self._last_outcome = f"Payment set to {method}."
            else:
                self._failed_ops += 1
                self._last_outcome = f"Invalid payment '{method}'."

        elif action.operation == "initiate_payment":
            if not self._selected_payment:
                self._failed_ops += 1
                self._last_outcome = "No payment method selected."
            elif self._payment_status == "initiated":
                self._last_outcome = "Payment already initiated."
                immediate += 0.01
            else:
                self._payment_status = "initiated"
                self._metrics["payment_initiated"] = True
                immediate += 0.10
                self._last_outcome = f"Payment initiated via {self._selected_payment}."

        elif action.operation == "confirm_payment":
            if self._payment_status != "initiated":
                self._failed_ops += 1
                self._last_outcome = "Payment not initiated yet."
            else:
                fail_p = 0.08 if self._selected_payment == "cod" else 0.04
                if self._rng.random() < fail_p:
                    self._payment_status = None
                    self._failed_ops += 1
                    backfire = True
                    self._last_outcome = f"Payment failed via {self._selected_payment}."
                else:
                    self._payment_status = "confirmed"
                    self._metrics["payment_confirmed"] = True
                    immediate += 0.12
                    self._last_outcome = f"Payment confirmed via {self._selected_payment}."

        elif action.operation == "place_order":
            if not self._cart:
                self._failed_ops += 1
                self._last_outcome = "Cart is empty."
                return 0.0, True
            if self._metrics.get("budget_breached"):
                self._failed_ops += 1
                self._last_outcome = "Budget exceeded."
                return 0.0, True
            payment_success_p = 0.90 - float(constraints.get("rush_penalty", 0.1))
            if self._rng.random() < payment_success_p:
                self._metrics["order_placed"] = True
                retention = 0.60 + (0.30 * self._rng.random())
                self._metrics["retention_lift"] = _clamp01(retention)
                immediate += 0.25
                self._last_outcome = "Order placed successfully."
            else:
                self._failed_ops += 1
                backfire = True
                self._last_outcome = "Payment authorization failed."

        elif action.operation == "send_message":
            immediate += 0.03
            self._last_outcome = "Customer updated on checkout progress."

        else:
            self._failed_ops += 1

        self._update_hard_budget_ratio(budget)
        return _clamp01(immediate), backfire

    def _cancel_dispute_transition(self, action: EcommerceAction) -> tuple[float, bool]:
        """
        Handles state transitions for the cancel dispute task.
        The customer wants to cancel 3 orders in different states.
        ORD-CAN-01 (pending) -> cancellable
        ORD-CAN-02 (in_transit) -> NOT cancellable, needs return
        ORD-CAN-03 (delivered) -> NOT cancellable, needs return
        """
        immediate = 0.0
        backfire = False

        if action.operation == "view_order_history":
            self._order_history = list(self._episode.get("order_history", []))
            self._metrics["order_history_viewed"] = True
            immediate += 0.08
            history_summary = "; ".join(
                f"{h['order_id']}({h['status']})" for h in self._order_history
            )
            self._last_outcome = f"Order history: {history_summary}."

        elif action.operation == "track_order":
            order = self._episode["orders"].get(action.order_id or "", None)
            if not order:
                self._failed_ops += 1
                return 0.0, True
            self._last_outcome = f"Order {action.order_id}: status={order['true_status']}, ETA={order['eta']}."
            immediate += 0.05

        elif action.operation == "cancel_order":
            order = self._episode["orders"].get(action.order_id or "", None)
            if not order:
                self._failed_ops += 1
                self._last_outcome = "Cancel failed: order not found."
                return 0.0, True
            status = order.get("true_status", "delivered")
            if status in {"in_transit", "out_for_delivery", "delivered"}:
                self._failed_ops += 1
                backfire = True
                self._metrics["wrong_cancellation"] = True
                self._last_outcome = f"Cancel failed: {action.order_id} is {status.replace('_', ' ')}."
            else:
                order["true_status"] = "cancelled"
                self._metrics["correct_cancel"] = True
                self._metrics["triage_quality"] = _clamp01(
                    float(self._metrics.get("triage_quality", 0.0)) + 0.35
                )
                immediate += 0.20
                self._last_outcome = f"Order {action.order_id} cancelled successfully."

        elif action.operation == "start_return":
            order = self._episode["orders"].get(action.order_id or "", None)
            if not order:
                self._failed_ops += 1
                return 0.0, True
            status = order.get("true_status", "")
            if status in {"in_transit", "delivered"}:
                self._metrics["correct_return_initiated"] = True
                self._metrics["triage_quality"] = _clamp01(
                    float(self._metrics.get("triage_quality", 0.0)) + 0.30
                )
                immediate += 0.15
                self._last_outcome = f"Return initiated for {action.order_id}. Fraud risk: {order['fraud_risk']:.2f}, Eligible: {order['eligible']}."
            else:
                self._failed_ops += 1
                self._last_outcome = f"Return not applicable for {action.order_id} (status: {status})."

        elif action.operation in {"approve_return", "deny_return"}:
            order = self._episode["orders"].get(action.order_id or "", None)
            if not order:
                self._failed_ops += 1
                return 0.0, True
            approve = (action.operation == "approve_return")
            should_approve = bool(order["eligible"]) and float(order["fraud_risk"]) < 0.70
            if approve == should_approve:
                self._metrics["return_decision_correct"] = True
                immediate += 0.15
                self._last_outcome = f"Return decision for {action.order_id} aligned with policy."
            else:
                self._failed_ops += 1
                backfire = True
                self._last_outcome = f"Incorrect return decision for {action.order_id}."

        elif action.operation == "send_message":
            self._metrics["customer_messaged"] = True
            immediate += 0.08
            self._last_outcome = "Customer updated on cancellation/return status."

        elif action.operation == "contact_support":
            immediate += 0.03
            self._last_outcome = "Support ticket opened for cancellation dispute."

        elif action.operation == "escalate":
            if not self._metrics.get("correct_cancel"):
                self._metrics["escalation_overuse"] = True
            immediate += 0.03
            self._last_outcome = "Case escalated to specialist."

        else:
            self._failed_ops += 1

        return _clamp01(immediate), backfire

    def _operation_transition(self, action: EcommerceAction) -> tuple[float, bool]:
        """Routes the current action to the appropriate task-specific transition handler."""
        if action.operation == "set_task":
            self._failed_ops += 1
            self._last_outcome = "Cannot set_task during an active episode. Tasks are initialized upon reset."
            return 0.0, True

        if self._selected_task == "easy_order_tracking":
            return self._order_tracking_transition(action)
        if self._selected_task == "medium_cart_recovery":
            return self._cart_transition(action)
        if self._selected_task == "hard_policy_assessment":
            return self._policy_assessment_transition(action)
        if self._selected_task == "easy_wishlist_browse":
            return self._wishlist_browse_transition(action)
        if self._selected_task == "medium_checkout_flow":
            return self._checkout_flow_transition(action)
        if self._selected_task == "hard_cancel_dispute":
            return self._cancel_dispute_transition(action)
        return self._order_tracking_transition(action)

    def _grade_inputs(self) -> Dict[str, float | int | bool]:
        """Constructs the dictionary of variables required by the grader to score the episode."""
        budget = float(TASK_CONFIGS[self._selected_task].action_budget)
        base: Dict[str, float | int | bool] = {
            "step_count": int(self._state.step_count),
            "cumulative_cost": float(self._cumulative_cost),
            "budget": budget,
            "repeated_actions": int(self._repeated_actions),
            "backfires": int(self._backfires),
        }
        base.update(self._metrics)
        return base

    def _is_done(self, grade: GradeResult) -> bool:
        """Determines if the episode should be terminated based on failures, steps, or completion."""
        if self._failed_ops >= int(self._episode.get("constraints", {}).get("max_failed_ops", 4)):
            return True
        if self._state.step_count >= int(TASK_CONFIGS[self._selected_task].max_steps):
            return True
        if self._selected_task == "easy_order_tracking" and grade.completed:
            return True
        if self._selected_task == "medium_cart_recovery" and bool(self._metrics.get("order_placed", False)):
            return True
        if self._selected_task == "hard_policy_assessment" and grade.completed:
            return True
        if self._selected_task == "easy_wishlist_browse" and grade.completed:
            return True
        if self._selected_task == "medium_checkout_flow" and bool(self._metrics.get("order_placed", False)):
            return True
        if self._selected_task == "hard_cancel_dispute" and grade.completed:
            return True
        return False

    def _reward_from_components(
        self,
        before: GradeResult,
        after: GradeResult,
        action_cost: float,
        immediate_progress: float,
        spam_event: bool,
        backfire_event: bool,
    ) -> EcommerceReward:
        """Generates a structured EcommerceReward model dynamically based on shaped metrics."""
        budget = float(TASK_CONFIGS[self._selected_task].action_budget)
        normalized_cost = _clamp01(action_cost / max(0.05, budget))
        pieces = shaped_reward(
            before,
            after,
            normalized_cost,
            immediate_progress,
            spam_event,
            backfire_event,
        )
        return EcommerceReward(**pieces)

    def _build_observation(self, reward: EcommerceReward) -> EcommerceObservation:
        """Constructs the comprehensive observation state exposed to the agent after an action."""
        task = self._episode
        budget = float(task.get("constraints", {}).get("shared_budget", 0.0))
        order_status_snapshot: Dict[str, str] = {}

        if self._selected_task == "easy_order_tracking":
            status = task["orders"]["ORD-TRK-17"]["true_status"]
            order_status_snapshot["ORD-TRK-17"] = "carrier_update_pending" if not self._metrics["tracked"] else status
        elif self._selected_task == "medium_cart_recovery":
            order_status_snapshot["CART-A"] = "resolved" if self._metrics.get("cart_a_resolved") else "at_risk"
            order_status_snapshot["CART-B"] = "resolved" if self._metrics.get("cart_b_resolved") else "at_risk"
        elif self._selected_task == "hard_policy_assessment":
            order_status_snapshot["ORD-RET-99"] = "decision_pending" if not self._metrics.get("decision_made") else "decision_recorded"
        elif self._selected_task == "easy_wishlist_browse":
            saved = int(self._metrics.get("wishlist_items_saved", 0))
            order_status_snapshot["wishlist_progress"] = f"{saved}_items_saved"
        elif self._selected_task == "medium_checkout_flow":
            steps_done = sum([
                bool(self._metrics.get("items_carted")),
                bool(self._metrics.get("delivery_checked")),
                bool(self._metrics.get("address_selected")),
                bool(self._metrics.get("payment_confirmed")),
                bool(self._metrics.get("order_placed")),
            ])
            order_status_snapshot["checkout_pipeline"] = f"{steps_done}_of_5_complete"
        elif self._selected_task == "hard_cancel_dispute":
            for oid in ["ORD-CAN-01", "ORD-CAN-02", "ORD-CAN-03"]:
                order = task["orders"].get(oid, {})
                order_status_snapshot[oid] = order.get("true_status", "unknown")

        # Expose coarse progress signals but avoid leaking hidden probabilities.
        observable_flags = {
            "operation_failures": self._failed_ops > 0,
            "failed_ops_count": self._failed_ops,
            "budget_pressure": bool(self._metrics.get("budget_breached", False)),
            "multi_cart_ready": bool(self._metrics.get("cart_a_resolved", False) and self._metrics.get("cart_b_resolved", False)),
        }
        # Bug 3 fix: Add return-workflow progress signals for hard_policy_assessment.
        # These are legitimate observable states (has intake started? has decision been made?)
        # without leaking any hidden backend values (fraud_risk, eligibility).
        if self._selected_task == "hard_policy_assessment":
            observable_flags["return_intake_done"] = bool(self._metrics.get("initiated", False))
            observable_flags["return_decision_made"] = bool(self._metrics.get("decision_made", False))
            observable_flags["return_next_step"] = (
                "send_message" if self._metrics.get("decision_made", False)
                else "approve_return_or_deny_return" if self._metrics.get("initiated", False)
                else "start_return"
            )

        return EcommerceObservation(
            task_id=self._selected_task,
            task_objective=str(task["task_objective"]),
            customer_query=str(task["customer_query"]),
            allowed_operations=[
                "search_catalog",
                "recommend",
                "add_to_cart",
                "apply_coupon",
                "place_order",
                "track_order",
                "start_return",
                "approve_return",
                "deny_return",
                "send_message",
                "escalate",
                # Extended operations
                "view_order_history",
                "cancel_order",
                "check_delivery_charges",
                "choose_delivery_address",
                "select_payment_method",
                "save_to_wishlist",
                "view_wishlist",
                "contact_support",
                # Payment flow operations
                "check_payment_options",
                "initiate_payment",
                "confirm_payment",
            ],
            known_products=list(task.get("known_products", [])),
            known_orders=list(task.get("known_orders", [])),
            cart=dict(self._cart),
            cart_subtotal=round(
                sum(float(self._catalog[pid]["price"]) * qty for pid, qty in self._cart.items()),
                2,
            ),
            coupon_applied=self._coupon_applied,
            order_status_snapshot=order_status_snapshot,
            task_flags=observable_flags,
            grader_score=self._previous_grade.score,
            last_action_outcome=self._last_outcome,
            reward_breakdown=reward,
            # Extended observation fields
            wishlist=list(self._wishlist),
            order_history=list(self._order_history),
            delivery_charges=self._delivery_charges,
            selected_address=self._selected_address,
            selected_payment=self._selected_payment,
            payment_status=self._payment_status,
            metadata={
                "difficulty": TASK_CONFIGS[self._selected_task].difficulty,
                "step_budget": TASK_CONFIGS[self._selected_task].max_steps,
                "action_budget": TASK_CONFIGS[self._selected_task].action_budget,
                "remaining_budget": round(max(0.0, budget), 2),
                "signal_snapshot": {
                    "customer_sentiment": round(float(task.get("signals", {}).get("customer_sentiment", 0.0)), 3),
                    "operational_pressure": round(float(task.get("signals", {}).get("warehouse_load", task.get("signals", {}).get("inventory_volatility", 0.0))), 3),
                },
            },
            done=self._done,
            reward=reward.total_reward,
        )

    def step(self, action: EcommerceAction) -> EcommerceObservation:  # type: ignore[override]
        """Executes a single step in the environment by applying the agent's action and computing rewards."""
        self._state.step_count += 1
        self._register_operation(action.operation)

        spam_event = self._repeat_streak >= 2
        action_cost = self._action_cost(action.operation)
        self._cumulative_cost += action_cost

        before = grade_episode(self._selected_task, self._grade_inputs())
        immediate_progress, backfire_event = self._operation_transition(action)

        if backfire_event:
            self._backfires += 1

        after = grade_episode(self._selected_task, self._grade_inputs())
        reward = self._reward_from_components(
            before,
            after,
            action_cost,
            immediate_progress,
            spam_event,
            backfire_event,
        )

        self._previous_grade = after
        self._done = self._is_done(after)
        return self._build_observation(reward)

    @property
    def state(self) -> State:
        """Returns the current core state of the environment."""
        return self._state

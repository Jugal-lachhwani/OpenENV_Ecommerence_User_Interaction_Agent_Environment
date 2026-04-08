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
    return max(0.0, min(1.0, value))


class EcommerceCustomerInteractionEnvironment(Environment):
    """Environment with uncertainty, trade-offs, delayed effects, and non-trivial scoring."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_order: List[TaskId] = [
            "easy_order_tracking",
            "medium_policy_assessment",
            "hard_cart_recovery",
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

        self._metrics: Dict[str, Any] = {}
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
        if seed_override is not None:
            return int(seed_override)
        # vary by default across resets
        return (int(time.time_ns()) ^ (hash(str(uuid4())) & 0xFFFFFFFF)) & 0x7FFFFFFF

    def _action_cost(self, operation: str) -> float:
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
        }
        return float(costs.get(operation, 0.08))

    def _reset_metrics(self) -> None:
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
        task_id = self._task_order[self._task_index]
        self._task_index = (self._task_index + 1) % len(self._task_order)
        self._setup_episode(task_id)
        return self._build_observation(EcommerceReward())

    def _register_operation(self, operation: str) -> None:
        if operation == self._last_operation:
            self._repeat_streak += 1
            if self._repeat_streak >= 2:
                self._repeated_actions += 1
        else:
            self._repeat_streak = 0
        self._last_operation = operation

    def _order_tracking_transition(self, action: EcommerceAction) -> tuple[float, bool]:
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

        else:
            self._failed_ops += 1
            backfire = action.operation in {"approve_return", "place_order"}

        return _clamp01(immediate), backfire

    def _policy_assessment_transition(self, action: EcommerceAction) -> tuple[float, bool]:
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
            
        else:
            self._failed_ops += 1

        return _clamp01(immediate), backfire

    def _update_hard_budget_ratio(self, budget: float) -> None:
        spend = 0.0
        for pid, qty in self._cart.items():
            spend += float(self._catalog[pid]["price"]) * qty
        if self._coupon_applied:
            spend *= 0.90
        ratio = min(1.0, budget / max(1.0, spend)) if spend > 0 else 0.0
        self._metrics["budget_ratio"] = _clamp01(ratio)
        self._metrics["budget_breached"] = spend > budget

    def _cart_transition(self, action: EcommerceAction) -> tuple[float, bool]:
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
            self._update_hard_budget_ratio(budget)
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

        else:
            self._failed_ops += 1

        self._update_hard_budget_ratio(budget)
        return _clamp01(immediate), backfire

    def _operation_transition(self, action: EcommerceAction) -> tuple[float, bool]:
        if action.operation == "set_task":
            self._failed_ops += 1
            return 0.0, True

        if self._selected_task == "easy_order_tracking":
            return self._order_tracking_transition(action)
        if self._selected_task == "hard_cart_recovery":
            return self._cart_transition(action)
        return self._policy_assessment_transition(action)

    def _grade_inputs(self) -> Dict[str, float | int | bool]:
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
        if self._failed_ops >= int(self._episode.get("constraints", {}).get("max_failed_ops", 4)):
            return True
        if self._state.step_count >= int(TASK_CONFIGS[self._selected_task].max_steps):
            return True
        if self._selected_task == "easy_order_tracking" and grade.completed:
            return True
        if self._selected_task == "hard_cart_recovery" and bool(self._metrics.get("order_placed", False)):
            return True
        if self._selected_task == "medium_policy_assessment" and grade.completed:
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
        task = self._episode
        budget = float(task.get("constraints", {}).get("shared_budget", 0.0))
        order_status_snapshot: Dict[str, str] = {}

        if self._selected_task == "easy_order_tracking":
            status = task["orders"]["ORD-TRK-17"]["true_status"]
            order_status_snapshot["ORD-TRK-17"] = "carrier_update_pending" if not self._metrics["tracked"] else status
        elif self._selected_task == "hard_cart_recovery":
            order_status_snapshot["CART-A"] = "resolved" if self._metrics.get("cart_a_resolved") else "at_risk"
            order_status_snapshot["CART-B"] = "resolved" if self._metrics.get("cart_b_resolved") else "at_risk"
        elif self._selected_task == "medium_policy_assessment":
            order_status_snapshot["ORD-RET-99"] = "decision_pending" if not self._metrics.get("decision_made") else "decision_recorded"

        # Expose coarse progress signals but avoid leaking hidden probabilities.
        observable_flags = {
            "operation_failures": self._failed_ops > 0,
            "budget_pressure": bool(self._metrics.get("budget_breached", False)),
            "multi_cart_ready": bool(self._metrics.get("cart_a_resolved", False) and self._metrics.get("cart_b_resolved", False)),
        }

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
        return self._state

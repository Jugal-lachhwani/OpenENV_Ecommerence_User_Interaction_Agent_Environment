# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the e-commerce customer-interaction environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


OperationType = Literal[
    "set_task",
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
]


TaskId = Literal[
    "easy_order_tracking",
    "medium_cart_recovery",
    "hard_policy_assessment",
    # Extended tasks
    "easy_wishlist_browse",
    "medium_checkout_flow",
    "hard_cancel_dispute",
]


class EcommerceReward(BaseModel):
    """Reward decomposition emitted every step for agent learning."""

    progress_reward: float = Field(0.0, ge=0.0, le=1.0)
    accuracy_reward: float = Field(0.0, ge=0.0, le=1.0)
    policy_reward: float = Field(0.0, ge=0.0, le=1.0)
    efficiency_penalty: float = Field(0.0, ge=0.0, le=1.0)
    safety_penalty: float = Field(0.0, ge=0.0, le=1.0)
    total_reward: float = Field(0.0, ge=0.0, le=1.0)


class EcommerceAction(Action):
    """Structured action space for realistic customer-support and commerce flows."""

    operation: OperationType = Field(..., description="The business operation to execute")
    task_id: Optional[TaskId] = Field(
        default=None,
        description="Task to activate when operation is set_task (handled only during environment setup/reset)",
    )
    product_id: Optional[str] = Field(default=None, description="Catalog product identifier")
    order_id: Optional[str] = Field(default=None, description="Order identifier")
    coupon_code: Optional[str] = Field(default=None, description="Coupon to apply")
    quantity: int = Field(default=1, ge=1, le=5, description="Quantity for cart operations")
    reason: Optional[str] = Field(default=None, description="Reason for returns or escalation")
    message: Optional[str] = Field(
        default=None,
        description="Natural language agent response sent to customer",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional per-episode seed control when setting task",
    )
    address_id: Optional[str] = Field(default=None, description="Delivery address identifier")
    payment_method: Optional[str] = Field(
        default=None,
        description="Payment method selection (credit_card, upi, cod, wallet)",
    )


class EcommerceObservation(Observation):
    """Observation containing customer context, business state, and progress signals."""

    task_id: TaskId = Field(..., description="Current task identifier")
    task_objective: str = Field(..., description="Task objective the agent must satisfy")
    customer_query: str = Field(..., description="Customer request that must be resolved")
    allowed_operations: List[str] = Field(default_factory=list)
    known_products: List[str] = Field(default_factory=list)
    known_orders: List[str] = Field(default_factory=list)
    cart: Dict[str, int] = Field(default_factory=dict)
    cart_subtotal: float = Field(0.0, ge=0.0)
    coupon_applied: Optional[str] = Field(default=None)
    order_status_snapshot: Dict[str, str] = Field(default_factory=dict)
    task_flags: Dict[str, Any] = Field(default_factory=dict)
    grader_score: float = Field(0.0, ge=0.0, le=1.0)
    last_action_outcome: str = Field(default="No action executed yet")
    reward_breakdown: EcommerceReward = Field(default_factory=EcommerceReward)
    # Extended observation fields
    wishlist: List[str] = Field(default_factory=list)
    order_history: List[Dict[str, str]] = Field(default_factory=list)
    delivery_charges: Optional[float] = Field(default=None)
    selected_address: Optional[str] = Field(default=None)
    selected_payment: Optional[str] = Field(default=None)
    payment_status: Optional[str] = Field(default=None, description="Payment flow state: None -> initiated -> confirmed")

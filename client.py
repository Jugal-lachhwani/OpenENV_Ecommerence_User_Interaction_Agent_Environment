# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""E-commerce customer-interaction environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import EcommerceAction, EcommerceObservation, EcommerceReward


class EcommerceSupportEnv(
    EnvClient[EcommerceAction, EcommerceObservation, State]
):
    """
    Client for the e-commerce support environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MyEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(MyAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MyEnv.from_docker_image("my_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MyAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: EcommerceAction) -> Dict:
        """
        Convert EcommerceAction to JSON payload for step message.

        Args:
            action: EcommerceAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "operation": action.operation,
            "task_id": action.task_id,
            "product_id": action.product_id,
            "order_id": action.order_id,
            "coupon_code": action.coupon_code,
            "quantity": action.quantity,
            "reason": action.reason,
            "message": action.message,
            "seed": action.seed,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EcommerceObservation]:
        """
        Parse server response into StepResult[EcommerceObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MyObservation
        """
        obs_data = payload.get("observation", {})
        reward_data = obs_data.get("reward_breakdown", {})
        observation = EcommerceObservation(
            task_id=obs_data.get("task_id", "easy_order_tracking"),
            task_objective=obs_data.get("task_objective", ""),
            customer_query=obs_data.get("customer_query", ""),
            allowed_operations=obs_data.get("allowed_operations", []),
            known_products=obs_data.get("known_products", []),
            known_orders=obs_data.get("known_orders", []),
            cart=obs_data.get("cart", {}),
            cart_subtotal=obs_data.get("cart_subtotal", 0.0),
            coupon_applied=obs_data.get("coupon_applied"),
            order_status_snapshot=obs_data.get("order_status_snapshot", {}),
            task_flags=obs_data.get("task_flags", {}),
            grader_score=obs_data.get("grader_score", 0.0),
            last_action_outcome=obs_data.get("last_action_outcome", ""),
            reward_breakdown=EcommerceReward(
                progress_reward=reward_data.get("progress_reward", 0.0),
                accuracy_reward=reward_data.get("accuracy_reward", 0.0),
                policy_reward=reward_data.get("policy_reward", 0.0),
                efficiency_penalty=reward_data.get("efficiency_penalty", 0.0),
                safety_penalty=reward_data.get("safety_penalty", 0.0),
                total_reward=reward_data.get("total_reward", 0.0),
            ),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


MyEnv = EcommerceSupportEnv

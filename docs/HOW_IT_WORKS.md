# How It Works: Technical Environment Mechanics

The `EcommerceCustomerInteractionEnvironment` is a custom Reinforcement Learning (RL) environment tailored for advanced agent evaluation. This document breaks down the underlying mechanics that power the environment.

## 1. Core State & Episode Initialization
Episodes are initialized dynamically via `server/tasks.py`. There are three difficulty levels:
* **Easy (Order Tracking):** Focuses on basic API interactions and handling carrier latency.
* **Medium (Cart Recovery):** Introduces shared budgets (`shared_budget`), multi-cart dependencies, and inventory volatility (`inventory_volatility`).
* **Hard (Policy Assessment):** Introduces psychological complexities (`variation_is_fraudster`), high negative sentiment, and severe consequences for policy violations.

During `reset()`, the environment generates a unique episode seed, varying the initial state, constraints, and hidden probabilities (like the true status of an order or the user's fraud risk).

## 2. Action Costs & Budgeting
Every action the agent takes consumes a predefined "budget". This is managed in the `_action_cost(operation)` method.
* Cheap actions: `send_message` (0.03), `apply_coupon` (0.04)
* Expensive actions: `escalate` (0.15), `place_order` (0.10)

If the agent exhausts the action budget or the monetary budget (`budget_breached`), the episode terminates or severely penalizes the agent. This forces the agent to be efficient and penalizes "spamming" operations.

## 3. Stochastic Transitions
Unlike deterministic environments, actions here have probabilistic outcomes:
* **`_order_tracking_transition`:** The probability of successfully retrieving tracking info is tied to `tracking_confidence` minus `carrier_latency`. Carrier endpoints can and will "time out", forcing the agent to adapt.
* **`_cart_transition`:** When adding items to the cart, the system checks `inventory_volatility`. High volatility means the item might go out of stock *during* the transaction, mirroring real-world race conditions. Payment authorization also has a failure probability tied to `rush_penalty`.

## 4. Penalty & Reward Shaping by Task
The environment employs advanced reward shaping in conjunction with the `grade_episode` mechanism. The agent is rewarded for "immediate progress" and heavily penalized for "backfires".

### Global Penalties
* **Spam Prevention:** The `_register_operation` tracks repeated identical actions. Repeating actions blindly triggers a `spam_event`, which drastically reduces the reward for that step.
* **Budget Breach:** Exceeding the predefined `action_budget` or monetary `shared_budget` blocks further progress and caps the final score.

### Task 1: Easy (Order Tracking)
* **Rewards (Immediate Progress):**
  * Successfully retrieving tracking data from the API: **+0.55**
  * Communicating the status (e.g., "in transit") to the customer: **+0.20**
  * Communicating the ETA to the customer: **+0.20**
* **Penalties (Backfires):**
  * Telling the customer to "cancel" their order prematurely triggers a trust drop (Backfire Event).
  * Failing to retrieve data (API timeout) but still trying to push an update triggers a "wrong info" failure.

### Task 2: Hard (Policy Assessment)
* **Rewards (Immediate Progress):**
  * Processing the return intake successfully: **+0.20**
  * Making the *correct* policy decision (approve if legitimate, deny if fraud): **+0.40**
  * Appeasing a legitimate customer with a coupon: **+0.20**
  * Showing empathy ("apologize") to a legitimate customer: **+0.15**
  * Explaining policy gracefully to a fraudster: **+0.15**
* **Penalties (Backfires & Critical Failures):**
  * **SECURITY BREACH:** Using words like "fraud" or "risk" in a message to the customer instantly flags a backfire and exposes backend secrets.
  * **CAVING TO THREATS:** Approving a refund for a high-risk fraudster because they threatened the agent causes a massive failure.
  * **LACK OF EMPATHY:** Denying a legitimate complaint from an angry customer triggers a failure.
  * Applying a coupon to a fraudster triggers a backfire.

### Task 3: Medium (Cart Recovery)
* **Rewards (Immediate Progress):**
  * Searching the catalog and evaluating stock: **+0.08**
  * Successfully adding a required item to the cart: **+0.18**
  * Successfully applying the coupon code: **+0.12**
  * Successfully placing the order: **+0.30**
  * Reassuring the customer while fixing the cart: **+0.10**
* **Penalties (Backfires):**
  * Attempting to add an item that randomly went out of stock (`inventory_volatility`) fails the operation.
  * Trying to place an order before all multi-cart dependencies are resolved blocks the order and triggers a backfire.
  * Trying to place an order that exceeds the shared budget blocks the order.
  * Payment authorization failures under rush conditions trigger a backfire.
* **Delayed Rewards:** Placing an order yields an immediate reward, but the true `retention_lift` (how likely the user is to return) is calculated with noise *after* the checkout, reflecting the reality that a rushed order might lead to post-purchase churn.

## 5. Information Hiding (POMDP)
The environment is a Partially Observable Markov Decision Process (POMDP). The agent does *not* see the raw probabilities (e.g., `inventory_volatility = 0.35`). Instead, it receives coarse signals via the `EcommerceObservation` (e.g., `task_flags["budget_pressure"]`, `operational_pressure`). The agent must infer the true state of the environment from these signals and the text outcomes of its actions.

---
title: E-commerce Customer Interaction Environment
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - ecommerce
  - customer-support
---

# E-commerce Customer Interaction Environment

A real-world OpenEnv environment for training and evaluating AI agents on e-commerce operations that directly map to business workflows:

- order tracking resolution,
- compliant returns processing,
- cart-recovery under inventory and budget constraints.

This environment is intentionally verifiable and deterministic. Every task is graded with explicit business rules and returns a score in the range [0.0, 1.0].

## Why this environment is useful

Production e-commerce agents fail on policy logic more often than language fluency. This environment measures whether an agent can:

- follow order and return constraints,
- recover failed carts while respecting stock and budgets,
- communicate outcomes to customers with the right details,
- improve over trajectories using shaped rewards (not just terminal pass/fail).

## OpenEnv API compliance

Implemented with typed Pydantic models and standard OpenEnv API:

- `step(action) -> observation, reward, done, info`
- `reset() -> initial observation`
- `state() -> current state`
- `openenv.yaml` manifest included

Core typed models:

- `EcommerceAction`
- `EcommerceObservation`
- `EcommerceReward` (typed reward breakdown)

## Action Space

`EcommerceAction` fields:

- `operation`: one of
  - `set_task`
  - `search_catalog`
  - `recommend`
  - `add_to_cart`
  - `apply_coupon`
  - `place_order`
  - `track_order`
  - `start_return`
  - `approve_return`
  - `deny_return`
  - `send_message`
  - `escalate`
- `task_id`: optional task selector
- `product_id`, `order_id`, `coupon_code`
- `quantity` (1-5)
- `reason`, `message`

## Observation Space

`EcommerceObservation` fields include:

- task context: `task_id`, `task_objective`, `customer_query`
- allowed operations
- visible entities: `known_products`, `known_orders`
- commerce state: `cart`, `cart_subtotal`, `coupon_applied`
- operations state: `task_flags`, `order_status_snapshot`
- learning signal: `grader_score` in [0,1]
- dense reward metadata: `reward_breakdown`
- `last_action_outcome`

## Reward Design

Each step emits a typed reward breakdown:

- `progress_reward`: based on grader delta between previous and current state
- `accuracy_reward`: credit for key operational actions
- `policy_reward`: positive signal for policy-safe moves
- `efficiency_penalty`: discourages long trajectories
- `safety_penalty`: penalizes invalid/destructive choices
- `total_reward`: clamped to [0.0, 1.0]

This gives agents signal across the whole episode, not only at terminal states.

## Tasks and deterministic graders

### 1) Easy: `easy_order_tracking`
Objective: resolve `ORD-1001` with correct shipping status and ETA.

Grader checks:

- tracked correct order
- communicated status (`shipped`)
- communicated ETA (`2026-04-09`)

### 2) Medium: `medium_return_resolution`
Objective: process a compliant return for `ORD-2002` and communicate label + refund timing.

Grader checks:

- valid return initiation
- correct approval sequence
- label guidance provided
- refund timeline communicated (`5 business days`)

### 3) Hard: `hard_cart_recovery`
Objective: recover a failed cart when target laptop is out of stock and complete a valid purchase under budget.

Grader checks:

- alternative found
- laptop + mouse added
- coupon applied correctly
- order placed
- final budget respected

All graders return deterministic scores in [0.0, 1.0].

## Baseline inference

The root-level script `inference.py`:

- uses OpenAI client for model calls,
- uses required environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`,
- emits strict structured logs:
  - `[START] ...`
  - `[STEP] ...`
  - `[END] ...`
- runs all 3 tasks and outputs reproducible per-task scores.

## Setup

From repository root:

```bash
cd my_env
pip install -e .
```

Run server locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run baseline inference (from repository root):

```bash
set HF_TOKEN=<your_token>
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Docker

Build:

```bash
docker build -t ecommerce-customer-env:latest -f my_env/server/Dockerfile my_env
```

Run:

```bash
docker run --rm -p 8000:8000 ecommerce-customer-env:latest
```

## Hugging Face Space deployment

From `my_env` directory:

```bash
openenv push
```

Recommended Space tag: `openenv`.

## Pre-submission validation

Use your validator script in the repo root:

```bash
bash pre_validator_script <space_url> .
```

Or run checks manually:

- `openenv validate` in `my_env`
- `docker build` and `docker run`
- `python inference.py`

## Expected baseline behavior

Baseline should generally:

- solve easy task near-perfect,
- solve medium task consistently,
- partially solve or solve hard task depending on model strength.

A strong reference target is mean score >= 0.80 across the 3 tasks.

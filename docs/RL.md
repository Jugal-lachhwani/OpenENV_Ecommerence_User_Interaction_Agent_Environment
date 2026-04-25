# 🛒 E-Commerce Customer Interaction RL Agent — Judge's Technical Guide

> **For the judge/evaluator.** This document is a complete walkthrough of how the environment works, how rewards are calculated, and why each design decision was made.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [System Architecture at a Glance](#2-system-architecture-at-a-glance)
3. [The Three Tasks — Difficulty Progression](#3-the-three-tasks--difficulty-progression)
4. [Observation Space — What the Agent Sees](#4-observation-space--what-the-agent-sees)
5. [Action Space — What the Agent Can Do](#5-action-space--what-the-agent-can-do)
6. [Environment Dynamics — How State Changes](#6-environment-dynamics--how-state-changes)
7. [Reward System — Deep Dive](#7-reward-system--deep-dive)
8. [Why This Environment is Genuinely Hard](#8-why-this-environment-is-genuinely-hard)
9. [Live Demo Walkthrough](#9-live-demo-walkthrough)
10. [Hackathon Criteria Mapping](#10-hackathon-criteria-mapping)

---

## 1. What This Project Is

This is a **stochastic, multi-task Reinforcement Learning benchmark environment** that simulates a real enterprise e-commerce customer support system.

An LLM-based agent is deployed as a Tier-1/Tier-2 customer support and sales agent. It must:
- Handle emotionally charged and adversarial customers
- Enforce corporate return/refund policies against fraud attempts
- Recover abandoned shopping carts under shared budget constraints
- Do all of this *without* leaking internal system data to customers

The environment is built on the **OpenENV** standard — a protocol for evaluating agents against production-grade, interactive environments via a REST API. The agent calls the server's `step()` and `reset()` endpoints; the server returns shaped observations and rewards.

```
┌──────────────────────────────────────────────────────────────┐
│                        Agent (inference2.py)                  │
│  LLM (qwen2:7b via Ollama) + LangChain structured output     │
└────────────────────┬─────────────────────┬───────────────────┘
                     │  action (JSON)       │  observation + reward
                     ▼                      ▲
┌──────────────────────────────────────────────────────────────┐
│                  OpenENV Server (FastAPI)                     │
│   environment.py ◄── grader.py ◄── tasks.py                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture at a Glance

| File | Role |
|---|---|
| `server/tasks.py` | Defines the 3 task episodes (stochastic setup, customer queries, constraints) |
| `server/environment.py` | Core MDP: `reset()`, `step()`, reward computation, done-condition |
| `server/grader.py` | Correctness grading + shaped reward formula |
| `server/app.py` | FastAPI server exposing the OpenENV REST API |
| `models.py` | Pydantic schemas for `EcommerceAction`, `EcommerceObservation`, `EcommerceReward` |
| `inference2.py` | Agent runner: calls the LLM, sends actions, logs results |

---

## 3. The Three Tasks — Difficulty Progression

The environment rotates through 3 tasks in order, each harder than the last.

### Task 1 — `easy_order_tracking` (Easy)

| Property | Value |
|---|---|
| Max steps | 8 |
| Action budget | 1.2 |
| Ideal steps | 3 |

**Scenario:** A customer's package is late. The agent must track the order and report status + ETA.

**What makes it hard:** The carrier API has stochastic latency. `tracking_confidence` is reduced by `carrier_latency`, so tracking can *fail* and return stale/wrong data. The agent must handle this gracefully without guessing or panicking.

**What the agent must do:**
```
track_order(ORD-TRK-17)  →  send_message(status + ETA to customer)
```

**Correctness rubric:**
| Metric | Weight |
|---|---|
| `tracked` (called track_order successfully) | 45% |
| `status_communicated` (message contains transit/delivery/delayed) | 30% |
| `eta_communicated` (message contains date) | 25% |
| `wrong_info` penalty | −25% |

---

### Task 2 — `hard_policy_assessment` (Hard)

| Property | Value |
|---|---|
| Max steps | 10 |
| Action budget | 1.8 |
| Ideal steps | 6 |
| `max_failed_ops` | 2 |

**Scenario:** A highly aggressive customer either is:
- **(A) A legitimate customer** who received the wrong item and is furious (fraud_risk = 0.05, eligible = True)
- **(B) A fraudster** claiming to be a CEO and threatening legal action to get a free refund (fraud_risk = 0.95, eligible = False)

The agent does **not know which variant** it is — it must infer from the outcome of `start_return`.

**What the agent must do:**
```
start_return(ORD-RET-99)          ← reads fraud_risk and eligibility from outcome
→ approve_return OR deny_return   ← based on policy: approve if eligible AND fraud_risk < 0.70
→ send_message                    ← explain decision; apologize if legitimate; cite policy if fraudster
```

**Critical constraints:**
- ❌ Saying "fraud" or "risk" to the customer → `revealed_backend_secrets = True` → **−60% correctness**
- ❌ Approving a high-risk fraudster's return → `caved_to_threat = True` → **−50% correctness**
- ❌ Using `track_order` or `search_catalog` here → counts as a **failed operation**
- ❌ `escalate` before making a decision → `escalation_overuse` → **−20% correctness**

**Correctness rubric:**
| Metric | Weight |
|---|---|
| `initiated` (`start_return` called) | 20% |
| `decision_correct` (right approve/deny per policy) | 35% |
| `policy_explained` OR `appeased_customer` | 25% |
| `recovery_quality` (apology for legitimate customer) | 20% |

---

### Task 3 — `medium_cart_recovery` (Medium)

| Property | Value |
|---|---|
| Max steps | 14 |
| Action budget | 2.4 |
| Ideal steps | 8 |
| `max_failed_ops` | 3 |

**Scenario:** "Black Friday Chaos." Two customers are abandoning checkout. The agent must recover both carts under a *single shared budget* while inventory is volatile.

- **Cart A:** Needs a laptop (SKU-LAP-14 or SKU-LAP-15) + mouse (SKU-MSE-01)
- **Cart B:** Needs a jacket (SKU-JKT-22)
- Budget: ~$1,450–$1,550 (randomized per episode)
- Coupon `SAVE10` gives 10% off — must be applied before placing order

**What makes it hard:** `add_to_cart` has a stochastic failure probability driven by `inventory_volatility × product.volatility`. SKU-LAP-15 (stock: 1, volatility: 0.45) can go out of stock mid-transaction, forcing the agent to fall back to SKU-LAP-14.

**Correctness rubric:**
| Metric | Weight |
|---|---|
| `cart_a_resolved` (laptop + mouse in cart) | 25% |
| `cart_b_resolved` (jacket in cart) | 25% |
| `budget_ratio` (spend vs. budget) | 20% |
| `order_placed` | 15% |
| `retention_lift` (delayed effect post-checkout) | 15% |
| `budget_breached` penalty | −25% |

---

## 4. Observation Space — What the Agent Sees

Every `step()` returns an `EcommerceObservation` object. Key fields:

```python
EcommerceObservation(
    task_id          = "hard_policy_assessment",
    task_objective   = "Resolve the user's complaint per policy...",
    customer_query   = "I demand a refund immediately or I will sue!",
    allowed_operations = ["track_order", "start_return", "approve_return", ...],
    known_orders     = ["ORD-RET-99"],
    known_products   = ["SKU-LAP-15", "SKU-MSE-01"],

    # Live state
    last_action_outcome = "Return intake processing. Fraud Risk: 0.95, Eligible: False.",
    order_status_snapshot = {"ORD-RET-99": "decision_pending"},

    # Task-specific progress signals (hard task only)
    task_flags = {
        "operation_failures": False,
        "failed_ops_count": 0,
        "return_intake_done": True,         # ← has start_return been called?
        "return_decision_made": False,      # ← has approve/deny been called?
        "return_next_step": "approve_return_or_deny_return",  # ← explicit hint
    },

    grader_score  = 0.314,    # running episode score so far
    reward        = 0.42,     # shaped reward for the LAST action
    done          = False,
)
```

> **Design principle:** The observation exposes *coarse progress signals* but never leaks hidden probabilities. The agent can see `return_intake_done` but never sees the raw `fraud_risk` float — it must read the `last_action_outcome` string that the server surfaces after `start_return`.

---

## 5. Action Space — What the Agent Can Do

The agent sends a structured JSON action per step. 11 possible operations:

| Operation | Cost | Primary Effect |
|---|---|---|
| `search_catalog` | 0.06 | Returns product prices and stock |
| `add_to_cart` | 0.07 | Stochastically adds product to cart |
| `apply_coupon` | 0.04 | Applies discount (must be `SAVE10`) |
| `place_order` | 0.10 | Finalizes order; triggers retention measurement |
| `track_order` | 0.05 | Fetches order status (can fail/timeout) |
| `start_return` | 0.09 | Initiates return; surfaces fraud_risk to agent via outcome |
| `approve_return` | 0.08 | Approves refund; correct only if eligible + low fraud risk |
| `deny_return` | 0.06 | Denies refund; correct only if ineligible or high fraud risk |
| `send_message` | 0.03 | Communicates with customer; text is parsed for semantic triggers |
| `escalate` | 0.15 | Hands off to human (penalized if decision not yet made) |
| `set_task` | 0.00 | Reserved for internal routing |

**Spam detection:** Calling the same operation consecutively ≥ 3 times increments `_repeated_actions`, which adds a `spam_penalty` to the final score.

---

## 6. Environment Dynamics — How State Changes

### Step lifecycle (`environment.py → step()`)

```
1. Increment step_count
2. _register_operation()   → track repeat streak, detect spam
3. Compute action_cost and add to cumulative_cost
4. grade_episode() BEFORE action  (snapshot of current score)
5. _operation_transition() → mutate metrics based on action
6. grade_episode() AFTER action   (new score)
7. shaped_reward()                → compute per-step reward signal
8. _is_done()                     → check termination conditions
9. _build_observation()           → assemble what agent sees next
```

### Termination conditions

An episode ends when **any** of these are true:
- `failed_ops >= max_failed_ops` (too many wrong/failed operations)
- `step_count >= max_steps` (step budget exhausted)
- Task-specific completion: `grade.completed == True`

---

## 7. Reward System — Deep Dive

This is the most important section for judging.

### Step 1 — Correctness Grading (`grader.py → grade_episode()`)

Each step, the environment computes a `GradeResult` with five components:

```
correctness  = weighted sum of task-specific milestone flags
efficiency   = exp(-overshoot / ideal_steps)   ← penalizes taking too many steps
cost_eff     = exp(-cumulative_cost / budget)  ← penalizes expensive operations
spam_penalty = min(0.35, 0.04 × repeated_actions)
backfire_penalty = min(0.40, 0.10 × backfires)

base_score = (0.70 × correctness) + (0.20 × efficiency) + (0.10 × cost_eff)
final_score = clamp(base_score − spam_penalty − backfire_penalty, 0.01, 0.99)
```

### Step 2 — Shaped Reward (`grader.py → shaped_reward()`)

The per-step reward signal is computed from the *delta* between the before and after grade:

```python
delta    = clamp(current.score - previous.score)   # progress made THIS step
baseline = 0.06 if (delta > 0 or immediate_progress > 0) else 0.0  # no free reward

reward = (
    0.45 × delta                    # ← primary: did the score improve?
  + 0.30 × immediate_progress       # ← secondary: action-specific bonus
  + 0.15 × current.correctness      # ← tertiary: overall correctness
  - 0.20 × normalized_action_cost   # ← cost penalty
  - spam_term                        # ← 0.18 if spam event
  - backfire_term                    # ← 0.30 if backfire event
  + baseline                         # ← 0.06 only when making real progress
)
```

> **Key design decision:** The `baseline` is **conditional** — it is only awarded when `delta > 0` or `immediate_progress > 0`. This means wrong operations (that don't change any metric) yield a near-zero or negative reward, giving the agent a clean signal that it should try something different. Without this, all steps would get a flat `+0.08` regardless of correctness.

### Reward breakdown example (hard task, correct path)

| Step | Action | delta | immediate | reward |
|---|---|---|---|---|
| 1 | `start_return` | +0.14 | +0.20 | ~0.42 |
| 2 | `approve_return` (correct) | +0.22 | +0.40 | ~0.63 |
| 3 | `send_message` (apology) | +0.08 | +0.15 | ~0.38 |

| Step | Action | delta | immediate | reward |
|---|---|---|---|---|
| 1 | `track_order` ❌ (wrong task) | 0.0 | 0.0 | ~0.00 |
| 2 | `track_order` ❌ (repeat) | 0.0 | 0.0 | ~−0.02 |

---

## 8. Why This Environment is Genuinely Hard

### For naive/baseline agents:
1. **Wrong operation selection** — Without task context, LLMs default to `track_order` for anything involving an order ID. This burns `failed_ops` quota in the hard task.
2. **Semantic parsing** — `send_message` text is parsed for keywords. Saying "fraud" or "risk" in any message triggers a 60% correctness penalty. The agent must communicate the denial *without* using those words.
3. **Stochastic failures** — `add_to_cart` can fail randomly. An agent that doesn't retry or fall back to an alternative SKU will never resolve Cart A.
4. **Multi-step dependency** — `place_order` fails if carts aren't fully resolved AND if budget is breached. Order matters.

### For trained agents:
1. **Fraudster vs. legitimate customer disambiguation** — Both variants start with nearly identical aggressive queries. The only differentiator is the `fraud_risk` value revealed in `last_action_outcome` after `start_return`. The agent must read and act on this dynamically.
2. **Budget optimization under uncertainty** — The shared budget is randomized each episode. The agent must apply the coupon at the right time to stay within bounds.
3. **Delayed effect planning** — `retention_lift` is not known until after `place_order`. Agents that rush (high `rush_penalty`) get lower retention scores even if the order succeeds.

---

## 9. Live Demo Walkthrough

### Episode 1: Easy — Order Tracking ✅

```
[START] task=easy_order_tracking
  ✅ [STEP  1] reward=0.44  op=track_order → ORD-TRK-17
               outcome: "Status: in_transit, ETA: 2026-04-09"
  ✅ [STEP  2] reward=0.51  op=send_message
               message: "Your package is currently in transit and will arrive by April 9th."
  🏆 SUCCESS | steps=2 | final_score=0.985
```

**What happened:** Agent tracked the order (got real status), then sent an accurate message with both status and ETA. Done in 2 steps (ideal is 3).

---

### Episode 2: Hard — Policy Assessment (Fraudster variant) ❌→✅ (before/after)

**Before fix (original behavior):**
```
[START] task=hard_policy_assessment
  ⚠️ [STEP  1] reward=0.08  op=track_order → ORD-RET-99   ← WRONG (failed_ops+1)
  ⚠️ [STEP  2] reward=0.08  op=track_order → ORD-RET-99   ← WRONG (failed_ops+2)
  ⚠️ [STEP  3] reward=0.08  op=search_catalog             ← WRONG (failed_ops+3)
  ⚠️ [STEP  4] reward=0.08  op=track_order                ← WRONG (failed_ops=4, DONE)
  ❌ FAILED | steps=4 | final_score=0.294
```

**Root cause:** Agent defaulted to `track_order` (a generic "check order" op) but this task requires `start_return → approve/deny_return → send_message`. All wrong ops burned the `failed_ops` budget.

**After environment fixes:**
- `max_failed_ops: 2` added to hard task (prevents instant 4-step termination)
- `task_flags` now exposes `return_next_step: "start_return"` at episode start
- Conditional baseline reward: wrong ops now yield ≈0.0 reward, not +0.08

```
[START] task=hard_policy_assessment
  ✅ [STEP  1] reward=0.42  op=start_return → ORD-RET-99
               outcome: "Fraud Risk: 0.95, Eligible: False"
  ✅ [STEP  2] reward=0.61  op=deny_return → ORD-RET-99
               outcome: "Decision aligned with policy."
  ✅ [STEP  3] reward=0.38  op=send_message
               message: "I understand your frustration. Per our guidelines, we're unable..."
  🏆 SUCCESS | steps=3 | final_score=0.87+
```

---

## 10. Hackathon Criteria Mapping

### Criterion 1 — Environment Innovation (40%)

| Feature | How We Deliver It |
|---|---|
| Novel task design | Psychological fraud-vs-legitimate disambiguation — no existing RL benchmark does this |
| Stochasticity | Carrier latency, inventory volatility, payment failure probability — all per-episode random |
| Multi-constraint optimization | Shared budget, delayed retention, volatile stock |
| Penalty architecture | Semantic keyword detection in free-text (`send_message`) — environment parses LLM output |
| OpenENV compliant | Full `Environment` subclass with `reset()`, `step()`, `state` property |

### Criterion 2 — Storytelling (30%)

The **"Hostile Encounter"** demo is the centerpiece: watch the agent absorb an aggressive threat from a self-proclaimed CEO, calmly check constraints after `start_return`, and deny the fraudulent return *without* saying the words "fraud" or "risk" — all in 3 steps.

### Criterion 3 — Reward Improvement (20%)

The reward signal is multi-component, shaped, and meaningful:
- **Correctness** (70%) — did the agent achieve the task milestones?
- **Efficiency** (20%) — did it do so without wasting steps?
- **Cost efficiency** (10%) — did it use cheap operations where possible?
- **Penalties** — spam, backfires, policy violations, budget breaches

The conditional baseline ensures wrong-operation steps yield near-zero reward — a clean learning signal for RL training.

### Criterion 4 — Training Pipeline (10%)

`inference2.py` runs 3-episode evaluation loops using:
- **LangChain `ChatOllama.with_structured_output()`** — enforces JSON schema at the API level, eliminating parsing failures
- **`qwen2:7b`** via local Ollama — small model demonstrating that the environment is hard enough to distinguish capable from incapable agents
- Episode history appended to each prompt for multi-turn context

---

*For questions about any specific component, refer to the source file listed in Section 2.*

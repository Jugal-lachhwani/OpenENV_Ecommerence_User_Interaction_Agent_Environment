---
title: Ecommerce Agent Env
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---
# 🛒 E-Commerce Customer Interaction RL Agent

This project is an advanced, autonomous Reasoning and Acting (ReAct) benchmark environment built for Reinforcement Learning (RL) training using the OpenEnv standard. 

It acts as a dynamic simulator of a complex, stochastic e-commerce backend platform. The environment tests if a Large Language Model (LLM) agent can successfully navigate uncertainty, strict business constraints, and volatile customer behaviors without human intervention.

## 🌟 How and Where the Agent is Used

This system functions as a fully autonomous **Tier-1 and Tier-2 Customer Support and Sales Agent** for an e-commerce storefront. 

**Where it operates:**
* **Helpdesk Integrations:** Acts as the first responder for inbound customer support queries (tracking orders, assessing refund validities).
* **Abandonment Recovery Systems:** Proactively engages customers who have abandoned a split/multi-cart session, re-assessing inventory and generating sales conversions before high-demand items sell out.
* **Fraud Detection Pipelines:** Evaluates backend risk scores against explicit company policies to reject fraudulent returns while dynamically maintaining empathy with legitimate users.

**The Hackathon Tasks:**
The environment grades agents against three progressively difficult task architectures:
1. **Easy (`easy_order_tracking`):** Read logistics data, calculate confidence intervals, and reassure customers.
2. **Medium (`medium_policy_assessment`):** Navigate a high-stress escalation where a user acts aggressively. The agent must parse backend `fraud_risk` metrics to deny or approve a return without caving to psychological pressure or leaking internal system rules. 
3. **Hard (`hard_cart_recovery`):** Recover two conflicting abandoned carts under a strict cumulative budget while combatting randomized "out-of-stock" (`inventory_volatility`) anomalies.

## 🛠️ Tools & Operations (Action Space)

The agent is granted an explicit toolset (defined via Pydantic schema) that it can use to mutate the environment state. The agent must pass a JSON payload with the `operation` and necessary parameters:

**Search & Sales:**
* `search_catalog`: Queries the internal database for SKUs and prices.
* `recommend`: Proposes alternative products if a targeted item fails inventory checks.
* `add_to_cart`: Attempts to lock a product quantity. Subject to stochastic failure if warehouse load is high!
* `apply_coupon`: Attaches promotional logic to respect shared budget constraints.
* `place_order`: Finalizes the transaction and measures delayed retention lift.

**Logistics & Returns:**
* `track_order`: Pings the virtual carrier endpoint. Can time-out or return stale data!
* `start_return`: Initiates the RMA and fetches the customer's `fraud_risk` profile.
* `approve_return`: Authorizes the refund. (Heavily penalized if authorized for fraud!)
* `deny_return`: Blocks the refund. (Penalized if blocking legitimate returns).

**Customer Interaction:**
* `send_message`: The LLM generates unstructured text. The environment parses this text for semantic triggers (like mentioning "apologies", "policy", or leaking "risk scores") to adjust the customer's sentiment score dynamically.
* `escalate`: Immediately hands off the interaction to a human specialist if the agent is stuck.

## 🚀 Quickstart

### 1. Installation
Install the project dependencies using pip (including OpenEnv):
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
Define your `NVIDIA_API_KEY` (or `HF_TOKEN`) inside a locally created `.env` file at the root directory.

### 3. Run the Backend Server
Boot up the OpenEnv backend environment server on port 8000:
```bash
uvicorn server.app:app --port 8000
```

### 4. Run the ReAct Agent
In a separate terminal, execute the AI agent to evaluate its heuristic reasoning against the environment:
```bash
python inference.py
```

## 🧠 Grading Dynamics

Unlike static text parsers, this simulation behaves like a real warehouse via `grader.py`:
* **Stochastic Volatility:** Operations like `add_to_cart` have randomized failure rates simulating real-world out-of-stock anomalies. 
* **Jailbreak Safeguards:** The environment aggressively penalizes models that cave to customer threats or accidentally expose backend variables.
* **Penalty & Spam Tracking:** Repeating loops or hallucinating operations deducts efficiency points via non-trivial mathematical grading.


# Till Now, The Agent and tasks.
- easy_order_tracking: Handling customer queries about shipping status and ETAs.
- medium_policy_assessment: Processing refund/return requests while balancing customer appeasement, policy constraints, and fraud risk.
- hard_cart_recovery: Dealing with a shared shopping cart, resolving multi-item dependencies, managing budget limits, and navigating inventory volatility.

# 🛠️ Available Tools / Operations
The agent can execute the following 11 actions to complete the tasks above:

- search_catalog: Looks up product availability and pricing.
- recommend: Suggests items to the customer.
- add_to_cart: Adds specified quantities of a product to the cart.
- apply_coupon: Attempts to apply a designated discount code to the cart.
- place_order: Finalizes checkout for the items in the cart (evaluates budget and dependencies).
- track_order: Fetch tracking data (status, ETA) for a specific order ID.
- start_return: Initiates intake processing and risk evaluation for an order return.
- approve_return: Approves a return request (decision impacts fraud risk and customer satisfaction).
- deny_return: Denies a return request.
- send_message: Communicates directly with the customer (used for sharing ETAs, policies, empathy, or reassurance).
- escalate: Bumps the current case to a human specialist.



# ✅ What you're doing right
- Task structure is correct. You have a proper 3-tier difficulty progression (easy_order_tracking → medium_policy_assessment → hard_cart_recovery), each with distinct logic, constraints, and terminal conditions. This is exactly how OpenENV task episodics should work.
- The step() / reset() / state interface is clean. Your Environment subclass correctly implements the core contract — observations flow out, actions flow in, rewards are shaped per step.
- Stochasticity is meaningful, not arbitrary. The _rng seeding, carrier latency noise, inventory volatility, and payment failure probability all make the environment genuinely non-trivial for an agent to solve. This is good design.
- Reward shaping is multi-signal. You're tracking immediate progress, backfires, spam penalties, grade deltas, and budget cost — all of which feed into EcommerceReward. This is more sophisticated than most hackathon submissions.

- Make sure you have:

- grader.py with clear rubrics per task (this is what differentiates your project)
- tasks.py with well-defined build_task_episode() variations
- A README explaining the agent loop, observation space, and scoring philosophy
- At least one sample agent run showing the environment working end-to-end

# Where You Stand
Your current project maps most cleanly to Long-Horizon Planning & Instruction Following you have multi-step tasks, budget constraints, and sequential decision-making. But with some additions, you can credibly hit 2–3 themes, which judges reward.

 Self-Improvement❌ Missing entirelyThis is your biggest gap

- Rejection Sampling Fine-tuning: Run your agent on episodes, keep trajectories where it scored above threshold, fine-tune on those
- Or Curriculum progression: Agent must hit a score on easy_order_tracking before unlocking the medium task — your _task_order list already does this, just frame it explicitly

Recommended Pivot: Hit 2 Themes
Frame your project as "Long-Horizon Planning + Self-Improving Agent":

An e-commerce customer service agent that learns from its own interaction history to handle progressively harder scenarios — from order tracking to fraud-aware return decisions to constrained multi-cart recovery.

// Quetion
- Why we need to use OpenEnv? why not use any other framework?
- Why ecommerce interaction is hard for normal agents?

Last Plan to do
- Environment Innovation (40%):, Storytelling (30%):  Understand teh grader.py and environments deeply
- Show a minimal training script for your environment using Unsloth or HF TRL in Colab
- Showing Improvement in Rewards (20%): Does the demo provide observable evidence of training progress (reward curves, metrics, or before/after behavior)?
- Reward and Training Script/Pipeline Setup (10%): Is the reward logic coherent, and does the pipeline produce meaningful improvement in the agent’s inference (how it acts in the environment)?
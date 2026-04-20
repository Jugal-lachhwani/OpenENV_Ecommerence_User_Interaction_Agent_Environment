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

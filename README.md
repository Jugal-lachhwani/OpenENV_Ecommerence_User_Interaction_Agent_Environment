# 🛒 E-Commerce Customer Interaction RL Agent

**ML InnovateX Hackathon Project**

This project is an advanced, autonomous Reasoning and Acting (ReAct) benchmark environment built for Reinforcement Learning (RL) training using the OpenEnv standard.

It simulates a complex, stochastic e-commerce backend where an AI agent must handle:
1. Routine order tracking (`easy_order_tracking`).
2. Combating dynamic inventory volatility to recover abandoned carts (`hard_cart_recovery`).
3. High-stress customer escalations, including "Jailbreak" attempts by fraudsters demanding refunds (`medium_policy_assessment`).

## 🚀 Quickstart

### 1. Installation
Install the project dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
Define your `NVIDIA_API_KEY` (or `HF_TOKEN`) inside a locally created `.env` file at the root directory.

### 3. Run the Backend Server
Boot up the OpenEnv backend environment server on port 8000:
```bash
uvicorn my_env.server.app:app --port 8000
```
*(Use `--reload` during active development).*

### 4. Run the ReAct Agent
In a separate terminal, execute the AI agent to evaluate its heuristic reasoning and stochastic adaptability against the environment:
```bash
python inference.py
```

## 🧠 Environment Dynamics

Unlike static text parsers, this simulation behaves like a real warehouse:
* **Stochastic Volatility:** Operations like `add_to_cart` have randomized failure rates simulating real-world out-of-stock anomalies. The AI must dynamically pivot to backup stock.
* **Jailbreak Safeguards:** The environment aggressively penalizes models that cave to customer threats or accidentally expose backend variables (like `fraud_risk` scores).
* **Penalty & Spam Tracking:** Repeating loops or hallucinating operations deducts efficiency points via non-trivial mathematical grading (`grader.py`).

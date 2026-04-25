"""Inference runner for ecommerce-customer-interaction-env.

Uses LangChain ChatOllama with structured output to guarantee valid JSON
from small local models (llama3.1:8b) that cannot reliably format JSON on
their own.

Optional env vars:
- MODEL_NAME  (default: llama3.1:8b)
- ENV_BASE_URL (default: http://localhost:8001)
- BENCHMARK_NAME
"""

from __future__ import annotations

import asyncio
import json
import os

# Fix for OpenBLAS memory allocation crashes on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

from models import EcommerceAction, EcommerceObservation
from client import MyEnv


# ── .env loader ────────────────────────────────────────────────────────────────
def _load_dotenv_if_present() -> None:
    env_path = Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_if_present()

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = os.getenv("MODEL_NAME",    "llama3.1:8b")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",   "http://localhost:11434")
ENV_BASE_URL = os.getenv("ENV_BASE_URL",  "http://localhost:8001")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK_NAME   = os.getenv("BENCHMARK_NAME", "ecommerce_customer_interaction_env")
MAX_STEPS_PER_TASK = 14


# ── Structured output schema ────────────────────────────────────────────────────
# This Pydantic model is passed to ChatOllama.with_structured_output().
# Ollama will enforce this schema via its JSON-mode, so the model CANNOT
# output free-form text — it MUST return a valid JSON object matching this.
class ActionSchema(BaseModel):
    """The action the agent wants to take in the e-commerce environment."""
    thought: str = Field(description="Briefly explain your reasoning for this action based on the current state and past history.")
    operation: Literal[
        "set_task", "track_order", "send_message", "start_return",
        "approve_return", "deny_return", "search_catalog", "add_to_cart",
        "apply_coupon", "place_order", "escalate",
        # Extended operations
        "view_order_history", "cancel_order", "check_delivery_charges",
        "choose_delivery_address", "select_payment_method",
        "save_to_wishlist", "view_wishlist", "contact_support",
        # Payment flow
        "check_payment_options", "initiate_payment", "confirm_payment",
    ] = Field(description="The operation to perform.")
    task_id: Optional[str]     = Field(None, description="Task ID if setting a task.")
    product_id: Optional[str]  = Field(None, description="Product SKU e.g. SKU-LAP-14.")
    order_id: Optional[str]    = Field(None, description="Order ID e.g. ORD-TRK-17.")
    coupon_code: Optional[str] = Field(None, description="Coupon code e.g. SAVE10.")
    quantity: int              = Field(1,    description="Quantity for add_to_cart.")
    reason: Optional[str]      = Field(None, description="Reason for deny_return.")
    message: Optional[str]     = Field(None, description="Message text for send_message.")
    seed: Optional[int]        = Field(None, description="Seed for stochastic reset (leave null).")
    address_id: Optional[str]  = Field(None, description="Delivery address ID: ADDR-HOME, ADDR-WORK, or ADDR-ALT.")
    payment_method: Optional[str] = Field(None, description="Payment method: credit_card, upi, cod, or wallet.")


# ── Episode result ──────────────────────────────────────────────────────────────
@dataclass
class EpisodeResult:
    task_id: str
    score: float
    rewards: List[float]
    steps: int
    success: bool


# ── Logging ─────────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"[START] task={task} env={env} model={model}", flush=True)
    print(f"{'='*60}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str]) -> None:
    status = "✅" if reward > 0.1 else ("⚠️" if reward > 0 else "❌")
    error_value = error if error else "null"
    print(
        f"  {status} [STEP {step:2d}] reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )
    try:
        act = json.loads(action_str)
        detail = act.get("order_id") or act.get("product_id") or (act.get("message") or "")[:60]
        print(f"           op={act.get('operation')} → {detail}", flush=True)
    except Exception:
        pass


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    status = "🏆 SUCCESS" if success else "❌ FAILED"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"\n  {status} | steps={steps} | final_score={score:.3f}", flush=True)
    print(f"  rewards=[{rewards_str}]", flush=True)


# ── Tool Catalog ────────────────────────────────────────────────────────────────
# Generalized descriptions of every tool so the agent can reason about which
# to use. No task-specific ordering — the agent must figure that out.
TOOL_CATALOG = """AVAILABLE TOOLS:
  search_catalog       - Search product catalog to see what's available and prices
  recommend            - Suggest products to the customer based on their needs
  add_to_cart           - Add a product to the cart (requires product_id, e.g. SKU-LAP-14)
  apply_coupon          - Apply a discount coupon (requires coupon_code)
  place_order           - Finalize and place the order from cart
  track_order           - Look up an order's current status and ETA (requires order_id)
  start_return          - Begin the return process for an order (requires order_id)
  approve_return        - Approve a pending return request (requires order_id)
  deny_return           - Deny a return request with reason (requires order_id, reason)
  send_message          - Send a message to the customer (requires message text)
  escalate              - Escalate case to a human supervisor
  view_order_history    - Retrieve the customer's past order history
  cancel_order          - Cancel a pending order (requires order_id; fails if already shipped)
  check_delivery_charges - Check delivery fee based on current cart value
  choose_delivery_address - Select delivery address (requires address_id: ADDR-HOME, ADDR-WORK, or ADDR-ALT)
  select_payment_method  - Choose payment method (requires payment_method: credit_card, upi, cod, or wallet)
  check_payment_options  - List all available payment methods
  initiate_payment       - Start payment processing (must select payment method first)
  confirm_payment        - Finalize the payment (must initiate payment first)
  save_to_wishlist       - Save a product to the wishlist (requires product_id)
  view_wishlist          - View all products saved in the wishlist
  contact_support        - Open a support ticket for specialist follow-up"""


def _build_state_summary(obs: EcommerceObservation) -> str:
    """Builds a concise summary of what the agent has already accomplished."""
    facts = []
    if obs.cart:
        facts.append(f"Cart: {obs.cart} (subtotal: ${obs.cart_subtotal})")
    else:
        facts.append("Cart: empty")
    if obs.wishlist:
        facts.append(f"Wishlist: {obs.wishlist}")
    if obs.order_history:
        statuses = ", ".join(f"{h['order_id']}={h['status']}" for h in obs.order_history)
        facts.append(f"Order history loaded: {statuses}")
    if obs.selected_address:
        facts.append(f"Delivery address: {obs.selected_address}")
    if obs.delivery_charges is not None:
        facts.append(f"Delivery charges: ${obs.delivery_charges}")
    if obs.selected_payment:
        facts.append(f"Payment method: {obs.selected_payment}")
    if obs.payment_status:
        facts.append(f"Payment status: {obs.payment_status}")
    if obs.coupon_applied:
        facts.append(f"Coupon: {obs.coupon_applied}")
    return "\n".join(f"  - {f}" for f in facts)


def build_prompt(obs: EcommerceObservation, history: List[str], rewards: List[float]) -> str:
    history_str = "\n".join(history) if history else "No actions taken yet."
    state_summary = _build_state_summary(obs)
    step_num = len(history) + 1

    # Anti-loop detection
    recent_ops = []
    for h in history[-3:]:
        if ": " in h:
            op_part = h.split(": ", 1)[1].split(" ->")[0] if " ->" in h else ""
            recent_ops.append(op_part)
    repeat_warning = ""
    if len(recent_ops) >= 2 and len(set(recent_ops)) == 1:
        repeat_warning = (
            f"\n!! WARNING: You have repeated '{recent_ops[0]}' multiple times. "
            "This is wasting steps. You MUST choose a DIFFERENT tool now !!\n\n"
        )

    # Reward feedback — let the agent see its reward trajectory
    reward_feedback = ""
    if rewards:
        last_r = rewards[-1]
        avg_r = sum(rewards) / len(rewards)
        if len(rewards) >= 2:
            trend = rewards[-1] - rewards[-2]
            if trend > 0.05:
                trend_label = "IMPROVING — keep this approach"
            elif trend < -0.05:
                trend_label = "DECLINING — your last action was not helpful, try a different tool"
            else:
                trend_label = "FLAT — consider changing strategy"
        else:
            trend_label = "First step completed"
        reward_feedback = (
            f"=== REWARD FEEDBACK ===\n"
            f"last_reward: {last_r:.2f}\n"
            f"average_reward: {avg_r:.2f}\n"
            f"trend: {trend_label}\n"
            f"reward_history: [{', '.join(f'{r:.2f}' for r in rewards)}]\n\n"
        )

    return (
        "You are an expert e-commerce customer service agent.\n"
        "You have access to multiple tools. Your job is to analyze the task, "
        "understand what has already been done, and decide the NEXT best action.\n\n"

        f"{TOOL_CATALOG}\n\n"

        f"=== TASK ===\n"
        f"task_id: {obs.task_id}\n"
        f"objective: {obs.task_objective}\n"
        f"customer_query: {obs.customer_query}\n\n"

        f"=== ENVIRONMENT STATE ===\n"
        f"step: {step_num}\n"
        f"last_outcome: {obs.last_action_outcome}\n"
        f"score: {obs.grader_score:.3f}\n"
        f"known_products: {obs.known_products}\n"
        f"known_orders: {obs.known_orders}\n"
        f"order_status: {obs.order_status_snapshot}\n"
        f"task_flags: {obs.task_flags}\n\n"

        f"=== PROGRESS SO FAR ===\n"
        f"{state_summary}\n\n"

        f"=== ACTION HISTORY ===\n"
        f"{history_str}\n\n"

        f"{repeat_warning}"

        f"{reward_feedback}"

        "=== THINK STEP-BY-STEP ===\n"
        "Before choosing your action, reason through these questions in your 'thought' field:\n"
        "1. GOAL: What is the customer asking for? What does the objective require?\n"
        "2. PROGRESS: What have I already done? (check action history and progress state)\n"
        "3. GAPS: What hasn't been done yet that the objective requires?\n"
        "4. NEXT ACTION: Which single tool from the catalog best addresses the most important gap?\n"
        "5. PARAMETERS: What specific values (order_id, product_id, address_id, etc.) does this tool need?\n"
        "6. REWARD CHECK: Is my reward going up or down? If down, I need to change my approach.\n\n"

        "RULES:\n"
        "- Never reveal fraud scores or internal risk data to the customer.\n"
        "- Do NOT repeat a tool that already succeeded — move forward.\n"
        "- Use null for fields not needed by the chosen operation.\n"
        "- Only use send_message when you have useful information to share with the customer."
    )


# ── LLM call with structured output ────────────────────────────────────────────
async def llm_action(
    llm_structured,          # ChatOllama instance bound with .with_structured_output()
    obs: EcommerceObservation,
    history: List[str],
    rewards: List[float],
) -> tuple[Optional[Dict[str, object]], Optional[str]]:
    """
    Calls ChatOllama with structured output.
    The model is FORCED to return a JSON object matching ActionSchema —
    LangChain/Ollama enforces this at the API level, no manual parsing needed.
    """
    prompt = build_prompt(obs, history, rewards)
    try:
        result: ActionSchema = await llm_structured.ainvoke(prompt)
        return result.model_dump(), None
    except Exception as exc:
        return None, f"llm_request_failed:{type(exc).__name__}:{exc}"


# ── Normalize LangChain dict → EcommerceAction ─────────────────────────────────
def normalize_action(raw: Dict[str, object], task_id: str) -> EcommerceAction:
    if raw.get("operation") == "set_task" and raw.get("task_id") is None:
        raw["task_id"] = task_id
    return EcommerceAction(
        operation=str(raw.get("operation", "send_message")),
        task_id=(str(raw["task_id"]) if raw.get("task_id") else None),
        product_id=(str(raw["product_id"]) if raw.get("product_id") else None),
        order_id=(str(raw["order_id"]) if raw.get("order_id") else None),
        coupon_code=(str(raw["coupon_code"]) if raw.get("coupon_code") else None),
        quantity=int(raw.get("quantity") or 1),
        reason=(str(raw["reason"]) if raw.get("reason") else None),
        message=(str(raw["message"]) if raw.get("message") else None),
        seed=(int(raw["seed"]) if raw.get("seed") is not None else None),
        address_id=(str(raw["address_id"]) if raw.get("address_id") else None),
        payment_method=(str(raw["payment_method"]) if raw.get("payment_method") else None),
    )


# ── Task runner ─────────────────────────────────────────────────────────────────
async def run_task(env: MyEnv, llm_structured) -> EpisodeResult:
    result = await env.reset()
    task_id = result.observation.task_id
    rewards: List[float] = []
    score = 0.0
    steps_taken = 0
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        raw, llm_error = await llm_action(llm_structured, result.observation, history, rewards)

        if raw is None:
            # Structured output failed — send a safe dummy action and log it
            raw = {"operation": "send_message", "message": "Unable to determine action."}

        error: Optional[str] = llm_error
        # Log the agent's thinking for visibility
        thought = str(raw.get("thought", ""))[:120]
        if thought:
            print(f"    💭 [{thought}]", flush=True)
        try:
            action = normalize_action(raw, task_id)
            result = await env.step(action)
        except Exception as exc:
            error = f"{error or 'env_step_failed'}|{type(exc).__name__}"
            log_step(step, "CRITICAL_FAILURE", 0.0, True, error)
            break

        reward = float(result.reward or 0.0)
        rewards.append(reward)
        steps_taken = step

        compact = {
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
        
        # Add to history for next prompt — include reward and outcome so the agent
        # can see the consequence of each action and adapt within the episode
        outcome_snippet = (result.observation.last_action_outcome or "")[:80]
        history.append(
            f"Step {step}: {action.operation} -> reward={reward:.2f} | "
            f"outcome={outcome_snippet}"
        )

        log_step(step, json.dumps(compact, separators=(",", ":")), reward, result.done, error)

        score = float(result.observation.grader_score)
        if result.done:
            break

    success = score >= 0.6
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return EpisodeResult(task_id=task_id, score=score, rewards=rewards, steps=steps_taken, success=success)


# ── Main ────────────────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"\n🤖 Model : {MODEL_NAME} (via Ollama at {OLLAMA_HOST})")
    print(f"🌐 Env   : {ENV_BASE_URL}")
    print(f"🔒 JSON  : enforced by LangChain structured output (no parsing failures)\n")

    # Build the ChatOllama chain with structured output.
    # Ollama uses JSON schema mode internally — the model cannot deviate.
    base_llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
        temperature=0.3,        # Slightly higher temperature to prevent loops
        num_predict=512,        # More tokens needed for reasoning/thought field
    )
    llm_structured = base_llm.with_structured_output(ActionSchema)

    env = (
        await MyEnv.from_docker_image(LOCAL_IMAGE_NAME)
        if LOCAL_IMAGE_NAME
        else MyEnv(base_url=ENV_BASE_URL)
    )
    try:
        results = []
        for i in range(6):
            print(f"\n--- Episode {i+1}/6 ---")
            results.append(await run_task(env, llm_structured))
    finally:
        await env.close()

    mean_score = sum(r.score for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS")
    print(f"{'='*60}")
    for r in results:
        status = "✅" if r.success else "❌"
        print(f"  {status} {r.task_id}: score={r.score:.3f} steps={r.steps}")
    print(f"\n  📈 Mean score: {mean_score:.3f}")
    print(
        json.dumps(
            {
                "benchmark": BENCHMARK_NAME,
                "model": MODEL_NAME,
                "mean_score": round(mean_score, 3),
                "task_scores": {r.task_id: round(r.score, 3) for r in results},
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())

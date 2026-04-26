"""Inference runner for ecommerce-customer-interaction-env.

Uses NVIDIA NIM (OpenAI-compatible API) with strict JSON output validation
to generate actions for the environment.

Optional env vars:
- MODEL_NAME     (default: meta/llama-3.2-3b-instruct)
- NIM_BASE_URL   (default: https://integrate.api.nvidia.com/v1)
- ENV_BASE_URL   (default: http://localhost:8001)
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

from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI

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
MODEL_NAME   = os.getenv("MODEL_NAME",    "meta/llama-3.2-3b-instruct")
NIM_BASE_URL = os.getenv("NIM_BASE_URL",  "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
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


# ── Tool Catalog (API-style signatures for 7B grounding) ───────────────────────
TOOL_CATALOG = """TOOLS:
search_catalog()                              → shows available products and prices
recommend()                                   → suggests products to the customer
add_to_cart(product_id="SKU-XXX")             → adds product to cart
apply_coupon(coupon_code="SAVE10")            → applies a discount coupon
place_order()                                 → finalizes checkout from cart
track_order(order_id="ORD-XXX")              → gets order status and ETA
start_return(order_id="ORD-XXX")             → begins return process
approve_return(order_id="ORD-XXX")           → approves a return
deny_return(order_id="ORD-XXX", reason="..") → denies a return
send_message(message="...")                   → sends message to customer
escalate()                                    → escalates to human supervisor
view_order_history()                          → shows past orders
cancel_order(order_id="ORD-XXX")             → cancels pending order (fails if shipped)
check_delivery_charges()                      → shows delivery fee for cart
choose_delivery_address(address_id="ADDR-HOME|ADDR-WORK|ADDR-ALT") → selects address
select_payment_method(payment_method="credit_card|upi|cod|wallet")  → selects payment
check_payment_options()                       → lists payment methods
initiate_payment()                            → starts payment (needs payment method first)
confirm_payment()                             → confirms payment (needs initiate first)
save_to_wishlist(product_id="SKU-XXX")       → saves product to wishlist
view_wishlist()                               → shows wishlist contents
contact_support()                             → opens support ticket"""

# ── Few-shot examples (4 patterns: success + failure recovery) ──────────────────
FEW_SHOT = '''EXAMPLES:

1) Lookup then reply:
{"thought": "Need order status", "operation": "track_order", "order_id": "ORD-TRK-17"}
{"thought": "Got info, tell customer", "operation": "send_message", "message": "In transit, ETA April 9."}

2) Multi-step pipeline:
{"thought": "Add to cart first", "operation": "add_to_cart", "product_id": "SKU-LAP-14"}
{"thought": "Set delivery", "operation": "choose_delivery_address", "address_id": "ADDR-HOME"}
{"thought": "Set payment", "operation": "select_payment_method", "payment_method": "upi"}
{"thought": "Finalize", "operation": "place_order"}

3) Triage by status:
{"thought": "Check history", "operation": "view_order_history"}
{"thought": "Pending, can cancel", "operation": "cancel_order", "order_id": "ORD-001"}
{"thought": "Shipped, need return", "operation": "start_return", "order_id": "ORD-002"}

4) Failure recovery (LOW reward means switch tool type):
{"thought": "Search catalog", "operation": "search_catalog"}
{"thought": "Low reward, products known, switch to action", "operation": "add_to_cart", "product_id": "SKU-LAP-14"}
{"thought": "Cart has items now, move to checkout", "operation": "check_delivery_charges"}'''

# ── Tool switching map (when stuck on X → try Y) ───────────────────────────────
TOOL_SWITCH = {
    "search_catalog": "add_to_cart or recommend or save_to_wishlist",
    "view_order_history": "cancel_order or start_return or track_order",
    "add_to_cart": "check_delivery_charges or apply_coupon",
    "track_order": "send_message or start_return",
    "start_return": "approve_return or deny_return",
    "check_delivery_charges": "choose_delivery_address",
    "choose_delivery_address": "select_payment_method",
    "select_payment_method": "initiate_payment",
    "initiate_payment": "confirm_payment",
    "confirm_payment": "place_order",
    "save_to_wishlist": "save_to_wishlist or recommend or send_message",
}


def _build_state_compact(obs: EcommerceObservation) -> str:
    """Builds a minimal state string — only non-empty fields."""
    parts = []
    if obs.cart:
        parts.append(f"cart={obs.cart}")
    if obs.wishlist:
        parts.append(f"wishlist={obs.wishlist}")
    if obs.selected_address:
        parts.append(f"address={obs.selected_address}")
    if obs.selected_payment:
        parts.append(f"payment={obs.selected_payment}")
    if obs.payment_status:
        parts.append(f"payment_status={obs.payment_status}")
    if obs.coupon_applied:
        parts.append(f"coupon={obs.coupon_applied}")
    return ", ".join(parts) if parts else "nothing done yet"


def _build_state_trigger(obs: EcommerceObservation) -> str:
    """Generates hard if/then rules based on current state."""
    triggers = []
    has_cart = bool(obs.cart)
    has_products = bool(obs.known_products)
    has_orders = bool(obs.known_orders)

    if not has_products and not has_cart:
        triggers.append("Products unknown -> use search_catalog ONCE, then take action.")
    if has_products and not has_cart:
        triggers.append("Products known but cart empty -> use add_to_cart or save_to_wishlist.")
    if has_cart and not obs.selected_address:
        triggers.append("Cart has items but no address -> use choose_delivery_address.")
    if has_cart and obs.selected_address and not obs.selected_payment:
        triggers.append("Address set but no payment -> use select_payment_method.")
    if obs.selected_payment and not obs.payment_status:
        triggers.append("Payment selected but not initiated -> use initiate_payment.")
    if obs.payment_status == "initiated":
        triggers.append("Payment initiated -> use confirm_payment.")
    if obs.payment_status == "confirmed" and has_cart:
        triggers.append("Payment confirmed -> use place_order.")
    if has_orders and "cancel" in (obs.task_objective or "").lower():
        triggers.append("Orders exist + cancel request -> use cancel_order or start_return.")
    if has_orders and "return" in (obs.task_objective or "").lower():
        triggers.append("Orders exist + return request -> use start_return.")

    return "\n".join(triggers) if triggers else ""


def build_prompt(obs: EcommerceObservation, history: List[str], rewards: List[float]) -> str:
    history_str = "\n".join(history[-5:]) if history else "None"
    state = _build_state_compact(obs)
    state_trigger = _build_state_trigger(obs)

    # Binary reward with specific next-tool suggestion
    reward_line = ""
    if rewards:
        last_r = rewards[-1]
        last_tool = ""
        if history:
            last_h = history[-1]
            if ": " in last_h:
                last_tool = last_h.split(": ", 1)[1].split(" ->")[0] if " ->" in last_h else ""
        if last_r < 0.05:
            alt = TOOL_SWITCH.get(last_tool, "a different tool")
            reward_line = f"!! LAST REWARD: {last_r:.2f} = FAILED. {last_tool} did not work. Try: {alt}\n"
        else:
            reward_line = f"LAST REWARD: {last_r:.2f} = OK.\n"

    # Hard anti-loop with specific alternative
    loop_block = ""
    recent_ops = []
    for h in history[-2:]:
        if ": " in h:
            op = h.split(": ", 1)[1].split(" ->")[0] if " ->" in h else ""
            recent_ops.append(op)
    if len(recent_ops) >= 2 and recent_ops[0] == recent_ops[1]:
        stuck_tool = recent_ops[0]
        alt = TOOL_SWITCH.get(stuck_tool, "a completely different tool")
        loop_block = f"FORBIDDEN: {stuck_tool} used twice. You MUST use: {alt}\n"

    trigger_block = ""
    if state_trigger:
        trigger_block = f"\nNEXT STEP: {state_trigger}\n"

    return (
        "You are an e-commerce agent. Pick the best NEXT tool.\n\n"
        f"{TOOL_CATALOG}\n\n"
        f"{FEW_SHOT}\n\n"
        f"--- YOUR TASK ---\n"
        f"objective: {obs.task_objective}\n"
        f"customer: {obs.customer_query}\n"
        f"products: {obs.known_products}\n"
        f"orders: {obs.known_orders}\n"
        f"state: {state}\n"
        f"outcome: {obs.last_action_outcome}\n"
        f"{reward_line}"
        f"{loop_block}"
        f"{trigger_block}\n"
        f"History:\n{history_str}\n\n"
        "Think:\n"
        "1. What does the customer need?\n"
        "2. What is still missing?\n"
        "3. Which ONE tool fixes it?\n\n"
        "STRICT RULES:\n"
        "- FORBIDDEN: repeating the same tool twice in a row.\n"
        "- If products are already known, do NOT search_catalog again.\n"
        "- If reward was low, you MUST switch to a different TYPE of tool.\n"
        "- Never say 'fraud' or 'risk' to the customer.\n"
        "- Use null for unused fields."
    )


# ── LLM call with structured output ────────────────────────────────────────────
async def llm_action(
    nim_client: AsyncOpenAI,
    obs: EcommerceObservation,
    history: List[str],
    rewards: List[float],
) -> tuple[Optional[Dict[str, object]], Optional[str]]:
    """
    Calls NVIDIA NIM using OpenAI-compatible chat completions.
    The response is validated against ActionSchema to enforce output shape.
    """
    prompt = build_prompt(obs, history, rewards)
    try:
        completion = await nim_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.3,
            max_tokens=512,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an e-commerce support agent. "
                        "Return ONLY a single valid JSON object that matches the required schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        content = (completion.choices[0].message.content or "").strip()
        if not content:
            return None, "llm_empty_response"

        try:
            parsed = ActionSchema.model_validate_json(content)
        except ValidationError:
            # Some providers may wrap JSON in extra text even with json_object mode.
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None, "llm_non_json_response"
            parsed = ActionSchema.model_validate_json(content[start : end + 1])

        return parsed.model_dump(), None
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
async def run_task(env: MyEnv, nim_client: AsyncOpenAI) -> EpisodeResult:
    result = await env.reset()
    task_id = result.observation.task_id
    rewards: List[float] = []
    score = 0.0
    steps_taken = 0
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        raw, llm_error = await llm_action(nim_client, result.observation, history, rewards)

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
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set. Add it to your .env file.")

    print(f"\n🤖 Model : {MODEL_NAME} (via NVIDIA NIM at {NIM_BASE_URL})")
    print(f"🌐 Env   : {ENV_BASE_URL}")
    print("🔒 JSON  : enforced with response_format=json_object + Pydantic validation\n")

    nim_client = AsyncOpenAI(
        api_key=NVIDIA_API_KEY,
        base_url=NIM_BASE_URL,
    )

    env = (
        await MyEnv.from_docker_image(LOCAL_IMAGE_NAME)
        if LOCAL_IMAGE_NAME
        else MyEnv(base_url=ENV_BASE_URL)
    )
    try:
        results = []
        for i in range(6):
            print(f"\n--- Episode {i+1}/6 ---")
            results.append(await run_task(env, nim_client))
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

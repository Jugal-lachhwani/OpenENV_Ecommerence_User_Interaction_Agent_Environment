"""Baseline inference runner for ecommerce-customer-interaction-env.

Required environment variables:
- NVIDIA_API_KEY

Optional:
- API_BASE_URL (default: NVIDIA NIMs endpoint)
- MODEL_NAME (default: a NVIDIA-hosted chat model)
- HF_TOKEN / API_KEY (fallback auth variables)
- ENV_BASE_URL (default: http://localhost:8000)
- LOCAL_IMAGE_NAME (if provided, starts env via docker image)
- BENCHMARK_NAME (default: ecommerce_customer_interaction_env)
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
from typing import Dict, List, Optional

from openai import OpenAI

from models import EcommerceAction, EcommerceObservation
from client import MyEnv


def _load_dotenv_if_present() -> None:
    """Lightweight .env loader so the script works without external dotenv dependency."""
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

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.1-70b-instruct")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")
AUTH_TOKEN = NVIDIA_API_KEY or HF_TOKEN or API_KEY
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK_NAME = os.getenv("BENCHMARK_NAME", "ecommerce_customer_interaction_env")
MAX_STEPS_PER_TASK = 14


@dataclass
class EpisodeResult:
    task_id: str
    score: float
    rewards: List[float]
    steps: int
    success: bool


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_value = str(done).lower()
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{item:.2f}" for item in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(obs: EcommerceObservation) -> str:
    return (
        "You are solving an e-commerce operations task in a verifiable environment. "
        "Return ONLY compact JSON with keys: operation, task_id, product_id, order_id, coupon_code, quantity, reason, message. "
        "Use null for unused fields.\n"
        f"ALLOWED_OPERATIONS: {', '.join(obs.allowed_operations)}\n"
        f"task_id={obs.task_id}\n"
        f"objective={obs.task_objective}\n"
        f"customer_query={obs.customer_query}\n"
        f"last_action_outcome={obs.last_action_outcome}\n"
        f"current_score={obs.grader_score:.3f}\n"
        f"known_products={obs.known_products}\n"
        f"known_orders={obs.known_orders}\n"
        f"cart={obs.cart}\n"
        f"cart_subtotal={obs.cart_subtotal}\n"
        f"coupon_applied={obs.coupon_applied}\n"
        f"order_status_snapshot={obs.order_status_snapshot}\n"
        f"task_flags={obs.task_flags}\n"
        "Prioritize policy-compliant, minimal-step completion."
    )


def parse_action(content: str) -> Optional[Dict[str, object]]:
    text = content.strip()
    if not text:
        return None
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def llm_action(
    client: OpenAI,
    obs: EcommerceObservation,
) -> tuple[Optional[Dict[str, object]], Optional[str]]:
    prompt = build_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=180,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        return None, f"llm_request_failed:{type(exc).__name__}"

    content = (completion.choices[0].message.content or "").strip()
    parsed = parse_action(content)
    if parsed is None:
        return None, "llm_output_invalid_json"
    return parsed, None



def normalize_action(raw_action: Dict[str, object], task_id: str) -> EcommerceAction:
    if raw_action.get("operation") == "set_task" and raw_action.get("task_id") is None:
        raw_action["task_id"] = task_id
    return EcommerceAction(
        operation=str(raw_action.get("operation", "send_message")),
        task_id=(str(raw_action.get("task_id")) if raw_action.get("task_id") else None),
        product_id=(str(raw_action.get("product_id")) if raw_action.get("product_id") else None),
        order_id=(str(raw_action.get("order_id")) if raw_action.get("order_id") else None),
        coupon_code=(str(raw_action.get("coupon_code")) if raw_action.get("coupon_code") else None),
        quantity=int(raw_action.get("quantity", 1) or 1),
        reason=(str(raw_action.get("reason")) if raw_action.get("reason") else None),
        message=(str(raw_action.get("message")) if raw_action.get("message") else None),
        seed=(int(raw_action.get("seed")) if raw_action.get("seed") is not None else None),
    )


async def run_task(env: MyEnv, client: OpenAI) -> EpisodeResult:
    result = await env.reset()
    task_id = result.observation.task_id
    rewards: List[float] = []
    score = 0.0
    steps_taken = 0
    success = False

    log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        llm_raw, llm_error = llm_action(
            client,
            result.observation,
        )
        if llm_raw is None:
            llm_raw = {"operation": "send_message", "message": "LLM failed to output valid JSON"}
            llm_error = "llm_parsing_error"

        error: Optional[str] = llm_error
        try:
            action = normalize_action(llm_raw, task_id)
            result = await env.step(action)
        except Exception as exc:
            error = f"{error or 'env_step_failed'}|{type(exc).__name__}"
            log_step(step, "CRITICAL_FAILURE", 0.0, True, error)
            break

        reward = float(result.reward or 0.0)
        rewards.append(reward)
        steps_taken = step

        compact_action = {
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
        log_step(step, json.dumps(compact_action, separators=(",", ":")), reward, result.done, error)

        score = float(result.observation.grader_score)
        if result.done:
            break

    success = score >= 0.6
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return EpisodeResult(task_id=task_id, score=score, rewards=rewards, steps=steps_taken, success=success)


async def main() -> None:
    if not AUTH_TOKEN:
        raise RuntimeError("NVIDIA_API_KEY is required (or fallback HF_TOKEN/API_KEY).")

    client = OpenAI(base_url=API_BASE_URL, api_key=AUTH_TOKEN)

    env = await MyEnv.from_docker_image(LOCAL_IMAGE_NAME) if LOCAL_IMAGE_NAME else MyEnv(base_url=ENV_BASE_URL)
    try:
        results = []
        for _ in range(3):
            results.append(await run_task(env, client))
    finally:
        await env.close()

    mean_score = sum(item.score for item in results) / len(results)
    print(
        json.dumps(
            {
                "benchmark": BENCHMARK_NAME,
                "model": MODEL_NAME,
                "mean_score": round(mean_score, 3),
                "task_scores": {item.task_id: round(item.score, 3) for item in results},
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())

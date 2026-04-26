"""
build_dataset.py
================
Builds and saves the GRPO training dataset for the e-commerce agent.
Run this once, then load the saved dataset in the Colab notebook.

Usage:
    python build_dataset.py
    python build_dataset.py --push --hf-username YOUR_USERNAME
"""

import argparse
import json
from datasets import Dataset

SYSTEM_PROMPT = """You are an expert e-commerce customer service agent.

You have access to a variety of tools to interact with the environment including tracking orders, applying coupons, and processing returns.

RULES:
1. Never reveal fraud scores or internal risk data.
2. Always track orders before discussing their status.
3. Think before acting using <think>...</think> tags.
4. Do not repeat the same operation more than twice.
5. Budget constraints strictly apply to multi-cart purchases.
"""


def make_prompt(task_id: str, objective: str, query: str, orders: list, products: list) -> dict:
    parts = [
        f"TASK: {task_id}",
        f"OBJECTIVE: {objective}",
        f"CUSTOMER QUERY: {query}",
        f"KNOWN ORDERS: {orders}",
        f"KNOWN PRODUCTS: {products}",
        f"CART: {{}}",
        f"CURRENT SCORE: 0.00",
        f"LAST OUTCOME: Environment initialized",
        f"STATUS FLAGS: {{'operation_failures': False, 'budget_pressure': False, 'multi_cart_ready': False}}",
        f"DONE: False",
    ]
    user_content = '\n'.join(parts)
    
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "task_id": task_id,
    }


SCENARIOS = {
    "easy_order_tracking": [
        {
            "objective": "Help a customer get an accurate order status update and ETA.",
            "query": "Where is my order ORD-TRK-17? It's been a week.",
            "orders": ["ORD-TRK-17"],
            "products": ["SKU-BAG-07"]
        },
        {
            "objective": "Help a customer get an accurate order status update and ETA.",
            "query": "I need an update on ORD-TRK-17 urgently, it's a birthday gift.",
            "orders": ["ORD-TRK-17"],
            "products": ["SKU-BAG-07"]
        },
        {
            "objective": "Help a customer get an accurate order status update and ETA.",
            "query": "ORD-TRK-17 shows as pending still. Is there a problem?",
            "orders": ["ORD-TRK-17"],
            "products": ["SKU-BAG-07"]
        },
        {
            "objective": "Help a customer get an accurate order status update and ETA.",
            "query": "Can someone tell me when ORD-TRK-17 will arrive? I've been waiting.",
            "orders": ["ORD-TRK-17"],
            "products": ["SKU-BAG-07"]
        }
    ],
    "hard_policy_assessment": [
        {
            "objective": "Resolve the user's high-stress complaint according to corporate policy.",
            "query": "I want to return ORD-RET-99 and get a refund. The item arrived damaged.",
            "orders": ["ORD-RET-99"],
            "products": ["SKU-LAP-15", "SKU-MSE-01"]
        },
        {
            "objective": "Resolve the user's high-stress complaint according to corporate policy.",
            "query": "Give me a refund on ORD-RET-99 or I'll dispute the charge!",
            "orders": ["ORD-RET-99"],
            "products": ["SKU-LAP-15", "SKU-MSE-01"]
        },
        {
            "objective": "Resolve the user's high-stress complaint according to corporate policy.",
            "query": "I'd like to return the item from ORD-RET-99, wrong size ordered.",
            "orders": ["ORD-RET-99"],
            "products": ["SKU-LAP-15", "SKU-MSE-01"]
        },
        {
            "objective": "Resolve the user's high-stress complaint according to corporate policy.",
            "query": "ORD-RET-99 return please. Item not as described.",
            "orders": ["ORD-RET-99"],
            "products": ["SKU-LAP-15", "SKU-MSE-01"]
        }
    ],
    "medium_cart_recovery": [
        {
            "objective": "Recover two at-risk carts under a shared budget while handling inventory uncertainty.",
            "query": "I need a laptop with a mouse and a jacket, total budget $1400. Coupon SAVE10.",
            "orders": ["CART-A", "CART-B"],
            "products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"]
        },
        {
            "objective": "Recover two at-risk carts under a shared budget while handling inventory uncertainty.",
            "query": "Help me buy a laptop setup and a jacket as a gift, max $1400. I have coupon SAVE10.",
            "orders": ["CART-A", "CART-B"],
            "products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"]
        },
        {
            "objective": "Recover two at-risk carts under a shared budget while handling inventory uncertainty.",
            "query": "I want to complete my cart \u2014 laptop, mouse and a jacket, budget is $1400. Use SAVE10.",
            "orders": ["CART-A", "CART-B"],
            "products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"]
        }
    ],
    "easy_wishlist_browse": [
        {
            "objective": "Help the customer browse products, save items to their wishlist, and recommend alternatives.",
            "query": "I'm looking for a laptop and a bag. Can you show me what's available and save the best options to my wishlist?",
            "orders": [],
            "products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"]
        },
        {
            "objective": "Help the customer browse products, save items to their wishlist, and recommend alternatives.",
            "query": "I want to explore jackets and mice. Save the ones you recommend to my wishlist.",
            "orders": [],
            "products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"]
        },
        {
            "objective": "Help the customer browse products, save items to their wishlist, and recommend alternatives.",
            "query": "Show me your catalog and help me save a few items to my wishlist for later.",
            "orders": [],
            "products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"]
        },
    ],
    "medium_checkout_flow": [
        {
            "objective": "Complete a full checkout: add items to cart, check delivery, select address and payment, then place order.",
            "query": "I want to buy SKU-LAP-14 and SKU-BAG-07. Deliver to my home address and I'll pay with credit card.",
            "orders": [],
            "products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"]
        },
        {
            "objective": "Complete a full checkout: add items to cart, check delivery, select address and payment, then place order.",
            "query": "Help me buy a laptop and a bag. Use my work address and UPI payment.",
            "orders": [],
            "products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"]
        },
        {
            "objective": "Complete a full checkout: add items to cart, check delivery, select address and payment, then place order.",
            "query": "I need to order SKU-LAP-14 and SKU-BAG-07. Check delivery charges and complete payment.",
            "orders": [],
            "products": ["SKU-LAP-15", "SKU-LAP-14", "SKU-MSE-01", "SKU-BAG-07", "SKU-JKT-22"]
        },
    ],
    "hard_cancel_dispute": [
        {
            "objective": "Triage 3 orders: cancel pending ones, initiate returns for shipped/delivered ones.",
            "query": "Cancel ALL my orders: ORD-CAN-01, ORD-CAN-02, ORD-CAN-03. I don't want any of them!",
            "orders": ["ORD-CAN-01", "ORD-CAN-02", "ORD-CAN-03"],
            "products": ["SKU-LAP-14", "SKU-MSE-01", "SKU-JKT-22"]
        },
        {
            "objective": "Triage 3 orders: cancel pending ones, initiate returns for shipped/delivered ones.",
            "query": "I need to cancel ORD-CAN-01, ORD-CAN-02, and ORD-CAN-03 immediately. Changed my mind on everything.",
            "orders": ["ORD-CAN-01", "ORD-CAN-02", "ORD-CAN-03"],
            "products": ["SKU-LAP-14", "SKU-MSE-01", "SKU-JKT-22"]
        },
        {
            "objective": "Triage 3 orders: cancel pending ones, initiate returns for shipped/delivered ones.",
            "query": "I want a full cancellation on orders ORD-CAN-01, ORD-CAN-02, and ORD-CAN-03. Do it now!",
            "orders": ["ORD-CAN-01", "ORD-CAN-02", "ORD-CAN-03"],
            "products": ["SKU-LAP-14", "SKU-MSE-01", "SKU-JKT-22"]
        },
    ],
}

REPEATS = {
    "easy_order_tracking": 40,
    "hard_policy_assessment": 40,
    "medium_cart_recovery": 35,
    "easy_wishlist_browse": 40,
    "medium_checkout_flow": 35,
    "hard_cancel_dispute": 35,
}


def build(push: bool = False, hf_username: str = "") -> Dataset:
    records = []
    for task_id, scenario_list in SCENARIOS.items():
        repeat = REPEATS[task_id]
        for scenario in scenario_list:
            for _ in range(repeat):
                records.append(make_prompt(task_id, **scenario))

    dataset = Dataset.from_list(records)
    print(f"Dataset built: {len(dataset)} examples")

    import pandas as pd
    counts = dataset.to_pandas()["task_id"].value_counts()
    print("Distribution:")
    for task, count in counts.items():
        print(f"  {task}: {count}")

    dataset.save_to_disk("ecommerce_grpo_dataset")
    print("Saved to: ecommerce_grpo_dataset/")

    if push and hf_username:
        dataset.push_to_hub(f"{hf_username}/ecommerce-grpo-dataset")
        print(f"Pushed to: https://huggingface.co/datasets/{hf_username}/ecommerce-grpo-dataset")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hf-username", default="", help="Your HuggingFace username")
    args = parser.parse_args()
    build(push=args.push, hf_username=args.hf_username)

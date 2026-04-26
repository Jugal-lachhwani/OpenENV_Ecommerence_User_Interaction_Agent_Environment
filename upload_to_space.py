import os
from huggingface_hub import HfApi

# Load token from .env
with open(".env", "r") as f:
    for line in f:
        if line.startswith("HF_TOKEN"):
            token = line.split("=")[1].strip().strip('"').strip("'")
            break

api = HfApi(token=token)

print("Uploading to Hugging Face Space...")
api.upload_folder(
    folder_path=".",
    repo_id="Jugal15/ecommerce-agent-env",
    repo_type="space",
    ignore_patterns=[
        ".git/*",
        ".venv/*",
        "__pycache__/*",
        "*.ipynb", # Usually don't need notebooks in the space app
        "ecommerce_grpo_dataset/*", # Probably large and unnecessary for inference
        "scratch/*"
    ],
    commit_message="Pushing latest code to space"
)
print("Upload complete!")

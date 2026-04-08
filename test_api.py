import os
from pathlib import Path
from openai import OpenAI

def _load_dotenv_if_present():
    env_path = Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        print("No .env found")
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

api_key = os.environ.get("NVIDIA_API_KEY")
print(f"Loaded NVIDIA_API_KEY: {'[PRESENT]' if api_key else '[MISSING]'}")

if not api_key:
    api_key = os.environ.get("HF_TOKEN")
    print(f"Fallback to HF_TOKEN: {'[PRESENT]' if api_key else '[MISSING]'}")

base_url = "https://integrate.api.nvidia.com/v1"
model = "meta/llama-3.1-70b-instruct"

print(f"Initializing OpenAI client with Base URL: {base_url} and Model: {model}")

try:
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=5
    )
    print("API SUCCESS: ", response.choices[0].message.content)
except Exception as e:
    import traceback
    print("API FAILED:")
    traceback.print_exc()

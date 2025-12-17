import os
import requests
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = "gpt-5-nano"
api_version = "2024-12-01-preview"

url = f"{endpoint}openai/deployments/{deployment}/chat/completions?api-version={api_version}"
headers = {"Content-Type": "application/json", "api-key": api_key}

# Parameters to test
test_params = [
    ("temperature", 0.7),
    ("top_p", 0.9),
    ("max_completion_tokens", 100),
    ("n", 1),
    ("stream", False),
    ("stop", ["END"]),
    ("presence_penalty", 0.5),
    ("frequency_penalty", 0.5),
    ("logit_bias", {}),
    ("user", "test-user"),
    ("seed", 42),
    ("tools", []),
    ("tool_choice", "auto"),
    ("response_format", {"type": "text"}),
]

print("Testing GPT-5-nano parameter support:\n")
print(f"{'Parameter':<25} {'Status':<15} {'Notes'}")
print("=" * 70)

for param_name, param_value in test_params:
    data = {
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 5,
        param_name: param_value,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)

        if response.status_code == 200:
            print(f"{param_name:<25} ✓ SUPPORTED")
        else:
            error = response.json().get("error", {})
            if "Unsupported parameter" in error.get("message", ""):
                print(f"{param_name:<25} ✗ NOT SUPPORTED")
            else:
                print(f"{param_name:<25} ? ERROR: {error.get('message', '')[:30]}")

    except Exception as e:
        print(f"{param_name:<25} ? ERROR: {str(e)[:30]}")

print("\n" + "=" * 70)
print("Test complete!")

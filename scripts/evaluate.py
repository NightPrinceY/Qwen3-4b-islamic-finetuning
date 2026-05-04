"""
Step 5 — Compare base model vs fine-tuned model on the test split.

Requires the vLLM server to be running (serve.py).
"""
import json
from pathlib import Path
from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
BASE_MODEL = "qwen3-4b"
FINETUNED_MODEL = "qwen3-4b-islamic"

TEST_FILE = Path("data/splits/test.jsonl")


def load_test(path: Path, n: int = 20) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= n:
                break
    return samples


def ask(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages[:-1],  # exclude ground-truth assistant turn
        max_tokens=600,
        temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(base_url=BASE_URL, api_key="none")
    samples = load_test(TEST_FILE)

    print(f"Evaluating on {len(samples)} test samples\n")
    print("=" * 70)

    for i, sample in enumerate(samples, 1):
        messages = sample["messages"]
        question = messages[1]["content"]
        ground_truth = messages[2]["content"]

        base_answer = ask(client, BASE_MODEL, messages)
        ft_answer = ask(client, FINETUNED_MODEL, messages)

        print(f"\n[{i}] Q: {question[:80]}...")
        print(f"\n  Ground truth : {ground_truth[:150]}...")
        print(f"\n  Base model   : {base_answer[:150]}...")
        print(f"\n  Fine-tuned   : {ft_answer[:150]}...")
        print("-" * 70)


if __name__ == "__main__":
    main()

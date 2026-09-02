"""
GonkaRouter multi-model fact-checker (starter script)

What this does:
1. Takes a claim (text) as input
2. Sends it to several models on GonkaRouter with the same fact-check prompt
3. Asks each model for a 0-100 confidence score + short reasoning
4. Prints each model's score, reasoning, and Gonka Request ID
5. Computes a simple averaged "Truth Score"

Before running:
  pip install openai
  export GONKAROUTER_API_KEY="your-actual-key"      (Mac/Linux)
  set GONKAROUTER_API_KEY=your-actual-key           (Windows cmd)

NOTE: Confirm exact model ID strings in your GonkaRouter dashboard/docs —
the ones below are examples and may need updating.
"""

import os
import re
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["sk-RcsRzbUbHNLw51arJYveY5GtviqjTJRT7LWM2CPxnthasGOT"],
    base_url="https://api.gonkarouter.io/v1",
)

# Confirmed available on this account via list_models.py
MODELS = [
    "moonshotai/Kimi-K2.6",
    "MiniMaxAI/MiniMax-M2.7",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
]

FACT_CHECK_PROMPT = """You are a fact-checking assistant. Assess the following claim.

Claim: "{claim}"

Respond in exactly this format:
Score: <a number from 0 to 100, where 100 means definitely true and 0 means definitely false>
Reasoning: <2-3 sentences explaining your assessment>
"""


def query_model(model_name: str, claim: str):
    """Send the claim to one model and return (score, reasoning, request_id)."""
    prompt = FACT_CHECK_PROMPT.format(claim=claim)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content
    request_id = response.id  # this is what the track wants displayed

    # Some models leak internal reasoning in <think>...</think> tags — strip it
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Pull the score out of the model's reply (first match, after stripping <think>)
    score_match = re.search(r"Score:\s*(\d+)", text)
    score = int(score_match.group(1)) if score_match else None

    reasoning_match = re.search(r"Reasoning:\s*(.+)", text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()

    return score, reasoning, request_id


def fact_check(claim: str):
    results = []

    print(f"\nChecking claim: \"{claim}\"\n")
    print("-" * 60)

    for model_name in MODELS:
        try:
            score, reasoning, request_id = query_model(model_name, claim)
            results.append(score)

            print(f"Model: {model_name}")
            print(f"  Score: {score}")
            print(f"  Reasoning: {reasoning}")
            print(f"  Gonka Request ID: {request_id}")
            print("-" * 60)
        except Exception as e:
            print(f"Model: {model_name} — request failed: {e}")
            print("-" * 60)

    valid_scores = [s for s in results if s is not None]
    if valid_scores:
        truth_score = sum(valid_scores) / len(valid_scores)
        print(f"\nAveraged Truth Score: {truth_score:.1f}%")
    else:
        print("\nNo valid scores returned — check model IDs and API key.")


if __name__ == "__main__":
    # Quick manual test — replace with real input handling (or a CLI arg) later
    test_claim = "The Great Wall of China is visible from space with the naked eye."
    fact_check(test_claim)
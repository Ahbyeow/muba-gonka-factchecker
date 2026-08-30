"""
Vercel serverless function: POST /api/factcheck
Body: { "claim": "some claim text" }
Returns: JSON with each model's score/reasoning/request_id + averaged truth_score

Deploy notes:
- Place this file at:  api/factcheck.py  in your project root (Vercel auto-detects it)
- Set GONKAROUTER_API_KEY as an Environment Variable in your Vercel project settings
  (Project → Settings → Environment Variables) — never hardcode it here
- Add a requirements.txt in your project root containing: openai
"""

import os
import re
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GONKAROUTER_API_KEY"],
    base_url="https://api.gonkarouter.io/v1",
)

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
    prompt = FACT_CHECK_PROMPT.format(claim=claim)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content
    request_id = response.id

    # Strip any leaked <think>...</think> reasoning before parsing
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    score_match = re.search(r"Score:\s*(\d+)", text)
    score = int(score_match.group(1)) if score_match else None

    reasoning_match = re.search(r"Reasoning:\s*(.+)", text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()

    return {
        "model": model_name,
        "score": score,
        "reasoning": reasoning,
        "request_id": request_id,
    }


def run_fact_check(claim: str):
    results = []
    for model_name in MODELS:
        try:
            results.append(query_model(model_name, claim))
        except Exception as e:
            results.append({
                "model": model_name,
                "score": None,
                "reasoning": f"Request failed: {e}",
                "request_id": None,
            })

    valid_scores = [r["score"] for r in results if r["score"] is not None]
    truth_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return {
        "claim": claim,
        "results": results,
        "truth_score": truth_score,
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            claim = data.get("claim", "").strip()
        except json.JSONDecodeError:
            claim = ""

        if not claim:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing 'claim' in request body"}).encode())
            return

        result = run_fact_check(claim)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def do_OPTIONS(self):
        # Needed so the browser's CORS preflight request succeeds
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

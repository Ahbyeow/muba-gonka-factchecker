"""
Vercel serverless function: POST /api/factcheck
Body: { "claim": "some claim text OR a URL" }
Returns: JSON with each model's score/reasoning/request_id + averaged truth_score

If the input is a URL, the page is fetched and its main text extracted
before being sent to the models — the user never has to copy/paste text themselves.

Deploy notes:
- Place this file at:  api/factcheck.py  in your project root (Vercel auto-detects it)
- Set GONKAROUTER_API_KEY as an Environment Variable in your Vercel project settings
  (Project → Settings → Environment Variables) — never hardcode it here
- requirements.txt must contain: openai, requests, beautifulsoup4
"""

import os
import re
import json
import requests
from bs4 import BeautifulSoup
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

FACT_CHECK_PROMPT = """You are a fact-checking assistant. Assess the following content.

Content: "{claim}"

Respond in exactly this format:
Score: <a number from 0 to 100, where 100 means definitely true and 0 means definitely false>
Reasoning: <2-3 sentences explaining your assessment>
"""

URL_PATTERN = re.compile(r"^https?://\S+$")

# Keep the extracted text short enough to stay well within model context limits
MAX_EXTRACTED_CHARS = 6000


def is_url(text: str) -> bool:
    return bool(URL_PATTERN.match(text.strip()))


def extract_text_from_url(url: str) -> str:
    """Fetch a page and pull out its main readable text."""
    headers = {"User-Agent": "Mozilla/5.0 (Truthline fact-checker bot)"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strip elements that aren't real article content
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError("No readable text found on that page.")

    return text[:MAX_EXTRACTED_CHARS]


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


def run_fact_check(user_input: str):
    source_url = None
    content_to_check = user_input

    if is_url(user_input):
        source_url = user_input
        try:
            content_to_check = extract_text_from_url(user_input)
        except Exception as e:
            return {
                "claim": user_input,
                "source_url": user_input,
                "results": [],
                "truth_score": None,
                "error": f"Could not fetch or read that URL: {e}",
            }

    results = []
    for model_name in MODELS:
        try:
            results.append(query_model(model_name, content_to_check))
        except Exception as e:
            results.append({
                "model": model_name,
                "score": None,
                "reasoning": f"Request failed: {e}",
                "request_id": None,
            })

    valid_scores = [r["score"] for r in results if r["score"] is not None]
    truth_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    # Flag disagreement: a wide score spread means models genuinely conflict,
    # which is a different (and more important) signal than "uncertain"
    DISAGREEMENT_THRESHOLD = 40
    score_spread = max(valid_scores) - min(valid_scores) if len(valid_scores) >= 2 else 0
    disagreement = score_spread > DISAGREEMENT_THRESHOLD

    return {
        "claim": user_input,
        "source_url": source_url,
        "results": results,
        "truth_score": truth_score,
        "score_spread": score_spread,
        "disagreement": disagreement,
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

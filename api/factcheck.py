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
 
Detect the language the content is written in, and write your Reasoning in that
same language. Keep the words "Score:" and "Reasoning:" themselves in English exactly
as shown below, since they are used to parse your response — only the reasoning
sentences should be translated.
 
Respond in exactly this format:
Score: <a number from 0 to 100, where 100 means definitely true and 0 means definitely false>
Reasoning: <2-3 sentences explaining your assessment, in the same language as the content>
"""
 
CLAIM_CHECK_PROMPT = """Look at the following content and decide if it contains at least
one specific, checkable factual claim (something that could be true or false, like a
statistic, event, or assertion about the world) — as opposed to being pure opinion,
fiction, a recipe, or content with nothing factual to verify.
 
Content: "{content}"
 
Respond with exactly one word: YES or NO.
"""
 
CLAIM_EXTRACTION_PROMPT = """Read the following content and extract up to 5 distinct,
specific, checkable factual claims made in it — each one a self-contained sentence that
could be verified true or false on its own, without needing the rest of the article for
context. Skip opinions, predictions, and vague statements.
 
Content: "{content}"
 
Respond with ONLY a JSON array of strings, nothing else. Example format:
["Claim one as a full sentence.", "Claim two as a full sentence."]
 
If you genuinely cannot find any distinct checkable claims, respond with: []
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
 
 
def has_checkable_claim(content: str) -> bool:
    """One cheap model call to screen out content with nothing factual to check
    (opinions, recipes, fiction, etc.) before running the full multi-model pipeline."""
    response = client.chat.completions.create(
        model=MODELS[0],
        messages=[{"role": "user", "content": CLAIM_CHECK_PROMPT.format(content=content)}],
    )
    answer = response.choices[0].message.content
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer.strip().upper().startswith("YES")
 
 
# Below this length, skip the extraction call — a very short input (roughly
# one sentence) is essentially never going to contain multiple distinct claims,
# so this only saves a call on the trivial case, not on real multi-claim content.
MULTI_CLAIM_LENGTH_THRESHOLD = 120
 
 
def extract_claims(content: str) -> list:
    """Pull out up to 5 distinct checkable claims from the content. Falls back
    to treating the whole thing as one claim if extraction fails, finds nothing,
    or the content is short enough that splitting wouldn't apply anyway."""
    if len(content) < MULTI_CLAIM_LENGTH_THRESHOLD:
        return [content]
 
    try:
        response = client.chat.completions.create(
            model=MODELS[0],
            messages=[{"role": "user", "content": CLAIM_EXTRACTION_PROMPT.format(content=content)}],
        )
        text = response.choices[0].message.content
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
 
        # Models sometimes wrap JSON in a markdown code fence — strip that if present
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
 
        claims = json.loads(text)
        claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
 
        if not claims:
            return [content]
        return claims[:5]
    except Exception:
        # If extraction fails for any reason, fall back to checking the whole thing at once
        return [content]
 
 
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
 
 
def check_single_claim(claim_text: str) -> dict:
    """Run the full 3-model cross-verification on one specific claim."""
    results = []
    for model_name in MODELS:
        try:
            results.append(query_model(model_name, claim_text))
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
        "claim_text": claim_text,
        "results": results,
        "truth_score": truth_score,
        "score_spread": score_spread,
        "disagreement": disagreement,
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
                "claims": [],
                "truth_score": None,
                "error": f"Could not fetch or read that URL: {e}",
            }
 
    try:
        if not has_checkable_claim(content_to_check):
            return {
                "claim": user_input,
                "source_url": source_url,
                "claims": [],
                "truth_score": None,
                "no_checkable_claim": True,
                "error": "This content doesn't appear to contain a specific factual claim to check "
                         "(it may be opinion, fiction, or something else with nothing to verify).",
            }
    except Exception:
        # If the pre-check call itself fails, don't block the whole feature —
        # just proceed to the full pipeline as normal.
        pass
 
    claim_list = extract_claims(content_to_check)
 
    checked_claims = [check_single_claim(c) for c in claim_list]
 
    claim_scores = [c["truth_score"] for c in checked_claims if c["truth_score"] is not None]
    overall_truth_score = sum(claim_scores) / len(claim_scores) if claim_scores else None
    overall_disagreement = any(c["disagreement"] for c in checked_claims)
 
    return {
        "claim": user_input,
        "source_url": source_url,
        "claims": checked_claims,
        "truth_score": overall_truth_score,
        "disagreement": overall_disagreement,
        "multi_claim": len(checked_claims) > 1,
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
 
# Truthline

**MUBA Hacks 2026 — Gonka Track (AI for Society)**

A multi-model fact-checking tool. Paste a claim or a URL, and Truthline sends it to
several independent AI models through [GonkaRouter](https://gonkarouter.io), compares
their answers, and returns a truth score with full transparency into where the models
agree — and where they don't.

---

## Problem Statement

Most AI fact-checkers rely on a single model's judgment, presented as if it were
authoritative. That's misleading: a single model can be confidently wrong, and
presenting one AI's opinion as "the answer" hides the very real disagreement that
exists between models on ambiguous or contested claims.

Truthline addresses this by never trusting one model alone. Every claim is
independently checked by multiple models, and when they disagree, that disagreement
is surfaced to the user as a signal in itself — not averaged away into false
confidence.

## Project Description

Truthline accepts either a typed claim or a URL to an article:

- If given a URL, it fetches and extracts the readable article text automatically
- It screens out content with no checkable factual claim (opinion, fiction, recipes,
  etc.) before running the full pipeline
- For longer content, it automatically extracts **multiple distinct claims** and
  scores each one separately — rather than collapsing an entire article into one
  blended number
- Each claim is sent to **3 independent models** via GonkaRouter:
  - `moonshotai/Kimi-K2.6`
  - `MiniMaxAI/MiniMax-M2.7`
  - `deepseek-ai/DeepSeek-V4-Flash-0731`
- **Model disagreement** is flagged explicitly when scores diverge significantly,
  instead of being hidden inside a clean-looking average
- Responses are given in the same language as the input claim (multilingual support)
- The **GonkaRouter Request ID** for every model call is displayed, as verifiable
  proof each check actually ran through the network

## Blockchain Technology Used

**Not applicable.** This project is built for the Gonka Track (AI for Society), which
requires all AI reasoning to run through GonkaRouter's decentralized inference network —
it does not involve a blockchain, wallet, or smart contract layer. GonkaRouter's own
underlying network handles proof-of-compute verification for AI inference; this
application interacts with it only through its API, not on-chain.

## Smart Contract Addresses (Testnet)

**Not applicable** — no smart contracts are used in this submission.

## Tech Stack

- **Frontend:** Plain HTML / CSS / JavaScript (no framework)
- **Backend:** Python serverless function (Vercel)
- **AI Inference:** [GonkaRouter](https://gonkarouter.io) (OpenAI-compatible API)
- **Deployment:** Vercel
- **URL content extraction:** `requests` + `BeautifulSoup`

## Setup and Installation

### Prerequisites
- Python 3.12 or higher
- Node.js (for the Vercel CLI)
- A [GonkaRouter](https://gonkarouter.io) account and API key

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install the Vercel CLI
```bash
npm install -g vercel
```

### 3. Set up environment variables
Create a `.env.local` file in the project root:
```
GONKAROUTER_API_KEY=your-gonkarouter-api-key
```

### 4. Run locally
```bash
vercel dev
```
Visit the local URL it prints (typically `http://localhost:3000`).

### 5. Deploy
```bash
vercel --prod
```
Make sure `GONKAROUTER_API_KEY` is also set in your Vercel project's
**Settings → Environment Variables** for production.

## Project Structure
```
├── api/
│   └── factcheck.py      # Backend endpoint: claim/URL fact-checking logic
├── index.html             # Frontend UI
├── requirements.txt       # Python dependencies
└── .gitignore
```

## Live Demo
[your-deployed-vercel-url-here]

## Team Members

| Name | Role |
|---|---|
| Lew Jing Yuan| TP090232 |
| Lee Chuen Jin | TP089620 |
| Cheok Kin Fung | TP089654 |
| Ryan Lim Yi Heng | TP088622 |

---

*Built for MUBA Hacks 2026 — Gonka Track.*

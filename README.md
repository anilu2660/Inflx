# Inflx AI Sales Agent

**Inflx** is an AI-powered automated video editing SaaS product by **AutoStream**. This repository contains a conversational AI sales agent that understands user intent, retrieves accurate product information via RAG, identifies high-intent users, collects their details, and executes a lead capture workflow — powered by LangChain, LangGraph, and GPT-4o-mini.

---

## Product: Inflx by AutoStream

| Plan | Price | Videos | Resolution | AI Captions | Support |
|---|---|---|---|---|---|
| **Basic** | $29/month | 10 videos/month | 720p | No | Email (24hr) |
| **Pro** | $79/month | Unlimited | 4K | Yes | 24/7 Priority |

**Company Policies:**
- No refunds after 7 days
- 24/7 support available only on the Pro plan
- 14-day free trial — no credit card required
- Cancel anytime, no cancellation fees

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/anilu2660/Inflx.git
cd Inflx
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your OpenAI API key

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and set your key:

```
OPENAI_API_KEY=sk-your-key-here
```

### 5. Run the interactive agent

```bash
python main.py
```

### 6. Run the automated workflow test

```bash
python test_flow.py
```

Simulates the full 5-step expected conversation flow end-to-end and prints results with intent labels and state debug info.

---

## Architecture Explanation (~200 words)

The Inflx AI Sales Agent is built using **LangGraph**, a state-graph framework from LangChain that enables explicit, controllable multi-step agent workflows.

The agent is composed of **5 nodes**, each responsible for a single concern:

1. **`detect_intent`** — Uses GPT-4o-mini to classify every user message into one of three intents: `greeting`, `inquiry`, or `high_intent`.
2. **`handle_greeting`** — Returns a warm welcome message introducing Inflx.
3. **`handle_inquiry`** — Runs a RAG pipeline: retrieves the top-3 most relevant documents from a FAISS vector store built from `knowledge.json`, then generates a grounded answer using only that context — never hallucinating.
4. **`handle_high_intent`** — Extracts user details (name, email, platform) from the message using an LLM extraction call, identifies what's still missing, and asks for it naturally.
5. **`capture_lead`** — Called only when all three fields are present. Fires `mock_lead_capture(name, email, platform)`.

### Why LangGraph?

LangGraph was chosen over simple chain-based approaches because it provides **conditional routing** (each turn routes to a different node based on intent), **persistent typed state** across all turns via `AgentState`, and **clean node isolation** — making the system easier to test, extend, and debug. New nodes (e.g., upsell, churn prevention) can be added without touching existing logic.

### How State is Managed

`AgentState` is a `TypedDict` with LangGraph's `add_messages` reducer for conversation history, plus individual fields: `user_name`, `user_email`, `user_platform`, and `lead_captured`. These fields persist across every turn so the agent never re-asks for information already collected.

---

## WhatsApp Deployment — Webhook Integration

To deploy this agent on WhatsApp, we would use the **Meta WhatsApp Business API** with a webhook-based architecture:

### How It Works

```
WhatsApp User sends message
        |
  Meta Server (webhook POST)
        |
  FastAPI Endpoint: POST /webhook
        |
  Parse incoming message body
        |
  Pass message text to LangGraph agent
        |
  agent.invoke(state) --> generates response
        |
  Call WhatsApp Send Message API
        |
  User receives reply on WhatsApp
```

### Implementation Steps

1. **Wrap the agent in a FastAPI app** — expose a `POST /webhook` endpoint that Meta can call.

2. **Verify the webhook** — implement a `GET /webhook` endpoint to verify the endpoint with Meta's challenge token.

3. **Parse the payload** — extract the user's phone number and message text from the WhatsApp webhook JSON payload:

```python
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    message = data["entry"][0]["changes"][0]["value"]["messages"][0]
    user_phone = message["from"]
    user_text = message["text"]["body"]
    # pass to agent...
```

4. **Maintain per-user session state** — store each user's `AgentState` in a dictionary keyed by phone number so conversation memory persists across messages.

5. **Send reply via WhatsApp API** — after invoking the graph, POST the agent's response to Meta's messages endpoint:

```python
requests.post(
    f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
    json={
        "messaging_product": "whatsapp",
        "to": user_phone,
        "text": {"body": agent_response}
    }
)
```

6. **Deploy on a public server** — Meta requires a publicly accessible HTTPS URL. Deploy the FastAPI app on Railway, Render, or AWS with an SSL certificate.

This approach allows the same intent detection + RAG + lead capture pipeline to operate natively inside WhatsApp conversations at scale, with zero changes to the core agent logic.

---

## Project Structure

```
Inflx/
├── main.py                  <- CLI entry point (interactive REPL)
├── test_flow.py             <- Automated test of the 5-step conversation workflow
├── agent/
│   ├── graph.py             <- LangGraph workflow (5 nodes, conditional routing)
│   └── state.py             <- AgentState TypedDict schema
├── rag/
│   ├── knowledge.json       <- 15-entry Inflx product knowledge base
│   ├── loader.py            <- JSON -> LangChain Documents
│   └── retriever.py         <- FAISS vector store + retriever
├── tools/
│   └── lead_capture.py      <- mock_lead_capture(name, email, platform)
├── utils/
│   └── intent.py            <- GPT-4o-mini intent classifier
├── requirements.txt
├── .env.example
└── README.md
```

---

## Expected Conversation Flow

| Step | Who | Message | What Happens |
|---|---|---|---|
| 1 | User | *"Hi, tell me about your pricing."* | Intent -> `inquiry`; RAG fetches pricing docs |
| 2 | Agent | Answers with accurate Basic/Pro pricing | From knowledge base only |
| 3 | User | *"That sounds good, I want to try the Pro plan for my YouTube channel."* | Intent -> `high_intent`; platform `YouTube` extracted |
| 4 | Agent | Asks for name and email | Platform already known from message |
| 5 | User | *"I'm Anuj, anuj@email.com"* | Name + email extracted |
| 6 | Agent | Confirms + fires lead capture tool | `mock_lead_capture(Anuj, anuj@email.com, YouTube)` |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Framework | LangChain + LangGraph |
| LLM | GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS |
| Memory | LangGraph AgentState (add_messages reducer) |
| Environment | python-dotenv |

---

## Constraints

- Answers are **never hardcoded** — all product info comes from the RAG knowledge base
- Lead capture tool fires **only after** all 3 fields (name, email, platform) are collected
- Conversation **memory persists** across all turns via LangGraph state
- Intent is **LLM-classified** on every turn — no keyword matching

# AutoStream AI Sales Agent

An AI-powered conversational sales agent for **AutoStream** — an automated video editing SaaS platform for content creators. The agent understands user intent, retrieves accurate product information via RAG, identifies high-intent users, collects their details, and executes a lead capture workflow — all powered by LangChain, LangGraph, and GPT-4o-mini.

---

## Product: AutoStream

AutoStream is an AI-powered video editing SaaS that helps content creators produce professional-quality videos effortlessly.

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

## Features

- **Intent Detection** — LLM classifies each user message into `greeting`, `inquiry`, or `high_intent`
- **RAG Pipeline** — All product answers retrieved from a FAISS vector store (no hardcoded responses)
- **Conversation Memory** — State persists across all turns via LangGraph's state graph
- **Lead Qualification** — Detects buying intent; extracts platform from context; collects name and email
- **Tool Execution** — Calls `mock_lead_capture()` only after all required fields are collected
- **Styled CLI** — Interactive REPL with colored output

---

## Architecture

```
User Input (CLI)
      |
 detect_intent_node          <- GPT-4o-mini classifies intent
      |
 +------------------------------------+
 |  greeting    -> handle_greeting    |   (warm intro)
 |  inquiry     -> handle_inquiry     |   (RAG retrieval -> answer)
 |  high_intent -> handle_high_intent |
 |                    |               |
 |               All details present? |
 |               YES -> capture_lead  |   (mock_lead_capture called)
 |               NO  -> ask for missing fields
 +------------------------------------+
```

### Expected Conversation Flow

| Step | Who | Message | What Happens |
|---|---|---|---|
| 1 | User | *"Hi, tell me about your pricing."* | Intent → `inquiry`; RAG fetches pricing docs |
| 2 | Agent | Answers with accurate Basic/Pro pricing | From knowledge base only |
| 3 | User | *"That sounds good, I want to try the Pro plan for my YouTube channel."* | Intent → `high_intent`; platform `YouTube` extracted |
| 4 | Agent | Asks for name and email | Platform already known from message |
| 5 | User | *"I'm Anuj, anuj@email.com"* | Name + email extracted |
| 6 | Agent | Confirms + fires lead capture tool | `mock_lead_capture(Anuj, anuj@email.com, YouTube)` |

---

### Why LangGraph?

LangGraph provides an explicit state machine architecture for multi-step agents:

- **Persistent state** across turns without manual threading — all conversation history and collected fields (name, email, platform) live in `AgentState`
- **Conditional routing** — each turn can route to a different node based on intent and data completeness
- **Clean node boundaries** — intent detection, RAG, lead collection, and tool execution are fully isolated
- **Scalable** — new nodes (e.g., upsell, support escalation) can be added without refactoring

### How Memory Works

The `AgentState` TypedDict holds the full conversation history (`messages: list`) using LangGraph's `add_messages` reducer, which appends messages on each turn rather than replacing the list. Fields like `user_name`, `user_email`, `user_platform`, and `lead_captured` persist across turns, so the agent never re-asks for information already provided.

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
│   ├── knowledge.json       <- 15-entry product knowledge base
│   ├── loader.py            <- JSON -> LangChain Documents
│   └── retriever.py         <- FAISS vector store + retriever
├── tools/
│   └── lead_capture.py      <- mock_lead_capture() function
├── utils/
│   └── intent.py            <- GPT-4o-mini intent classifier
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Navigate to the project

```bash
cd path/to/Inflx
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

### 4. Configure environment variables

```bash
# Copy the template
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux

# Edit .env and paste your OpenAI API key
OPENAI_API_KEY=sk-...
```

---

## Running

### Interactive chat

```bash
python main.py
```

The agent loads the knowledge base, builds the FAISS index, and starts an interactive chat session.

### Run the automated workflow test

```bash
python test_flow.py
```

Simulates the full 5-step expected conversation flow and prints each step with intent labels and state debug info.

---

## Example Session

```
You: Hi, tell me about your pricing.

AutoStream Agent:
AutoStream offers two pricing plans.
  - Basic Plan: $29/month — 10 videos/month at 720p, email support
  - Pro Plan:   $79/month — unlimited videos at 4K, AI captions, 24/7 priority support
Both come with a 14-day free trial, no credit card required.

You: That sounds good, I want to try the Pro plan for my YouTube channel.

AutoStream Agent:
That's awesome! To get you started, could you please share your name and email address?

You: I'm Anuj, anuj@email.com

==================================================
  *** LEAD CAPTURED SUCCESSFULLY! ***
  Name:     Anuj
  Email:    anuj@email.com
  Platform: YouTube
==================================================

AutoStream Agent:
Great news, Anuj! I've captured your details. Our team will reach out
to you at anuj@email.com shortly. Start your 14-day free trial at
autostream.io — no credit card required!
```

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

---

## Future Enhancements

- Deploy as API using FastAPI
- Integrate with WhatsApp via Meta Business API webhooks
- Store leads in a database (PostgreSQL / Supabase)
- Add analytics dashboard for captured leads
- Multi-language support

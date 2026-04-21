# 📘 Project Plan: Social-to-Lead AI Agent (AutoStream)

You are a senior product manager and you should plan and build everything according to plan mentioned below:

## 🧠 Objective

Build a conversational AI agent that:

- Understands user intent
- Retrieves accurate product information using RAG
- Identifies high-intent users
- Collects user details
- Executes a lead capture function

---

## 🏗️ System Architecture Overview

The system consists of the following components:

1. **User Interface (CLI / Chat Interface)**
2. **Intent Detection Module**
3. **RAG Pipeline (Knowledge Retrieval)**
4. **Conversation Memory (State Management)**
5. **Lead Qualification Logic**
6. **Tool Execution (Lead Capture API)**

---

## ⚙️ Tech Stack

- **Language:** Python 3.9+
- **Framework:** LangChain + LangGraph
- **LLM:** GPT-4o-mini
- **Vector Store:** FAISS
- **Embeddings:** OpenAI /OpenAI Embeddings
- **Memory:** ConversationBufferMemory / LangGraph State

---

## 📅 Development Plan

### 🔹 Phase 1: Project Setup

- Initialize GitHub repository
- Create folder structure:

  ```
  project/
  ├── main.py
  ├── agent/
  ├── rag/
  ├── tools/
  ├── data/
  ├── utils/
  ├── requirements.txt
  └── README.md
  ```

- Install dependencies

---

### 🔹 Phase 2: Knowledge Base Creation

- Create `knowledge.json` with:
  - Pricing (Basic & Pro)
  - Features
  - Policies

- Convert data into documents for retrieval

---

### 🔹 Phase 3: RAG Pipeline

- Load knowledge base
- Generate embeddings
- Store in FAISS vector DB
- Create retriever function
- Test:
  - Query → Correct answer from knowledge base

---

### 🔹 Phase 4: Intent Detection Module

- Build LLM-based classifier
- Categories:
  - Greeting
  - Inquiry
  - High Intent

- Validate with test inputs

---

### 🔹 Phase 5: Memory & State Management

- Implement conversation memory
- Store:
  - Previous messages
  - Collected user details

- Ensure context persists across 5–6 turns

---

### 🔹 Phase 6: Lead Capture Tool

- Implement function:

  ```python
  def mock_lead_capture(name, email, platform):
      print(f"Lead captured successfully: {name}, {email}, {platform}")
  ```

- Add validation:
  - Trigger only after all fields are collected

---

### 🔹 Phase 7: Agent Logic (Core Brain)

Flow:

```
User Input
   ↓
Intent Detection
   ↓
IF Greeting → Respond
IF Inquiry → RAG Response
IF High Intent → Collect Details
   ↓
Check Missing Info
   ↓
Call Tool (Lead Capture)
```

---

### 🔹 Phase 8: LangGraph Workflow (Recommended)

Nodes:

- Input Node
- Intent Node
- RAG Node
- Lead Collection Node
- Tool Execution Node

Benefits:

- Clear state transitions
- Better scalability
- Cleaner architecture

---

### 🔹 Phase 9: Testing Scenarios

Test cases:

1. Greeting → Proper response
2. Pricing query → Accurate RAG answer
3. High intent → Detected correctly
4. Missing details → Agent asks correctly
5. Tool execution → Triggered only when complete

---

### 🔹 Phase 10: Documentation (README.md)

Include:

- Setup instructions
- How to run
- Architecture explanation (~200 words)
- Design decisions:
  - Why LangGraph
  - How memory works

- WhatsApp integration explanation

---

### 🔹 Phase 11: Demo Video

Show:

1. User asks pricing
2. Agent answers using RAG
3. User shows interest
4. Agent collects details
5. Lead capture executed successfully

---

## 🔁 Conversation Flow Example

**User:** Hi, what’s your pricing?
→ Agent responds using RAG

**User:** I want the Pro plan
→ Agent detects high intent

**Agent:** Can I get your name, email, and platform?

**User:** Anuj, [anuj@email.com](mailto:anuj@email.com), YouTube
→ Agent calls lead capture function

---

## ⚠️ Constraints & Rules

- Do NOT hardcode answers (must use RAG)
- Do NOT call tool before collecting all inputs
- Must maintain conversation memory
- Must correctly classify intent

---

<!-->

## 🚀 Future Enhancements

- Deploy as API using FastAPI
- Integrate with WhatsApp via Webhooks
- Add database for storing leads
- Add analytics dashboard
- Multi-language support

---

-->

## ✅ Success Criteria

- Accurate intent detection
- Correct RAG responses
- Smooth conversation flow
- Proper state handling
- Clean and modular code

---

## 🧩 Final Output

A working AI agent that:

- Acts like a sales assistant
- Converts conversations into leads
- Demonstrates real-world AI system design

---

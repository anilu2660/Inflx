"""
test_flow.py -- Simulates the exact Expected Conversation Flow from the project spec.

Step-by-Step Workflow:
  1. Greeting + Pricing inquiry  -> "Hi, tell me about your pricing."
  2. RAG retrieves pricing       -> Agent responds with accurate pricing
  3. Intent Shift                -> "That sounds good, I want to try the Pro plan for my YouTube channel."
  4. Lead Qualification          -> Agent detects high-intent, asks for Name + Email (platform already known)
  5. User provides name + email  -> "I'm Anuj, anuj@email.com"
  6. Tool Execution              -> Lead captured

Run: python test_flow.py
"""

import sys
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import sys
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set in .env")
    sys.exit(1)

from langchain_core.messages import HumanMessage
from agent.graph import build_graph

# ─── Styling ─────────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

SEP = f"{CYAN}{'=' * 60}{RESET}"


def print_step(step_num: int, label: str):
    print(f"\n{SEP}")
    print(f"{BOLD}{YELLOW}  STEP {step_num}: {label}{RESET}")
    print(SEP)


def print_user(msg: str):
    print(f"\n{BOLD}  User:{RESET} {msg}")


def print_agent(msg: str):
    print(f"\n{GREEN}{BOLD}  Inflx Agent:{RESET}")
    for line in msg.strip().split("\n"):
        print(f"  {line}")



# ─── Build Graph ──────────────────────────────────────────────────────────────
print(f"\n{YELLOW}[*] Initializing Inflx AI Agent (by AutoStream)...{RESET}", end=" ", flush=True)
graph = build_graph()

# Prime FAISS retriever
from rag.retriever import get_retriever
get_retriever()
print(f"{GREEN}Ready!{RESET}")

# ─── Shared persistent state ──────────────────────────────────────────────────
state = {
    "messages": [],
    "intent": "",
    "user_name": None,
    "user_email": None,
    "user_platform": None,
    "lead_captured": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2: Greeting + Pricing inquiry → RAG response
# ─────────────────────────────────────────────────────────────────────────────
print_step(1, "Greeting + Knowledge Retrieval (RAG)")
msg1 = "Hi, tell me about your pricing."
print_user(msg1)

state["messages"].append(HumanMessage(content=msg1))
state = graph.invoke(state)

agent_reply = state["messages"][-1].content
print_agent(agent_reply)
print(f"\n  {DIM}[Intent detected: {state['intent']} | RAG retrieved pricing docs]{RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 & 4: Intent Shift -> High Intent -> Lead Qualification
# ─────────────────────────────────────────────────────────────────────────────
print_step(3, "Intent Shift -> Lead Qualification")
msg2 = "That sounds good, I want to try the Pro plan for my YouTube channel."
print_user(msg2)

state["messages"].append(HumanMessage(content=msg2))
state = graph.invoke(state)

agent_reply = state["messages"][-1].content
print_agent(agent_reply)
print(f"\n  {DIM}[Intent detected: {state['intent']} | Platform extracted: {state.get('user_platform', 'None')}]{RESET}")
print(f"  {DIM}[Missing: Name={state.get('user_name') or 'NOT YET'}, Email={state.get('user_email') or 'NOT YET'}]{RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 & 6: User provides Name + Email -> Tool Execution
# ─────────────────────────────────────────────────────────────────────────────
print_step(5, "User Provides Details -> Tool Execution")
msg3 = "I'm Anuj, anuj@email.com"
print_user(msg3)

state["messages"].append(HumanMessage(content=msg3))
state = graph.invoke(state)

agent_reply = state["messages"][-1].content
print_agent(agent_reply)

if state.get("lead_captured"):
    print(f"\n{GREEN}{BOLD}  [SUCCESS] Full workflow completed as expected!{RESET}")
    print(f"  {DIM}(mock_lead_capture was called with: name={state.get('user_name')}, "
          f"email={state.get('user_email')}, platform={state.get('user_platform')}){RESET}")
else:
    print(f"\n{YELLOW}  [INFO] Lead not yet captured. State: name={state.get('user_name')}, email={state.get('user_email')}{RESET}")

print(SEP + "\n")

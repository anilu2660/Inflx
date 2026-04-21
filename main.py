"""
Inflx AI Sales Agent -- by AutoStream
CLI entry point -- runs the agent in an interactive REPL loop.

Usage:
    python main.py

Ensure you have a .env file with OPENAI_API_KEY set.
"""
import sys
import io
# Force UTF-8 output on Windows to handle emoji / unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found.")
    print("   Create a .env file with: OPENAI_API_KEY=your_key_here")
    exit(1)

from agent.graph import build_graph

# ANSI color codes for styled output
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

BANNER = f"""
{BOLD}{CYAN}================================================
            Inflx AI Sales Assistant
         AI Video Editing for Creators
              by AutoStream
================================================{RESET}

{YELLOW}Type your message and press Enter to chat.
Type 'quit' or 'exit' to end the session.{RESET}
"""


def print_agent_response(response: str):
    """Print the agent's response with styling."""
    print(f"{GREEN}{BOLD}Inflx Agent:{RESET}")
    print(f"{response}\n")
    print(f"{CYAN}{'─' * 50}{RESET}")


def run_agent():
    """Main REPL loop for the Inflx AI agent."""
    print(BANNER)

    # Build and compile the graph once
    print(f"{YELLOW}[*] Initializing Inflx AI Agent...{RESET}", end="\r")
    graph = build_graph()

    # Prime the retriever so first query is fast
    from rag.retriever import get_retriever
    get_retriever()
    print(f"{GREEN}[OK] Inflx Agent ready! Knowledge base loaded.            {RESET}\n")
    print(f"{CYAN}{'─' * 50}{RESET}")

    # Persistent state across conversation turns
    state = {
        "messages": [],
        "intent": "",
        "user_name": None,
        "user_email": None,
        "user_platform": None,
        "lead_captured": False,
    }

    while True:
        try:
            user_input = input(f"\n{BOLD}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{YELLOW}Goodbye! Thanks for chatting with Inflx by AutoStream.{RESET}\n")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print(f"\n{YELLOW}Thanks for your interest in Inflx! See you soon.{RESET}\n")
            break

        # Append the new human message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Invoke the graph with the current state
        try:
            result = graph.invoke(state)
            # Update the persistent state with the result
            state.update(result)

            # Extract and print the latest AI response
            if state["messages"]:
                latest_response = state["messages"][-1].content
                print_agent_response(latest_response)

            # Notify if lead was just captured
            if state.get("lead_captured"):
                print(
                    f"{YELLOW}[LEAD CAPTURED] "
                    f"Conversation will continue if you have more questions.{RESET}\n"
                )

        except Exception as e:
            print(f"\n{YELLOW}⚠  An error occurred: {e}{RESET}\n")
            print("Please try again or check your OPENAI_API_KEY.")


if __name__ == "__main__":
    run_agent()

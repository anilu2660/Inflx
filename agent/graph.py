"""
LangGraph workflow for the Inflx Social-to-Lead AI Agent.
Inflx is an AI-powered video editing product by AutoStream.

Nodes:
  - detect_intent_node    : Classifies user input intent
  - handle_greeting_node  : Responds to greetings
  - handle_inquiry_node   : RAG-based product Q&A
  - handle_high_intent_node: Collects user details or confirms all collected
  - capture_lead_node     : Executes mock lead capture

Routing:
  detect_intent → greeting | inquiry | high_intent
  high_intent   → capture_lead (if complete) | END (if details missing)
"""

import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from utils.intent import detect_intent
from rag.retriever import get_retriever
from tools.lead_capture import mock_lead_capture


# ---------------------------------------------------------------------------
# LLM factory — instantiated lazily inside each node so that the module
# can be imported before OPENAI_API_KEY is loaded into the environment.
# ---------------------------------------------------------------------------
def _get_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


# ---------------------------------------------------------------------------
# Helper: format conversation history for LLM
# ---------------------------------------------------------------------------
def _format_history(messages: list) -> str:
    """Format conversation history into a readable string."""
    lines = []
    for msg in messages[:-1]:  # exclude latest human message
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Agent: {msg.content}")
    return "\n".join(lines) if lines else "No prior conversation."


# ---------------------------------------------------------------------------
# Node 1: Detect Intent
# ---------------------------------------------------------------------------
def detect_intent_node(state: AgentState) -> dict:
    """Classify the latest user message and update state intent."""
    latest_message = state["messages"][-1].content
    intent = detect_intent(latest_message)
    return {"intent": intent}


# ---------------------------------------------------------------------------
# Node 2: Handle Greeting
# ---------------------------------------------------------------------------
def handle_greeting_node(state: AgentState) -> dict:
    """Respond warmly to greetings and introduce the agent."""
    history = _format_history(state["messages"])

    system = SystemMessage(content="""You are the Inflx AI sales assistant, a product by AutoStream.
Your job is to welcome users warmly and briefly introduce Inflx as AutoStream's AI-powered 
automated video editing platform for content creators. Keep your response short, friendly, and inviting.
Mention that you can help with pricing, features, or getting started with Inflx.""")

    human = HumanMessage(content=f"""Conversation so far:
{history}

Latest user message: {state["messages"][-1].content}

Respond with a warm, helpful greeting.""")

    response = _get_llm().invoke([system, human])
    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# Node 3: Handle Inquiry (RAG)
# ---------------------------------------------------------------------------
def handle_inquiry_node(state: AgentState) -> dict:
    """Answer product questions using RAG-retrieved knowledge."""
    user_query = state["messages"][-1].content
    history = _format_history(state["messages"])

    # Retrieve relevant documents
    retriever = get_retriever(k=3)
    docs = retriever.invoke(user_query)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    system = SystemMessage(content="""You are the Inflx AI sales assistant, a product by AutoStream company.
Inflx is AutoStream's AI-powered automated video editing SaaS for content creators.
Answer the user's question using ONLY the provided context below. 
Do NOT make up any information. If the context doesn't cover the question, say so honestly.
Be concise, helpful, and conversational. 
If the user seems interested, gently suggest they try Inflx's 14-day free trial.""")

    human = HumanMessage(content=f"""Context from Inflx knowledge base:
{context}

Conversation history:
{history}

User question: {user_query}

Answer the question using only the context provided.""")

    response = _get_llm().invoke([system, human])
    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# Node 4: Handle High Intent (Lead Collection)
# ---------------------------------------------------------------------------
def handle_high_intent_node(state: AgentState) -> dict:
    """
    Check which user details are missing and ask for them.
    If all details are present, acknowledge and prepare for capture.
    Also attempts to parse details from the current message.
    """
    user_message = state["messages"][-1].content
    history = _format_history(state["messages"])

    # Try to extract details from the current message
    name = state.get("user_name")
    email = state.get("user_email")
    platform = state.get("user_platform")

    # Use LLM to extract any user details from the message
    extract_system = SystemMessage(content="""Extract user details from the message if present.
Return a JSON object with these exact keys: name, email, platform.
Use null for any field not found in the message.
Platform refers to the platform they create content for (e.g., YouTube, Instagram, TikTok, Twitter, LinkedIn, Facebook).
Only extract information explicitly stated. Do not guess.
Return ONLY the JSON object, nothing else.

Example: {"name": "Anuj", "email": "anuj@email.com", "platform": "YouTube"}""")

    extract_human = HumanMessage(content=f"Message: {user_message}")
    extraction_response = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(
        [extract_system, extract_human]
    )

    # Parse extracted details
    import json
    try:
        extracted = json.loads(extraction_response.content.strip())
        if extracted.get("name") and not name:
            name = extracted["name"]
        if extracted.get("email") and not email:
            email = extracted["email"]
        if extracted.get("platform") and not platform:
            platform = extracted["platform"]
    except (json.JSONDecodeError, AttributeError):
        pass

    # Build list of missing fields
    missing = []
    if not name:
        missing.append("your name")
    if not email:
        missing.append("your email address")
    if not platform:
        missing.append("which social media platform you primarily use")

    updates = {
        "user_name": name,
        "user_email": email,
        "user_platform": platform,
    }

    if missing:
        # Ask for the missing details
        system = SystemMessage(content="""You are the Inflx AI sales assistant, a product by AutoStream company.
Inflx is AutoStream's AI-powered automated video editing platform for content creators.
The user has shown high interest in Inflx. 
You need to collect their contact details to capture them as a lead.
Be enthusiastic but not pushy. Ask naturally for the missing information.""")

        missing_str = ", ".join(missing)
        human = HumanMessage(content=f"""Conversation history:
{history}

User message: {user_message}

Missing information: {missing_str}

Ask the user for the missing information in a friendly, conversational way.
Keep it brief - one short question.""")

        response = _get_llm().invoke([system, human])
        updates["messages"] = [AIMessage(content=response.content)]
    else:
        # All details collected — confirm we're capturing the lead
        updates["messages"] = [
            AIMessage(
                content=f"Perfect! Let me lock that in for you right away, {name}!"
            )
        ]

    return updates


# ---------------------------------------------------------------------------
# Node 5: Capture Lead
# ---------------------------------------------------------------------------
def capture_lead_node(state: AgentState) -> dict:
    """Execute the lead capture and send a confirmation message."""
    name = state["user_name"]
    email = state["user_email"]
    platform = state["user_platform"]

    # Execute mock lead capture
    mock_lead_capture(name, email, platform)

    confirmation = (
        f"Great news, {name}! I've captured your details successfully.\n\n"
        f"Our team will reach out to you at {email} shortly.\n"
        f"We'll tailor our recommendations for your {platform} strategy.\n\n"
        f"In the meantime, you can start your 14-day free trial at autostream.io -- "
        f"no credit card required! Is there anything else I can help you with?"
    )

    return {
        "messages": [AIMessage(content=confirmation)],
        "lead_captured": True,
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------
def route_by_intent(state: AgentState) -> str:
    """Route to the appropriate handler based on detected intent."""
    intent = state.get("intent", "inquiry")
    if intent == "greeting":
        return "handle_greeting"
    elif intent == "high_intent":
        return "handle_high_intent"
    else:
        return "handle_inquiry"


def route_after_high_intent(state: AgentState) -> str:
    """Route to lead capture if all details are collected, else end turn."""
    if (
        state.get("user_name")
        and state.get("user_email")
        and state.get("user_platform")
        and not state.get("lead_captured")
    ):
        return "capture_lead"
    return END


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------
def build_graph():
    """Construct and compile the Inflx agent LangGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("handle_greeting", handle_greeting_node)
    graph.add_node("handle_inquiry", handle_inquiry_node)
    graph.add_node("handle_high_intent", handle_high_intent_node)
    graph.add_node("capture_lead", capture_lead_node)

    # Entry edge
    graph.add_edge(START, "detect_intent")

    # Route from intent detection
    graph.add_conditional_edges(
        "detect_intent",
        route_by_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_inquiry": "handle_inquiry",
            "handle_high_intent": "handle_high_intent",
        },
    )

    # Terminal edges for greeting and inquiry
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_inquiry", END)

    # Conditional routing after high-intent handling
    graph.add_conditional_edges(
        "handle_high_intent",
        route_after_high_intent,
        {
            "capture_lead": "capture_lead",
            END: END,
        },
    )

    # Lead capture always ends
    graph.add_edge("capture_lead", END)

    return graph.compile()

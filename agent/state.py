"""
Agent state schema for LangGraph.
Defines the typed state that flows through all graph nodes.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State schema for the Inflx (by AutoStream) sales agent graph.

    Attributes:
        messages: Conversation history (HumanMessage / AIMessage).
                  Uses add_messages reducer to append new messages.
        intent: The detected intent for the current user message.
        user_name: Collected user name (None if not yet provided).
        user_email: Collected user email (None if not yet provided).
        user_platform: Collected user platform (None if not yet provided).
        lead_captured: Whether the lead has been successfully captured.
    """
    messages: Annotated[list, add_messages]
    intent: str
    user_name: str | None
    user_email: str | None
    user_platform: str | None
    lead_captured: bool

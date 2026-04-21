"""
Intent detection module.
Uses GPT-4o-mini to classify user messages into intent categories.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# System prompt for intent classification
INTENT_SYSTEM_PROMPT = """You are an intent classifier for Inflx, an AI-powered video editing SaaS product by AutoStream.
Classify the user's message into exactly ONE of these categories:

1. "greeting" - The user is ONLY saying hello, hi, good morning, or making small talk with NO product question.
2. "inquiry" - The user is asking about pricing, features, plans, resolution, AI captions, policies, or any product-related question. This includes messages that start with a greeting but ALSO contain a product question.
3. "high_intent" - The user is expressing interest in buying, subscribing, signing up, trying the product, or providing their contact details (name, email, platform).

Rules:
- If the message contains BOTH a greeting AND a product question (e.g. "Hi, tell me about your pricing"), classify as "inquiry" NOT "greeting".
- If the user mentions wanting to buy, subscribe, sign up, try, get started, or says "I want the Pro/Basic plan", classify as "high_intent".
- If the user is providing their personal details (name, email, platform) in response to a request, classify as "high_intent".
- If the user asks "how much", "what does it cost", "what are the features", "what resolution", "do you have captions", classify as "inquiry".
- Only classify as "greeting" if there is NO product question at all.

Respond with ONLY the intent label: greeting, inquiry, or high_intent
Do NOT include any other text."""


def detect_intent(user_message: str) -> str:
    """
    Classify the user's message into an intent category.

    Args:
        user_message: The user's input text.

    Returns:
        One of: "greeting", "inquiry", "high_intent"
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    intent = response.content.strip().lower()

    # Validate the intent
    valid_intents = {"greeting", "inquiry", "high_intent"}
    if intent not in valid_intents:
        # Default to inquiry if classification is unclear
        intent = "inquiry"

    return intent

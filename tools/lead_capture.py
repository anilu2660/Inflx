"""
Lead capture tool.
Mock implementation that simulates saving a lead to a CRM/database.

Required function signature (from spec):
    def mock_lead_capture(name, email, platform):
        print(f"Lead captured successfully: {name}, {email}, {platform}")

The tool must NOT be triggered prematurely.
It is called ONLY after all three values (name, email, platform) are collected.
"""

from langchain_core.tools import tool

@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulate capturing a lead.

    This function should ONLY be called when all three fields
    (name, email, platform) have been collected from the user.

    Args:
        name: The user's name.
        email: The user's email address.
        platform: The creator platform (YouTube, Instagram, etc.)

    Returns:
        A confirmation message string.
    """
   
    return f"Lead captured successfully: {name}, {email}, {platform}"

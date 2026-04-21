"""
Loader module for the knowledge base.
Reads knowledge.json and converts entries into LangChain Document objects.
"""

import json
import os
from langchain_core.documents import Document


def load_knowledge_base() -> list[Document]:
    """
    Load the knowledge base from knowledge.json and return a list of
    LangChain Document objects ready for embedding.
    """
    knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge.json")

    with open(knowledge_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for entry in data:
        # Combine title and content for richer context
        page_content = f"{entry['title']}\n\n{entry['content']}"
        metadata = {
            "category": entry.get("category", "general"),
            "title": entry.get("title", ""),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents

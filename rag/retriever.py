"""
Retriever module for the RAG pipeline.
Builds a FAISS vector store from the knowledge base and provides a retriever.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rag.loader import load_knowledge_base


# Module-level cache to avoid rebuilding the index on every call
_retriever = None


def get_retriever(k: int = 3):
    """
    Build (or return cached) FAISS retriever from the knowledge base.

    Args:
        k: Number of top documents to retrieve.

    Returns:
        A LangChain retriever backed by FAISS.
    """
    global _retriever

    if _retriever is not None:
        return _retriever

    # Load documents from knowledge base
    documents = load_knowledge_base()

    # Create embeddings and build FAISS index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embeddings)

    # Create retriever
    _retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    return _retriever

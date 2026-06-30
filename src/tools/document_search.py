"""
Document search tool using RAG.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentSearchTool:
    """Tool for searching documents using RAG."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.description = "Search uploaded documents for relevant information"
    
    def execute(self, query: str) -> str:
        if not query.strip():
            return "Please provide a search query."
        if self.vector_store.get_count() == 0:
            return "No documents have been uploaded yet."
        results = self.vector_store.search(query, limit=5)
        if not results:
            return "No relevant documents found."
        formatted = []
        for i, r in enumerate(results, 1):
            content = r.get('content', '')[:500] + '...' if len(r.get('content', '')) > 500 else r.get('content', '')
            relevance = int((1 - r.get('distance', 0)) * 100) if r.get('distance') else 100
            formatted.append(f"📄 **Result {i}** (Relevance: {relevance}%)\n{content}")
        return "\n\n".join(formatted)
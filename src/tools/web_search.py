"""
Web Search Tool using DuckDuckGo.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class WebSearchTool:
    def __init__(self):
        self.description = "Search the web for latest information"
        self._ddgs = None
    
    def execute(self, query: str) -> str:
        if not query or not query.strip():
            return "Please provide a search query."
        try:
            from duckduckgo_search import DDGS
            self._ddgs = DDGS()
            results = list(self._ddgs.text(query, max_results=3))
            if not results:
                return f"🌐 No results found for '{query}'."
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get('title', 'No title')
                body = r.get('body', '')[:300] + '...' if len(r.get('body', '')) > 300 else r.get('body', '')
                href = r.get('href', 'N/A')
                formatted.append(f"📌 **{i}. {title}**\n{body}\n🔗 {href}")
            return "\n\n".join(formatted)
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"❌ Web search failed: {str(e)}"
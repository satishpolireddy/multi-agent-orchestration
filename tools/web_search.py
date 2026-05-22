"""WebSearchTool — DuckDuckGo search (no API key needed)."""
from __future__ import annotations
import logging
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool
logger = logging.getLogger(__name__)
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web via DuckDuckGo."
    _search: object = None
    def _run(self, query):
        if not self._search: object.__setattr__(self, "_search", DuckDuckGoSearchRun())
        try: return self._search.run(query) or "No results."
        except Exception as e: return f"Search error: {e}"
    async def _arun(self, query): return self._run(query)

"""SummarizerTool — LLM-powered text summarisation."""
from __future__ import annotations
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
logger = logging.getLogger(__name__)
PROMPT = "Summarise this text in 3-5 concise bullet points. Return ONLY the bullets."
class SummarizerTool(BaseTool):
    name: str = "summarizer"
    description: str = "Summarises long text using an LLM."
    llm: object = None
    def _run(self, text):
        if not self.llm: return "Error: no LLM."
        try: return self.llm.invoke([SystemMessage(content=PROMPT), HumanMessage(content=text[88000])]).content
        except Exception as e: return f"Error: {e}"
    async def _arun(self, t): return self._run(t)

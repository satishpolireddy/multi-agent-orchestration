"""
RAGTool — wraps RAGPipeline as a pluggable tool in the ToolRegistry.

The executor agent can call this tool when a step requires searching
over a pre-indexed knowledge base rather than the live web.

Usage in tool registry:
    from rag.rag_tool import RAGTool
    rag = RAGPipeline(llm=llm)
    rag.ingest_texts([...])
    registry.register(RAGTool(pipeline=rag))
"""

from __future__ import annotations

from typing import Any

from rag.pipeline import RAGPipeline


class RAGTool:
    """
    LangChain-compatible tool wrapper around RAGPipeline.

    Attributes
    ----------
    name:
        Tool identifier used by the ToolRegistry (``"rag"``).
    description:
        Human-readable description shown to the LLM planner.
    """

    name: str = "rag"
    description: str = (
        "Search a local knowledge base using Retrieval-Augmented Generation. "
        "Use this when the answer is likely in indexed documents rather than the web."
    )

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def run(self, query: str) -> str:
        """
        Query the RAG pipeline and return a synthesised answer.

        Parameters
        ----------
        query:
            The question or search string to run against the knowledge base.

        Returns
        -------
        String answer from the LLM, grounded in retrieved document chunks.
        """
        return self._pipeline.query(query)

    def retrieve_chunks(self, query: str) -> list[dict]:
        """Return raw retrieved chunks without LLM synthesis."""
        return self._pipeline.retrieve(query)

"""
RAGPipeline — Retrieval-Augmented Generation using LlamaIndex.

Provides:
  - Document ingestion from text, files, or URLs
  - Vector-store indexing (in-memory by default, swappable to persistent)
  - Similarity search with top-k retrieval
  - QA over the indexed corpus via query engine

Usage:
    rag = RAGPipeline(llm=llm)
    rag.ingest_texts(["LangGraph is a state machine library...", ...])
    answer = rag.query("What is LangGraph?")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Wraps LlamaIndex to provide document ingestion + retrieval QA.

    Parameters
    ----------
    llm:
        A LangChain LLM instance (ChatOpenAI, etc.).  Passed to LlamaIndex
        via its LangChain bridge so the same model is reused everywhere.
    embed_model:
        LlamaIndex embedding model string, e.g. "local" or "default".
        Defaults to OpenAI embeddings when OPENAI_API_KEY is present.
    similarity_top_k:
        Number of chunks to retrieve per query (default 4).
    chunk_size:
        Token size for document splitting (default 512).
    """

    def __init__(
        self,
        llm: Any,
        embed_model: str = "default",
        similarity_top_k: int = 4,
        chunk_size: int = 512,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        self.chunk_size = chunk_size
        self._index = None
        self._query_engine = None
        self._documents: list[Any] = []

        self._setup_llama_settings()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_llama_settings(self) -> None:
        """Configure LlamaIndex global Settings to use our LLM."""
        try:
            from llama_index.core import Settings
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.llms.langchain import LangChainLLM

            Settings.llm = LangChainLLM(llm=self.llm)
            Settings.node_parser = SentenceSplitter(chunk_size=self.chunk_size)
            if self.embed_model != "default":
                Settings.embed_model = self.embed_model
            logger.info("[RAG] LlamaIndex settings configured.")
        except ImportError as exc:
            logger.warning("[RAG] LlamaIndex not fully installed: %s", exc)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        """
        Index a list of raw text strings.

        Parameters
        ----------
        texts:
            List of plain-text documents to index.
        metadata:
            Optional list of metadata dicts (same length as texts).
        """
        from llama_index.core import Document, VectorStoreIndex

        metadata = metadata or [{} for _ in texts]
        docs = [
            Document(text=t, metadata=m)
            for t, m in zip(texts, metadata)
        ]
        self._documents.extend(docs)
        self._rebuild_index(self._documents)
        logger.info("[RAG] Ingested %d text documents. Total: %d", len(texts), len(self._documents))

    def ingest_files(self, paths: list[str | Path]) -> None:
        """
        Index documents from local file paths (.txt, .pdf, .md supported).

        Parameters
        ----------
        paths:
            List of file paths to read and index.
        """
        from llama_index.core import SimpleDirectoryReader

        path_strs = [str(p) for p in paths]
        reader = SimpleDirectoryReader(input_files=path_strs)
        docs = reader.load_data()
        self._documents.extend(docs)
        self._rebuild_index(self._documents)
        logger.info("[RAG] Ingested %d file(s). Total docs: %d", len(paths), len(self._documents))

    def ingest_directory(self, directory: str | Path) -> None:
        """
        Recursively index all supported files in a directory.

        Parameters
        ----------
        directory:
            Path to a folder; all .txt / .md / .pdf files are indexed.
        """
        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(input_dir=str(directory), recursive=True)
        docs = reader.load_data()
        self._documents.extend(docs)
        self._rebuild_index(self._documents)
        logger.info("[RAG] Ingested directory '%s'. Total docs: %d", directory, len(self._documents))

    def _rebuild_index(self, documents: list[Any]) -> None:
        """Rebuild the VectorStoreIndex from all documents."""
        from llama_index.core import VectorStoreIndex

        self._index = VectorStoreIndex.from_documents(documents)
        self._query_engine = self._index.as_query_engine(
            similarity_top_k=self.similarity_top_k
        )
        logger.debug("[RAG] Index rebuilt with %d documents.", len(documents))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict]:
        """
        Return the top-k most relevant chunks for a query (no LLM call).

        Returns
        -------
        List of dicts with keys: ``text``, ``score``, ``metadata``.
        """
        if self._index is None:
            logger.warning("[RAG] Index is empty — call ingest_* first.")
            return []

        retriever = self._index.as_retriever(similarity_top_k=self.similarity_top_k)
        nodes = retriever.retrieve(query)
        return [
            {
                "text": n.node.get_content(),
                "score": round(n.score, 4) if n.score else None,
                "metadata": n.node.metadata,
            }
            for n in nodes
        ]

    def query(self, question: str) -> str:
        """
        Answer a question using RAG: retrieve relevant chunks then generate.

        Parameters
        ----------
        question:
            Natural-language question to answer from the indexed corpus.

        Returns
        -------
        String answer synthesised by the LLM from retrieved context.
        """
        if self._query_engine is None:
            logger.warning("[RAG] No documents indexed yet. Returning empty answer.")
            return "No documents have been indexed. Please ingest documents first."

        logger.info("[RAG] Querying: %s", question[:100])
        response = self._query_engine.query(question)
        return str(response)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all indexed documents and reset the pipeline."""
        self._documents = []
        self._index = None
        self._query_engine = None
        logger.info("[RAG] Index cleared.")

    @property
    def document_count(self) -> int:
        """Number of documents currently indexed."""
        return len(self._documents)

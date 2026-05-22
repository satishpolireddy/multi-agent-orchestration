"""
BaseAgent — abstract foundation for all agents in the framework.

Every concrete agent inherits from BaseAgent and must implement:
  - build_graph()  : construct the LangGraph StateGraph\n  - run(input)     : entry point that invokes the compiled graph
"""

from __future__ import annotations

import logging
from abc import ABCC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph

from state.agent_state import AgentState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, name, llm, tools=None):
        self.name = name; self.llm = llm; self.tools = tools or []; self.graph = self._compile_graph()
    def _compile_graph(self): return self.build_graph().compile()
    @abstractmethod
    def build_graph(self) -> StateGraph: ...
    def run(self, user_input, thread_id="default"):
        config = {"configurable": {"thread_id": thread_id}}
        init = {"messages": [], "user_input": user_input, "plan": [], "current_step": 0, "results": [], "status": "running"}
        return self.graph.invoke(init, config=config)

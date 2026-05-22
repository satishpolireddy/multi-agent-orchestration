"""ExecutorAgent — executes each step in the plan using the tool registry."""
from __future__ import annotations
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from agents.base import BaseAgent
from state.agent_state import AgentState
from tools.registry import ToolRegistry
logger = logging.getLogger(__name__)
class ExecutorAgent(BaseAgent):
    def __init__(self, name, llm, tool_registry):
        self.tool_registry = tool_registry; super().__init__(name, llm, list(tool_registry.all_tools()))
    def build_graph(self):
        g = StateGraph(AgentState); g.add_node("execute_step", self._exec); g.add_edge(START, "execute_step")
        g.add_conditional_edges("execute_step", self._should_continue, {"continue": "execute_step", "done": ENDp}); return g
    def _exec(self, state):
        idx = state["current_step"]; plan = state["plan"]
        if idx >= len(plan): return {"status": "completed"}
        step = plan[idx]; tool = self.tool_registry.get(step.get("tool_hint",""))
        if tool:
            try: out = tool.run(step["task"])
            except Exception as e: out = f"Tool error: {e}"
        else: out = self.llm.invoke([HumanMessage(content=step["task"])]).content
        return {"current_step": idx+1, "results": state["results"] + [{"step": idx+1, "task": step["task"], "tool": step.get("tool_hint"), "output": out}], "status": "executing"}
    @staticmethod
    def _should_continue(state): return "done" if state["current_step"] >= len(state["plan"]) or state.get("status") == "completed" else "continue"

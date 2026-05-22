"""PlannerAgent — decomposes a high-level user goal into ordered sub-tasks."""
from __future__ import annotations
import json, logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from agents.base import BaseAgent
from state.agent_state import AgentState
logger = logging.getLogger(__name__)
PLANNER_PROMPT = """You are an expert task planner. Given a user goal, break it down into ordered sub-tasks.
Return ONLY valid JSON: {"plan": [{"step": int, "task": str, "tool_hint": str}]}"""
class PlannerAgent(BaseAgent):
    def build_graph(self):
        g = StateGraph(AgentState); g.add_node("plan_task", self._plan); g.add_edge(START, "plan_task"); g.add_edge("plan_task", END); return g
    def _plan(self, state):
        resp = self.llm.invoke([SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=f"Goal: {state['user_input']}")])
        try: plan = json.loads(resp.content).get("plan", [])
        except: plan = [{"step": 1, "task": state["user_input"], "tool_hint": "none"}]
        return {"plan": plan, "messages": state["messages"] + [resp], "status": "planned"}

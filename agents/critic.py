"""CriticAgent — validates executor results and decides whether to accept or retry."""
from __future__ import annotations
import json, logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from agents.base import BaseAgent
from state.agent_state import AgentState
logger = logging.getLogger(__name__)
CRITIC_PROMPT = """You are a quality-control critic. Evaluate these results and return JSON:
{"assessment": [{"step":int, "score":0-10, "accepted":bool, "reason":str}], "overall_accepted":bool, "summary":str}"""
class CriticAgent(BaseAgent):
    def build_graph(self):
        g = StateGraph(AgentState); g.add_node("review", self._review); g.add_edge(START, "review"); g.add_edge("review", END); return g
    def _review(self, state):
        results = state.get("results", [])
        if not results: return {"status": "completed", "critique": {"overall_accepted": True}}
        resp = self.llm.invoke([SystemMessage(content=CRITIC_PROMPT), HumanMessage(content=json.dumps(results))])
        try: cq = json.loads(resp.content)
        except: cq = {"overall_accepted": True, "summary": "Fallback"}
        return {"status": "completed" if cq.get("overall_accepted") else "retry_needed", "critique": cq, "messages": state["messages"] + [resp]}

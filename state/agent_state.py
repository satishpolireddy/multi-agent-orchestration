"""
AgentState — the shared state schema passed between all LangGraph nodes.

Every node receives this dict and returns a partial update to it.
LangGraph merges the updates automatically using reducer functions.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """
    Shared state for all agents in the orchestration framework.

    Fields:
        messages     : Conversation history (LangChain BaseMessage list).
        user_input   : The raw natural-language task from the user.
        plan         : List of step dicts produced by PlannerAgent.
                       Each step: {"step": int, "task": str, "tool_hint": str}
        current_step : Index into `plan` — tracks ExecutorAgent progress.
        results      : Accumulated results from each executed step.
                       Each result: {"step": int, "task": str, "tool": str, "output": Any}
        status       : Lifecycle marker.
                       Values: "running" | "planned" | "executing" | "completed" | "retry_needed" | "failed"
        critique     : Quality assessment produced by CriticAgent.
        metadata     : Arbitrary key-value pairs for extensibility.
    """

    messages: list[BaseMessage]
    user_input: str
    plan: list[dict[str, Any]]
    current_step: int
    results: list[dict[str, Any]]
    status: str
    critique: dict[str, Any]
    metadata: dict[str, Any]

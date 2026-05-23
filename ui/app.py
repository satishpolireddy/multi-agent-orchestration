"""
Streamlit Demo UI for the Multi-Agent Orchestration Framework.

Connects to the FastAPI backend (or runs the pipeline directly if
API_BASE_URL is not set) and renders the Planner → Executor → Critic
pipeline results in a clean, interactive web interface.

Run standalone:
    streamlit run ui/app.py

Run against Docker backend:
    API_BASE_URL=http://localhost:8000/api/v1 streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
PAGE_TITLE = "Multi-Agent Orchestration Framework"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def api_run(task: str, thread_id: str | None = None) -> dict:
    """Call POST /run and return the full result dict."""
    payload: dict[str, Any] = {"task": task}
    if thread_id:
        payload["thread_id"] = thread_id
    resp = requests.post(f"{API_BASE}/run", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def api_history() -> list[dict]:
    """Fetch task history from GET /history."""
    try:
        resp = requests.get(f"{API_BASE}/history", timeout=10)
        resp.raise_for_status()
        return resp.json().get("tasks", [])
    except Exception:
        return []

def run_direct(task: str, thread_id: str) -> dict:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_openai import ChatOpenAI
    from agents.planner import PlannerAgent
    from agents.executor import ExecutorAgent
    from agents.critic import CriticAgent
    from tools.registry import ToolRegistry
    from tools.calculator import CalculatorTool
    from tools.database import DatabaseTool
    from tools.web_search import WebSearchTool
    from tools.summarizer import SummarizerTool
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DatabaseTool())
    registry.register(WebSearchTool())
    registry.register(SummarizerTool(llm=llm))
    planner = PlannerAgent(name="planner", llm=llm)
    executor = ExecutorAgent(name="executor", llm=llm, tool_registry=registry)
    critic = CriticAgent(name="critic", llm=llm)
    planned = planner.run(task, thread_id=thread_id)
    executed = executor.graph.invoke({**planned, "current_step": 0, "results": [], "status": "running"})
    final = critic.graph.invoke({**executed})
    return {"status": final.get("status", "completed"), "plan": planned.get("plan", []), "results": executed.get("results", []), "critique": final.get("critique", {})}


with st.sidebar:
    st.title("⚙️ Settings")
    use_api = st.toggle("Use API backend", value=True, help=f"Connect to {API_BASE}")
    thread_id = st.text_input("Thread ID (optional)", placeholder="my-session-1")
    st.divider()
    st.subheader("📜 Task History")
    if use_api:
        history = api_history()
        if history:
            for h in history[-8:]:
                status_emoji = "✅" if h["status"] == "completed" else ("❌" if h["status"] == "failed" else "⏳")
                st.caption(f"{status_emoji} {h['task'][:50]}…")
        else:
            st.caption("No history yet.")
    else:
        st.caption("History available via API mode.")
    st.divider()
    st.caption("Multi-Agent Orchestration Framework")
    st.caption("Planner → Executor → Critic")

st.title("🤖 Multi-Agent Orchestration Framework")
st.markdown("Submit a task and watch the **Planner**, **Executor**, and **Critic** agents collaborate to solve it step-by-step.")

st.subheader("💡 Try an example")
cols = st.columns(3)
examples = [
    "Research the top 3 benefits of using LangGraph for AI agents",
    "Calculate compound interest: $10,000 at 7% for 20 years",
    "Summarise the key differences between RAG and fine-tuning",
]
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state["task_input"] = example

st.divider()
task = st.text_area("Enter your task", value=st.session_state.get("task_input", ""), height=100, placeholder="e.g. Research the latest trends in vector databases and summarise the top 3", key="task_input")
run_btn = st.button("🚀 Run", type="primary", disabled=not task.strip())

if run_btn and task.strip():
    tid = thread_id.strip() or None
    with st.spinner("Running pipeline…"):
        start = time.time()
        error = None
        result = None
        try:
            if use_api:
                result = api_run(task.strip(), tid)
            else:
                result = run_direct(task.strip(), tid or "streamlit-session")
        except Exception as exc:
            error = str(exc)
        elapsed = time.time() - start
    if error:
        st.error(f"❌ Pipeline failed: {error}")
    else:
        st.success(f"✅ Completed in {elapsed:.1f}s — Status: **{result.get('status', 'unknown').upper()}**")
        st.subheader("🧠 Execution Plan")
        plan = result.get("plan", [])
        if plan:
            for step in plan:
                tool_badge = f"`{step.get('tool_hint', 'none')}`"
                st.markdown(f"**Step {step['step']}** {tool_badge}  \n{step['task']}")
        else:
            st.caption("No plan returned.")
        st.divider()
        st.subheader("⚙️ Execution Results")
        results = result.get("results", [])
        if results:
            for r in results:
                with st.expander(f"Step {r.get('step', '?')} — {r.get('task', '')[:60]}", expanded=True):
                    st.caption(f"Tool: `{r.get('tool', 'none')}`")
                    output = r.get("output", "")
                    if isinstance(output, (dict, list)):
                        st.json(output)
                    else:
                        st.write(str(output))
        else:
            st.caption("No results returned.")
        st.divider()
        st.subheader("🔍 Quality Review")
        critique = result.get("critique", {})
        if critique:
            overall = critique.get("overall_accepted", True)
            verdict_color = "green" if overall else "red"
            verdict_label = "ACCEPTED ✅" if overall else "NEEDS RETRY ❌"
            st.markdown(f"**Overall verdict:** :{verdict_color}[{verdict_label}]")
            st.write(critique.get("summary", ""))
            assessments = critique.get("assessment", [])
            if assessments:
                for a in assessments:
                    score = a.get("score", 0)
                    bar_color = "normal" if score >= 7 else ("off" if score < 5 else "normal")
                    st.progress(score / 10, text=f"Step {a['step']} — Score: {score}/10 — {a.get('reason', '')}")
        else:
            st.caption("No critique returned.")
        st.divider()
        with st.expander("📄 Raw JSON response"):
            st.json(result)

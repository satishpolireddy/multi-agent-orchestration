"""
main.py — CLI entry point for the Multi-Agent Orchestration Framework.

Usage:
    python main.py --task "Research recent breakthroughs in quantum computing"
    python main.py --task "Calculate compound interest at 5% for 10 years on $10000" --thread my-session
    python main.py --list-tools

Environment variables (set in .env):
    OPENAI_API_KEY   — required for LLM calls
    MODEL_NAME       — default: gpt-4o-mini
    LOG_LEVEL        — default: INFO
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_llm():
    """Initialise the LLM from environment config."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Copy .env.example → .env and add your key.")
        sys.exit(1)

    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0, api_key=api_key)


def build_tool_registry(llm):
    """Register all available tools."""
    from tools.registry import ToolRegistry
    from tools.calculator import CalculatorTool
    from tools.database import DatabaseTool
    from tools.web_search import WebSearchTool
    from tools.summarizer import SummarizerTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DatabaseTool())
    registry.register(WebSearchTool())
    registry.register(SummarizerTool(llm=llm))
    return registry


def run_pipeline(task: str, thread_id: str):
    """Run the full Planner → Executor → Critic pipeline."""
    from agents.planner import PlannerAgent
    from agents.executor import ExecutorAgent
    from agents.critic import CriticAgent

    llm = build_llm()
    registry = build_tool_registry(llm)

    print(f"\n{'='*60}")
    print(f"  Task: {task}")
    print(f"  Thread: {thread_id}")
    print(f"{'='*60}\n")

    # Stage 1: Plan
    print("🧠 [1/3] Planning...")
    planner = PlannerAgent(name="planner", llm=llm)
    planned = planner.run(task, thread_id=thread_id)
    print(f"  → {len(planned['plan'])} steps generated\n")
    for step in planned["plan"]:
        print(f"  Step {step['step']}: {step['task']} [tool: {step.get('tool_hint','?')}]")

    # Stage 2: Execute
    print(f"\n⚙️  [2/3] Executing {len(planned['plan'])} steps...")
    executor = ExecutorAgent(name="executor", llm=llm, tool_registry=registry)
    # Inject the plan into initial state
    executed = executor.graph.invoke({**planned, "current_step": 0, "results": [], "status": "running"})
    for r in executed.get("results", []):
        print(f"\n  ✅ Step {r['step']} [{r['tool']}]:\n     {str(r['output'])[:200]}")

    # Stage 3: Critique
    print("\n🔍 [3/3] Reviewing quality...")
    critic = CriticAgent(name="critic", llm=llm)
    final = critic.graph.invoke({**executed})
    critique = final.get("critique", {})
    print(f"\n  Overall accepted: {critique.get('overall_accepted', '?')}")
    print(f"  Summary: {critique.get('summary', 'N/A')}")

    print(f"\n{'='*60}")
    print(f"  Status: {final.get('status', 'unknown').upper()}")
    print(f"{'='*60}\n")
    return final


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Orchestration Framework")
    parser.add_argument("--task", type=str, help="The task for the agent system to perform")
    parser.add_argument("--thread", type=str, default="default", help="Session thread ID for state persistence")
    parser.add_argument("--list-tools", action="store_true", help="List all registered tools and exit")
    args = parser.parse_args()

    if args.list_tools:
        llm = build_llm()
        registry = build_tool_registry(llm)
        print("Registered tools:")
        for name in registry.list_names():
            print(f"  • {name}")
        return

    if not args.task:
        parser.print_help()
        sys.exit(1)

    run_pipeline(args.task, args.thread)


if __name__ == "__main__":
    main()

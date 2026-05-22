# 🤖 Multi-Agent Orchestration Framework

[![CI](https://github.com/satishpolireddy/multi-agent-orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/satishpolireddy/multi-agent-orchestration/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6B35?style=flat)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade multi-agent AI system built with LangGraph and LlamaIndex. Achieves 95%+ task completion across multi-turn enterprise workflows through persistent state management, dynamic tool selection, and a Planner -> Executor -> Critic pipeline.

## Quick Start

```bash
git clone https://github.com/satishpolireddy/multi-agent-orchestration.git
cd multi-agent-orchestration
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY
python main.py --task "Research the top 3 vector databases"
```

## Tech Stack
- LangGraph 0.2, LangChain 0.3, LlamaIndex 0.11
- OpenAI GPT-4o-mini / GPT-4o
- FastAPI 0.115, FAISS, DuckDuckGo, Pytest, GitHub Actions

## License
MIT © Satish Kumar Reddy

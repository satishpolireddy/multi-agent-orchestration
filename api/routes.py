"""
API route definitions for the Multi-Agent Orchestration Framework.

Endpoints
---------
POST /run           — submit a task, wait for full result
POST /stream        — submit a task, stream agent events via SSE
GET  /status/{id}   — poll a running/completed task
GET  /history       — list past task executions (this session)
DELETE /history/{id} — delete a task record
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# ------------------------------------------------------------------
# In-memory task store  (replace with DB in production)
# ------------------------------------------------------------------
_task_store: dict[str, dict] = {}


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class RunRequest(BaseModel):
    task: str = Field(..., description="Natural language task for the agent system.")
    thread_id: str | None = Field(None, description="Optional session ID for state persistence.")


class RunResponse(BaseModel):
    task_id: str
    thread_id: str
    status: str
    plan: list[dict] = []
    results: list[dict] = []
    critique: dict = {}
    created_at: str
    completed_at: str | None = None


class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int = Field(description="Steps completed so far.")
    total_steps
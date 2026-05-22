"""API routes — stubs for Days 9-11."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
router = APIRouter()
@router.post("/run")
async def run_task(body: dict): return JSONResponse(status_code=501, content={"detail": "Coming in Days 9-11."})
@router.post("/stream")
async def stream_task(body: dict): return JSONResponse(status_code=501, content={"detail": "Coming in Days 9-11."})
@router.get("/status/{task_id}")
async def get_status(task_id: str): return JSONResponse(status_code=501, content={"detail": "Coming in Days 9-11."})
@router.get("/history")
async def get_history(): return JSONResponse(status_code=501, content={"detail": "Coming in Days 9-11."})

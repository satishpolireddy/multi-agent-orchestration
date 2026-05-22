"""FastAPI application for the Multi-Agent Orchestration Framework."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
app = FastAPI(title="Multi-Agent Orchestration", version="0.1.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")
@app.get("/health")
async def health(): return {"status": "ok"}

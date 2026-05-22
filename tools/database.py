"""DatabaseTool — read-only SQLite query runner."""
from __future__ import annotations
import logging, sqlite3
from pathlib import Path
from langchain_core.tools import BaseTool
logger = logging.getLogger(__name__)
class DatabaseTool(BaseTool):
    name: str = "database"
    description: str = "Run read-only SELECT sql queries against SQLite."
    db_path: str = "data/agent_memory.db"
    def _run(self, query):
        q = query.strip()
        if any(q.lower().startswith(k) for k in ["insert","update","delete","drop","alter","create"]): return "Error: Only SELECT queries allowed."
        if not Path(self.db_path).exists(): return f"DB not found: {self.db_path}"
        try:
            conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
            rows = conn.execute(q).fetchmany(50); conn.close()
            if not rows: return "No results."
            h = rows[0].keys(); return "\n".join([" | ".join(h)] + [" | ".join(str(r[k]) for k in h) for r in rows])
        except sqlite3.Error as e: return f"SQL error: {e}"
    async def _arun(self, q): return self._run(q)

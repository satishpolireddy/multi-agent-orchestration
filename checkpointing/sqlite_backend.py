"""
SQLiteCheckpointer — persists and restores AgentState to/from SQLite.

Each agent run is identified by a (thread_id, checkpoint_id) pair.
This lets users resume long-running tasks across process restarts.

Schema
------
Table: checkpoints
  thread_id     TEXT  — session / conversation identifier
  checkpoint_id TEXT  — monotonic step counter or UUID
  created_at    TEXT  — ISO-8601 timestamp
  state_json    TEXT  — full AgentState serialised as JSON

Usage:
    cp = SQLiteCheckpointer("data/checkpoints.db")
    cp.save(thread_id="run-1", checkpoint_id="step-3", state=my_state)
    state = cp.load(thread_id="run-1", checkpoint_id="step-3")
    latest = cp.load_latest(thread_id="run-1")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from state.agent_state import AgentState

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id     TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    state_json    TEXT NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_id)
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_thread_created
    ON checkpoints (thread_id, created_at DESC);
"""


class SQLiteCheckpointer:
    """
    Lightweight SQLite backend for AgentState persistence.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created automatically if it
        does not exist.  Use ``:memory:`` for ephemeral in-process storage.
    """

    def __init__(self, db_path: str | Path = "data/checkpoints.db"):
        self.db_path = str(db_path)
        self._ensure_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Create the database file and schema if they don't exist."""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
        logger.debug("[Checkpoint] Database ready at '%s'.", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        thread_id: str,
        state: AgentState,
        checkpoint_id: str | None = None,
    ) -> str:
        """
        Persist an AgentState snapshot.

        Parameters
        ----------
        thread_id:
            Identifies the session / run.
        state:
            The full AgentState dict to persist.
        checkpoint_id:
            Optional explicit ID.  If omitted, an ISO timestamp is used.

        Returns
        -------
        The checkpoint_id used for storage.
        """
        if checkpoint_id is None:
            checkpoint_id = datetime.now(timezone.utc).isoformat()

        created_at = datetime.now(timezone.utc).isoformat()

        # Serialise — messages contain LangChain objects, handle gracefully
        try:
            state_json = json.dumps(_serialise_state(state), ensure_ascii=False)
        except TypeError as exc:
            logger.error("[Checkpoint] Serialisation error: %s", exc)
            raise

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                    (thread_id, checkpoint_id, created_at, state_json)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, checkpoint_id, created_at, state_json),
            )

        logger.info(
            "[Checkpoint] Saved thread='%s' checkpoint='%s'.",
            thread_id,
            checkpoint_id,
        )
        return checkpoint_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, thread_id: str, checkpoint_id: str) -> AgentState | None:
        """
        Load a specific checkpoint.

        Returns
        -------
        AgentState dict or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM checkpoints WHERE thread_id=? AND checkpoint_id=?",
                (thread_id, checkpoint_id),
            ).fetchone()

        if row is None:
            logger.warning(
                "[Checkpoint] Not found: thread='%s' checkpoint='%s'.",
                thread_id,
                checkpoint_id,
            )
            return None

        return json.loads(row["state_json"])

    def load_latest(self, thread_id: str) -> AgentState | None:
        """
        Load the most recent checkpoint for a thread.

        Returns
        -------
        AgentState dict or None if the thread has no checkpoints.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT state_json FROM checkpoints
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()

        if row is None:
            logger.info("[Checkpoint] No checkpoints found for thread='%s'.", thread_id)
            return None

        logger.info("[Checkpoint] Loaded latest for thread='%s'.", thread_id)
        return json.loads(row["state_json"])

    # ------------------------------------------------------------------
    # Listing / housekeeping
    # ------------------------------------------------------------------

    def list_checkpoints(self, thread_id: str) -> list[dict]:
        """
        Return metadata for all checkpoints in a thread, newest first.

        Returns
        -------
        List of dicts with keys: ``thread_id``, ``checkpoint_id``, ``created_at``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT thread_id, checkpoint_id, created_at
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY created_at DESC
                """,
                (thread_id,),
            ).fetchall()

        return [dict(r) for r in rows]

    def list_threads(self) -> list[str]:
        """Return all distinct thread IDs stored in the database."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
        return [r["thread_id"] for r in rows]

    def delete_thread(self, thread_id: str) -> int:
        """
        Delete all checkpoints for a thread.

        Returns
        -------
        Number of rows deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
            )
        logger.info(
            "[Checkpoint] Deleted %d checkpoint(s) for thread='%s'.",
            cursor.rowcount,
            thread_id,
        )
        return cursor.rowcount

    def prune_old(self, keep_last_n: int = 10, thread_id: str | None = None) -> int:
        """
        Remove old checkpoints, keeping only the most recent ``keep_last_n`` per thread.

        Parameters
        ----------
        keep_last_n:
            How many checkpoints to retain per thread.
        thread_id:
            If provided, prune only that thread; otherwise prune all threads.

        Returns
        -------
        Total number of rows deleted.
        """
        threads = [thread_id] if thread_id else self.list_threads()
        total_deleted = 0

        for tid in threads:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM checkpoints
                    WHERE thread_id = ?
                      AND checkpoint_id NOT IN (
                          SELECT checkpoint_id FROM checkpoints
                          WHERE thread_id = ?
                          ORDER BY created_at DESC
                          LIMIT ?
                      )
                    """,
                    (tid, tid, keep_last_n),
                )
                total_deleted += cursor.rowcount

        logger.info("[Checkpoint] Pruned %d old checkpoint(s).", total_deleted)
        return total_deleted


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _serialise_state(state: dict) -> dict:
    """
    Convert an AgentState to a JSON-serialisable dict.

    LangChain message objects are converted to plain dicts; everything
    else falls back to ``str()`` if not natively serialisable.
    """
    out: dict[str, Any] = {}
    for key, value in state.items():
        if key == "messages":
            out[key] = [_serialise_message(m) for m in (value or [])]
        else:
            try:
                json.dumps(value)  # quick check
                out[key] = value
            except (TypeError, ValueError):
                out[key] = str(value)
    return out


def _serialise_message(msg: Any) -> dict:
    """Convert a LangChain message to a plain dict."""
    if isinstance(msg, dict):
        return msg
    return {
        "type": getattr(msg, "type", type(msg).__name__),
        "content": getattr(msg, "content", str(msg)),
    }

"""LangGraph checkpoint configuration."""

from __future__ import annotations

from app.core.config import settings


def get_checkpointer():
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except Exception:  # noqa: BLE001 - optional dependency
        return None

    conn_string = settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
    try:
        return PostgresSaver.from_conn_string(conn_string)
    except Exception:  # noqa: BLE001 - optional dependency
        return None

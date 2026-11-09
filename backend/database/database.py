"""
SQLite database setup and session management using aiosqlite.
"""
import aiosqlite
import logging
from pathlib import Path

from backend.config import DATABASE_PATH

logger = logging.getLogger(__name__)

_DB_PATH = DATABASE_PATH


async def get_db() -> aiosqlite.Connection:
    """Get an async SQLite connection."""
    db = await aiosqlite.connect(_DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    """Initialize the database and create tables if they don't exist."""
    db = await aiosqlite.connect(_DB_PATH)
    try:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                ann_genre TEXT,
                cnn_genre TEXT,
                ann_confidence REAL,
                cnn_confidence REAL,
                ann_top3 TEXT,
                cnn_top3 TEXT,
                audio_duration REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
        logger.info(f"Database initialized at {_DB_PATH}")
    finally:
        await db.close()


async def close_db():
    """Cleanup hook (currently a no-op since we create connections per-request)."""
    pass

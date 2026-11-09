"""
Database service for managing prediction history in SQLite.
"""
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

from backend.database.database import get_db

logger = logging.getLogger("history_service")


async def save_prediction_record(
    filename: str,
    ann_prediction: Optional[Dict],
    cnn_prediction: Optional[Dict],
    audio_duration: float,
) -> int:
    """Save an inference prediction record into the SQLite database."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    ann_genre = ann_prediction.get("predicted_genre") if ann_prediction else None
    ann_conf = ann_prediction.get("confidence") if ann_prediction else None
    ann_top3 = json.dumps(ann_prediction.get("top3", [])) if ann_prediction else None

    cnn_genre = cnn_prediction.get("predicted_genre") if cnn_prediction else None
    cnn_conf = cnn_prediction.get("confidence") if cnn_prediction else None
    cnn_top3 = json.dumps(cnn_prediction.get("top3", [])) if cnn_prediction else None

    db = await get_db()
    try:
        cursor = await db.execute(
            """
            INSERT INTO predictions (
                filename, timestamp, ann_genre, cnn_genre,
                ann_confidence, cnn_confidence, ann_top3, cnn_top3,
                audio_duration
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename, timestamp, ann_genre, cnn_genre,
                ann_conf, cnn_conf, ann_top3, cnn_top3,
                audio_duration
            )
        )
        await db.commit()
        record_id = cursor.lastrowid
        return record_id
    finally:
        await db.close()


async def get_prediction_history(limit: int = 50) -> List[Dict]:
    """Retrieve recent prediction history records."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT id, filename, timestamp, ann_genre, cnn_genre,
                   ann_confidence, cnn_confidence, audio_duration
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def get_prediction_by_id(record_id: int) -> Optional[Dict]:
    """Retrieve detailed prediction record by ID."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT * FROM predictions WHERE id = ?
            """,
            (record_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        
        record = dict(row)
        if record.get("ann_top3"):
            record["ann_top3"] = json.loads(record["ann_top3"])
        if record.get("cnn_top3"):
            record["cnn_top3"] = json.loads(record["cnn_top3"])
        return record
    finally:
        await db.close()


async def delete_prediction_by_id(record_id: int) -> bool:
    """Delete a prediction record by ID."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "DELETE FROM predictions WHERE id = ?",
            (record_id,)
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()

"""
API routes for prediction history management.
"""
from typing import List
from fastapi import APIRouter, HTTPException, Path
from backend.database.models import PredictionHistoryItem, PredictionHistoryDetail
from backend.services.history_service import (
    get_prediction_history,
    get_prediction_by_id,
    delete_prediction_by_id,
)

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=List[PredictionHistoryItem])
async def list_history(limit: int = 50):
    """Retrieve previous prediction records."""
    records = await get_prediction_history(limit=limit)
    return records


@router.get("/{id}", response_model=PredictionHistoryDetail)
async def get_history_detail(id: int = Path(..., description="Prediction record ID")):
    """Retrieve individual prediction details with top-3 probabilities."""
    record = await get_prediction_by_id(id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Prediction record {id} not found.")
    return record


@router.delete("/{id}")
async def delete_history_item(id: int = Path(..., description="Prediction record ID")):
    """Delete an individual prediction record."""
    deleted = await delete_prediction_by_id(id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Prediction record {id} not found.")
    return {"status": "success", "message": f"Record {id} deleted successfully."}

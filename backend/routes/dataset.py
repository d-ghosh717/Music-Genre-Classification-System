"""
API routes for dataset validation and statistics.
"""
from fastapi import APIRouter
from backend.database.models import DatasetStatusResponse
from ml.dataset_validator import validate_dataset

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


@router.get("/status", response_model=DatasetStatusResponse)
async def get_dataset_status():
    """
    Returns GTZAN dataset availability, genre counts, and file integrity status.
    """
    report = validate_dataset(check_audio=False, verbose=False)
    return report.to_dict()

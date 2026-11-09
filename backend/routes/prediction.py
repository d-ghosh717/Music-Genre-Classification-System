"""
API routes for audio prediction and model status.
Handles uploaded audio files safely with temporary file cleanup.
"""
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.config import ALLOWED_AUDIO_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES
from backend.database.models import ModelStatusResponse, PredictionResponse
from backend.services.history_service import save_prediction_record
from ml.inference.predictor import get_predictor

logger = logging.getLogger("prediction_routes")
router = APIRouter(prefix="/api", tags=["Prediction"])


@router.get("/models/status", response_model=ModelStatusResponse)
async def get_models_status():
    """Returns availability and load state of ANN and CNN models."""
    predictor = get_predictor()
    return predictor.get_models_status()


@router.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file (WAV or MP3), verifies format and size,
    extracts features / Mel-spectrograms, and classifies into genres
    using both ANN and CNN models.
    """
    # 1. Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    # 2. Safe temporary file handling
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
        tmp_path = tmp_file.name
        try:
            bytes_read = 0
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                bytes_read += len(chunk)
                if bytes_read > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB."
                    )
                tmp_file.write(chunk)
            tmp_file.flush()
        except HTTPException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")

    # 3. Predict audio using both models
    try:
        predictor = get_predictor()
        results = predictor.predict_audio(tmp_path)
        # Ensure original uploaded filename is returned
        results["filename"] = file.filename or Path(tmp_path).name

        # 4. Save to history database (only if at least one model produced predictions)
        if results.get("ann_prediction") or results.get("cnn_prediction"):
            try:
                await save_prediction_record(
                    filename=results["filename"],
                    ann_prediction=results.get("ann_prediction"),
                    cnn_prediction=results.get("cnn_prediction"),
                    audio_duration=results.get("audio_duration", 0.0),
                )
            except Exception as e:
                logger.warning(f"Could not persist prediction to history database: {e}")

        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Guaranteed cleanup of temporary file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove temp file {tmp_path}: {cleanup_err}")

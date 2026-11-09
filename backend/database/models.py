"""
Pydantic models for API request/response schemas.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    ann_model_available: bool
    cnn_model_available: bool
    dataset_available: bool
    model_details: Optional[Dict[str, Any]] = None


class DatasetStatusResponse(BaseModel):
    exists: bool
    genres_found: List[str]
    genres_missing: List[str]
    genre_counts: Dict[str, int]
    total_files: int
    total_valid: int
    total_invalid: int
    is_valid: bool
    summary: str


class ModelStatusResponse(BaseModel):
    ann_available: bool
    cnn_available: bool
    ann_model_path: Optional[str] = None
    cnn_model_path: Optional[str] = None


class GenreProbability(BaseModel):
    genre: str
    probability: float


class UncertaintyInfo(BaseModel):
    is_uncertain: bool
    entropy: float
    margin: float
    reason: str


class ModelPrediction(BaseModel):
    predicted_genre: str
    confidence: float
    top3: List[GenreProbability]
    model_type: str
    uncertainty: Optional[UncertaintyInfo] = None


class PredictionResponse(BaseModel):
    ann_prediction: Optional[ModelPrediction] = None
    cnn_prediction: Optional[ModelPrediction] = None
    models_agree: Optional[bool] = None
    audio_duration: float
    filename: str
    ann_error: Optional[str] = None
    cnn_error: Optional[str] = None
    uncertainty_status: Optional[str] = None
    out_of_taxonomy_warning: Optional[str] = None


class PredictionHistoryItem(BaseModel):
    id: int
    filename: str
    timestamp: str
    ann_genre: Optional[str] = None
    cnn_genre: Optional[str] = None
    ann_confidence: Optional[float] = None
    cnn_confidence: Optional[float] = None
    audio_duration: Optional[float] = None


class PredictionHistoryDetail(PredictionHistoryItem):
    ann_top3: Optional[List[GenreProbability]] = None
    cnn_top3: Optional[List[GenreProbability]] = None


class EvaluationMetrics(BaseModel):
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    test_loss: float
    classification_report: Dict
    confusion_matrix: List[List[int]]
    genres: List[str]


class EvaluationResponse(BaseModel):
    ann: Optional[EvaluationMetrics] = None
    cnn: Optional[EvaluationMetrics] = None
    available: bool


class ErrorResponse(BaseModel):
    detail: str

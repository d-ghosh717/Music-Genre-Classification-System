"""
Integration tests for FastAPI backend REST endpoints.
"""
import io
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.database import init_db
import asyncio

@pytest.fixture(autouse=True, scope="module")
def setup_database():
    asyncio.run(init_db())

client = TestClient(app)


@pytest.fixture
def test_audio_file():
    """Create a temporary valid wav file for API testing."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    sf.write(path, y, sr)
    yield path
    Path(path).unlink(missing_ok=True)


def test_health_endpoint():
    """Verify health endpoint response structure and loaded model audit details."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ann_model_available" in data
    assert "cnn_model_available" in data
    assert "dataset_available" in data
    assert "model_details" in data
    assert data["model_details"]["active_version"] == "Phase 8 Decontaminated Benchmark"
    assert data["model_details"]["ann"]["is_decontaminated"] is True
    assert data["model_details"]["cnn"]["is_decontaminated"] is True
    assert data["model_details"]["ann"]["scaler_features"] == 75


def test_dataset_status_endpoint():
    """Verify dataset status endpoint reports GTZAN information."""
    response = client.get("/api/dataset/status")
    assert response.status_code == 200
    data = response.json()
    assert "exists" in data
    assert "genres_found" in data
    assert "genre_counts" in data
    assert len(data["genres_found"]) == 10


def test_models_status_endpoint():
    """Verify model status endpoint."""
    response = client.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "ann_available" in data
    assert "cnn_available" in data


def test_prediction_history_endpoints():
    """Verify listing history."""
    response = client.get("/api/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_visualization_path_traversal_prevention():
    """Verify endpoint rejects path traversal attacks."""
    response = client.get("/api/visualizations/ann/../../etc/passwd")
    assert response.status_code in (400, 404, 422)


def test_predict_invalid_extension():
    """Verify endpoint rejects non-audio file formats."""
    fake_file = io.BytesIO(b"not an audio file")
    response = client.post(
        "/api/predict",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400


def test_evaluation_endpoint():
    """Verify evaluation endpoint returns metrics structure, Phase 7 diagnostic reports, and Phase 8 clean benchmark."""
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert "ann" in data
    assert "cnn" in data
    assert "baseline_vs_improved" in data
    assert "external_evaluation" in data
    assert "final_benchmark" in data
    if data["final_benchmark"]:
        assert "final_ann" in data["final_benchmark"]
        assert "final_cnn" in data["final_benchmark"]
        assert "decontamination_summary" in data["final_benchmark"]


def test_final_benchmark_visualizations():
    """Verify Phase 8 final benchmark evolution plot is served."""
    response = client.get("/api/visualizations/final/benchmark_evolution_comparison.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_baseline_vs_improved_visualization():
    """Verify baseline vs improved comparison plot is served."""
    response = client.get("/api/visualizations/comparison/baseline_vs_improved.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_predict_valid_audio_and_history_lifecycle(test_audio_file):
    """Verify prediction on audio file, and check history retrieval and deletion."""
    with open(test_audio_file, "rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("test_tone.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    pred_data = response.json()
    assert pred_data["filename"] == "test_tone.wav"
    assert "audio_duration" in pred_data
    assert "uncertainty_status" in pred_data
    if pred_data["ann_prediction"]:
        assert "uncertainty" in pred_data["ann_prediction"]
    if pred_data["cnn_prediction"]:
        assert "uncertainty" in pred_data["cnn_prediction"]

    # Check history contains the record
    hist_resp = client.get("/api/history")
    assert hist_resp.status_code == 200
    history_items = hist_resp.json()
    assert len(history_items) > 0

    latest_id = history_items[0]["id"]

    # Check get detail
    detail_resp = client.get(f"/api/history/{latest_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == latest_id

    # Check delete record
    del_resp = client.delete(f"/api/history/{latest_id}")
    assert del_resp.status_code == 200

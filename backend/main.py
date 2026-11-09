"""
FastAPI application entry point for the Music Genre Classification System.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import FRONTEND_URL, BACKEND_HOST, BACKEND_PORT
from backend.database.database import init_db
from backend.database.models import HealthResponse
from backend.routes.dataset import router as dataset_router
from backend.routes.evaluation import router as evaluation_router
from backend.routes.history import router as history_router
from backend.routes.prediction import router as prediction_router
from ml.config import dataset_config
from ml.inference.predictor import get_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Initializing SQLite history database...")
    await init_db()
    
    # Preload ML predictor
    logger.info("Checking ML models availability...")
    predictor = get_predictor()
    logger.info(f"Models status: {predictor.get_models_status()}")
    
    yield
    logger.info("Shutting down backend server.")


app = FastAPI(
    title="Music Genre Classification API",
    description="Deep Learning Based Audio Genre Classification using ANN and CNN architectures on the GTZAN dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
origins = [
    FRONTEND_URL,
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register sub-routers
app.include_router(dataset_router)
app.include_router(prediction_router)
app.include_router(evaluation_router)
app.include_router(history_router)


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns application health, model availability, dataset status, and loaded checkpoint versions."""
    predictor = get_predictor()
    dataset_exists = Path(dataset_config.dataset_dir).exists()
    
    return {
        "status": "healthy",
        "ann_model_available": predictor.is_ann_available(),
        "cnn_model_available": predictor.is_cnn_available(),
        "dataset_available": dataset_exists,
        "model_details": predictor.get_model_details(),
    }


# Optional static mounting for built frontend distribution
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from fastapi import HTTPException

    if (frontend_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)

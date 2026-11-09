"""
Backend configuration.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
UPLOAD_DIR = PROJECT_ROOT / "tmp" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Backend server
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# CORS
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# File upload limits
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".au"}

# Database
DATABASE_PATH = str(PROJECT_ROOT / "backend" / "predictions.db")

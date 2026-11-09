#!/usr/bin/env bash
# Quickstart script for Music Genre Classification System
set -e

echo "=========================================================="
echo " Starting Music Genre Classification System"
echo "=========================================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Ensure SQLite database is initialized
python -c "import asyncio; from backend.database.database import init_db; asyncio.run(init_db())"

echo "Starting FastAPI Backend on http://localhost:8000 ..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting React Frontend on http://localhost:5173 ..."
cd frontend
npm run dev -- --host &
FRONTEND_PID=$!
cd ..

cleanup() {
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo ""
echo "System online:"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to terminate."

wait

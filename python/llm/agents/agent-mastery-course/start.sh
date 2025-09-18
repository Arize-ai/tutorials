#!/bin/bash

# Trip Planner Startup Script

echo "🚀 Starting AI Trip Planner..."

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the trip_planner directory"
    exit 1
fi

echo "📡 Starting backend server..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

echo ""
echo "✅ Service started successfully!"
echo "📡 Backend API: http://localhost:8000"
echo "🖥️  Minimal UI served at / (frontend/index.html)"
echo "📊 Arize Traces: https://app.arize.com/"
echo ""
echo "Press Ctrl+C to stop the service"

# Function to cleanup when script is interrupted
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for services
wait

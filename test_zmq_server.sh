#!/bin/bash

# Script to test the ZeroMQ FAISS Proxy server

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID
    fi
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Check for Python
if ! command -v python &> /dev/null; then
    echo "Python not found. Please install Python 3.8+"
    exit 1
fi

# Check for ZeroMQ
python -c "import zmq" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ZeroMQ Python bindings not found. Installing..."
    pip install pyzmq
fi

# Check for msgpack
python -c "import msgpack" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "msgpack not found. Installing..."
    pip install msgpack
fi

# Check for FAISS
python -c "import faiss" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "FAISS not found. Installing..."
    pip install faiss-cpu
fi

# Check for NumPy
python -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "NumPy not found. Installing..."
    pip install numpy
fi

# Create data directory if it doesn't exist
mkdir -p ./data

# Start the server
echo "Starting ZeroMQ FAISS Proxy server..."
cd server
python run.py &
SERVER_PID=$!
cd ..

# Wait for server to start
echo "Waiting for server to start..."
sleep 2

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server failed to start!"
    exit 1
fi

echo "Server running with PID: $SERVER_PID"

# Run the test client
echo "Running test client..."
cd server
python test_zmq_client.py
cd ..

# Ask user if they want to keep the server running
read -p "Tests completed. Keep server running? (y/n): " keep_running

if [ "$keep_running" != "y" ]; then
    cleanup
else
    echo "Server is still running with PID: $SERVER_PID"
    echo "Press Ctrl+C to stop the server when done."
    # Wait for user to press Ctrl+C
    wait $SERVER_PID
fi

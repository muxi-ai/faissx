#!/bin/bash
# Run tests for the FAISSx client library

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if server is running, if not, start it in background
if ! nc -z localhost 45678 &>/dev/null; then
    echo "Starting FAISSx server in the background..."
    python -m faissx.server run --enable-auth --auth-keys "test-key-1:tenant-1" &
    SERVER_PID=$!

    # Give server time to start
    sleep 2

    echo "Server started with PID $SERVER_PID"
    KILL_SERVER=true
else
    echo "FAISSx server already running"
    KILL_SERVER=false
fi

# Install test requirements
pip install -r tests/requirements.txt

# Run the tests
echo "Running client tests..."
python -m unittest discover -s tests

# If we started the server, shut it down
if [ "$KILL_SERVER" = true ]; then
    echo "Shutting down FAISSx server (PID $SERVER_PID)..."
    kill $SERVER_PID
fi

echo "Tests completed!"

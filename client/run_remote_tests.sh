#!/bin/bash
#
# Run FAISSx server and remote tests together
#
# Copyright (C) 2025 Ran Aroussi

# Set up error handling
set -e

# Configuration
SERVER_PORT=45679  # Changed to avoid conflicts with any existing server
TEST_SCRIPT="test_remote_no_fallback.py"
SERVER_MODULE="../faissx/server/server.py"  # Updated to use the server module directly
LOG_FILE="server.log"

echo "===== Running FAISSx Remote Mode Tests ====="

# Check if port is in use
if netstat -tuln | grep -q ":$SERVER_PORT "; then
    echo "Error: Port $SERVER_PORT is already in use."
    echo "Stop any existing FAISSx server before running this test."
    exit 1
fi

# Move to script directory
cd "$(dirname "$0")"

# Start server in background
echo "Starting FAISSx server on port $SERVER_PORT..."
python "$SERVER_MODULE" --port "$SERVER_PORT" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "Server started with PID $SERVER_PID"

# Wait for server to initialize
echo "Waiting for server to start..."
sleep 3

# Check if server started successfully
if ! ps -p $SERVER_PID > /dev/null; then
    echo "Error: Server failed to start. Check $LOG_FILE for details."
    cat "$LOG_FILE"
    exit 1
fi

echo "Server is running. Starting tests..."

# Modify the test script to use the correct port
export FAISSX_TEST_PORT="$SERVER_PORT"

# Run the tests with the correct server port
PYTHONPATH=.. python -c "
import os
import sys
sys.path.insert(0, '..')
import faissx
test_port = os.environ.get('FAISSX_TEST_PORT', '45679')
faissx.configure(url=f'tcp://localhost:{test_port}')
# Import the module correctly
sys.path.append('.')
from test_remote_no_fallback import run_all_tests
run_all_tests()
"

TEST_RESULT=$?

# Stop the server
echo "Tests completed. Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo "Server stopped."

# Print test results and exit with appropriate code
if [ $TEST_RESULT -eq 0 ]; then
    echo "All tests passed! ✓"
else
    echo "Tests failed! ✗"
    echo "Check the output above for details."
fi

exit $TEST_RESULT

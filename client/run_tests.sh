#!/bin/bash
# Run tests for the FAISSx client

set -e

# Go to client directory
cd "$(dirname "$0")"

# Set environment variables for the tests
export FAISSX_SERVER=tcp://localhost:45678
export FAISSX_API_KEY=test-key-1
export FAISSX_TENANT_ID=tenant-1

# Run the tests
python -m unittest discover -s tests -p "test_*.py"

echo "All tests passed!"

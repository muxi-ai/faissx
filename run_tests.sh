#!/bin/bash
set -e

echo "==== Running server tests ===="
cd server
pip install -r tests/requirements.txt
pip install -e .
pytest tests/

echo "==== Running client tests ===="
cd ../client
pip install -r tests/requirements.txt
pip install -e .
pytest tests/test_unit/

# Run integration tests only if explicitly requested
if [ "$1" == "--integration" ]; then
  echo "==== Running integration tests ===="
  pytest tests/test_api/
else
  echo "==== Skipping integration tests (use --integration to run them) ===="
fi

echo "All tests completed successfully!"

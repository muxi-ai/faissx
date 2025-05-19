#!/bin/bash

# Run the ZeroMQ FAISS Proxy tests

# Set up Python path to include the parent directory
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Run the test client
echo "Running ZeroMQ test client..."
python -m server.tests.test_zmq.test_zmq_client

# Run pytest tests
echo "Running automated ZeroMQ tests..."
pytest -xvs server/tests/test_zmq

echo "ZeroMQ tests complete"

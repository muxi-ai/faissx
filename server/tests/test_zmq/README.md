# ZeroMQ FAISS Proxy Tests

This directory contains tests for the ZeroMQ implementation of the FAISS Proxy server.

## Overview

The tests in this directory verify the functionality of the ZeroMQ-based FAISS Proxy server, including:

- Creating indices
- Adding vectors
- Searching for similar vectors
- Retrieving index information

## Test Client

The `test_zmq_client.py` file contains a simple client that exercises the basic operations of the FAISS Proxy server. It's useful for manual testing and verification of the server functionality.

## Running Tests

You can run the tests using the `run_zmq_tests.sh` script in the server directory:

```bash
cd server
./run_zmq_tests.sh
```

Or run them directly with pytest:

```bash
# Make sure you're in the project root directory
pytest -xvs server/tests/test_zmq
```

## Standalone Test Client

You can also run the test client separately:

```bash
# From the project root
python -m server.tests.test_zmq.test_zmq_client

# Or using the convenience script
cd server
./test_client.py
```

This is useful for quick functional testing when making changes to the server implementation.

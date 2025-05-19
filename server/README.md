# FAISSx Server

This directory contains tools and utilities for the FAISSx Server.

## Structure

The main server code is in the `faissx/server` Python package. This directory contains:

- `test_client.py`: A simple client for testing server functionality
- `run_zmq_tests.sh`: Script to run all ZeroMQ-related tests

## Running Tests

To run the server tests:

```bash
./run_zmq_tests.sh
```

## Installation

The server is installed as part of the `faissx` package:

```bash
pip install -e .
```

## Running the Server

```bash
# Using the CLI
faissx.server run

# With custom configuration
faissx.server run --port 45678 --enable-auth --auth-keys "key1:tenant1,key2:tenant2"
```

See the documentation in `faissx/server/__init__.py` for more configuration options.

# FAISSx Server

This directory contains tools and utilities for the FAISSx Server.

## Structure

The main server code is in the `faissx/server` Python package. This directory contains:

- `Dockerfile`: Docker configuration for the server
- `build_docker.sh`: Script to build the Docker image
- `pytest.ini`: Configuration for tests

## Running Tests

To run the server tests:

```bash
# From the project root
pytest faissx/server/tests
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

## Docker Deployment

```bash
# Build the Docker image
./build_docker.sh

# Run with default settings
docker run -p 45678:45678 muxi/faissx:latest

# Run with persistent data
docker run -p 45678:45678 -v /path/to/data:/data muxi/faissx:latest
```

# FAISSx (FAISS Extended)

> High-performance vector database proxy built with FAISS and ZeroMQ.

## Run FAISS as a service for multi-server deployments

## Overview

FAISSx provides a lightweight, efficient interface to create, manage, and search vector indices using Facebook AI Similarity Search (FAISS). It uses ZeroMQ for high-performance binary communication, making it significantly faster than HTTP-based alternatives.

## Features

- Zero-copy binary messaging protocol with ZeroMQ and msgpack serialization
- Create and manage multiple vector indices
- Add, search, and manage vectors efficiently
- Multi-tenant support with API key authentication
- Python package with server and client components
- Docker-based deployment
- Optimized for production workloads

## Installation

```bash
# Install from PyPI (when published)
pip install faissx

# For development
git clone https://github.com/muxi-ai/faissx.git
cd faissx
pip install -e .
```

## Quick Start

### Setting up the server

#### Option 1: Using the Python API

```python
from faissx import server

server.configure(
    port=45678,  # default is 45678
    bind_address="0.0.0.0",  # default is "0.0.0.0"
    data_dir="/data",  # if omitted, faissx it will use in-memory indices
    auth_keys={"test-key-1": "tenant-1", "test-key-2": "tenant-2"},  # default is empty dict
    enable_auth=True,  # default is False
)

# Alternative: load API keys from a JSON file
# server.configure(
#     port=45678,
#     bind_address="0.0.0.0",
#     auth_file="path/to/auth.json",  # JSON file with API keys mapping
#     enable_auth=True,
# )

server.run()
```

#### Option 2: Using the CLI

After installing the package via pip, you can use the command-line interface:

```bash
# Start the server with default settings
faissx.server run

# Start with custom options
faissx.server run --port 45678 --data-dir ./data --enable-auth --auth-keys "key1:tenant1,key2:tenant2"

# Using authentication file instead of inline keys
faissx.server run --enable-auth --auth-file path/to/auth.json

# Show help
faissx.server run --help

# Show version
faissx.server --version
```

Note: For authentication, you can provide API keys either inline with `--auth-keys` or from a JSON file with `--auth-file`. The JSON file should have the format `{"api_key1": "tenant1", "api_key2": "tenant2"}`. Only one authentication method can be used at a time.

#### Option 3: Using Docker

```bash
# Pull and run the pre-built image
docker run -p 45678:45678 -v ./data:/data muxi/faissx:latest

# Or build and run using docker-compose
git clone https://github.com/muxi-ai/faissx.git
cd faissx
docker-compose up
```

### Using the client

```python
from faissx import client as faiss
import numpy as np

# Configure the client with authentication
faiss.configure(
    server="tcp://localhost:45678",  # ZeroMQ server address
    api_key="test-key-1",            # API key for authentication
    tenant_id="tenant-1"             # Tenant ID for multi-tenant isolation
)

# Use it like regular FAISS - drop-in replacement
index = faiss.IndexFlatL2(128)  # Create a new index with dimension 128
vectors = np.random.rand(100, 128).astype(np.float32)  # Generate random vectors
index.add(vectors)  # Add vectors to the index
D, I = index.search(np.random.rand(1, 128).astype(np.float32), k=5)  # Search for similar vectors
```

## Docker Deployment

We provide Docker images for easy deployment:

```bash
# Run with default settings
docker run -p 45678:45678 muxi/faissx:latest

# Run with persistent data and authentication
docker run -p 45678:45678 \
  -v /path/to/data:/data \
  -v /path/to/auth.json:/auth.json \
  -e FAISSX_DATA_DIR=/data \
  -e FAISSX_AUTH_FILE=/auth.json \
  -e FAISSX_ENABLE_AUTH=true \
  muxi/faissx:latest
```

## Documentation

- [Server Documentation](server/README.md): Detailed information about the server component
- [Client Documentation](client/README.md): Detailed information about the client library
- [Next Steps](NEXT_STEPS.md): Roadmap and upcoming features

## Performance

The ZeroMQ-based implementation provides significant performance improvements over HTTP-based alternatives:

- Binary protocol minimizes serialization overhead
- Persistent connections reduce latency
- Efficient vector operations through direct numpy integration
- No JSON encoding/decoding overhead for large vector data

## Development

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/muxi-ai/faissx.git
cd faissx

# Install in development mode with all dependencies
pip install -e .

# Run tests
pytest

# Run examples
python examples/server_example.py
```

### Running Client Tests

To run tests for the client component:

```bash
cd client
./run_tests.sh
```

### Docker Development

To build the Docker images:

```bash
cd server
./build_docker.sh
```

## Project Structure

```
/faissx       - Python package source code
  /server     - Server implementation
  /client     - Client library implementation
/server       - Server utilities, docker configs, tests
/client       - Client utilities and tests
/examples     - Example code for both client and server
/data         - Default directory for FAISS data files
```

## License

Apache 2.0

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
pip install faissx
```

## Quick Start

### Setting up the server

#### Option 1: Using the Python API

```python
from faissx import server

server.configure(
    port=45678,  # default is 45678
    bind_address="0.0.0.0",  # default is "0.0.0.0"
    # data_dir is omitted, so it will use in-memory indices
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
docker run -p 45678:45678 -v ./data:/data muxi/faissx:latest
```

### Using the client

```python
from faissx import client as faiss

faiss.configure(
    server="tcp://faiss-service:45678",
    api_key="test-key-1",
    tenant_id="tenant-1"
)

# use the client as a drop-in replacement for FAISS

index = faiss.IndexFlatL2(128)
index.add(np.random.rand(100, 128).astype(np.float32))
D, I = index.search(np.random.rand(1, 128).astype(np.float32), k=5)
```

## Docker Deployment

We provide two Docker images:

1. Standard Python-based container:

```bash
docker run -p 45678:45678 -v /path/to/data:/data muxi/faissx:latest
```

2. PyPy-based container for potential performance improvements:

```bash
docker run -p 45678:45678 -v /path/to/data:/data muxi/faissx:pypy
```

## Documentation

See the [server documentation](server/README.md) for detailed information about the server.

## Performance

The ZeroMQ-based implementation provides significant performance improvements over HTTP-based alternatives:

- Binary protocol minimizes serialization overhead
- Persistent connections reduce latency
- Efficient vector operations through direct numpy integration

## Development

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/muxi-ai/faissx.git
cd faissx

# Install in development mode
pip install -e .
pip install -r dev-requirements.txt

# Run examples
python examples/server_example.py
```

### Docker Development

To build the Docker images:

```bash
cd server
./build_docker.sh
```

## License

Apache 2.0

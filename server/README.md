# FAISSx Server

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/muxi-ai/faissx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A high-performance vector database service built on FAISS and ZeroMQ.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Protocol](#api-protocol)
- [Security](#security)
- [Docker Deployment](#docker-deployment)
- [Performance](#performance)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

The FAISSx server provides a high-performance vector database service with the following features:

- **ZeroMQ Communication**: Efficient binary protocol using ZeroMQ and msgpack
- **Multi-tenant Support**: Isolated data storage per tenant
- **Authentication**: API key-based authentication
- **Persistence**: Optional persistent storage for indices
- **FAISS Integration**: Full FAISS functionality exposed through a network API

## Architecture

The server follows a modular architecture:

```
faissx/server/
├── __init__.py      # Configuration and server setup
├── auth.py          # Authentication handling
├── cli.py           # Command-line interface
├── faiss_core.py    # FAISS integration and index management
├── protocol.py      # Communication protocol implementation
└── server.py        # Main server implementation
```

### Core Components

1. **Server**: Handles client connections and routes requests
2. **Protocol**: Manages serialization/deserialization of messages
3. **FAISS Core**: Manages FAISS indices and vector operations
4. **Authentication**: Validates API keys and manages tenant isolation

## Installation

### From PyPI

```bash
# Install the complete package
pip install faissx

# For server-only installation
pip install faissx[server]
```

### From Source

```bash
# Clone the repository
git clone https://github.com/muxi-ai/faissx.git
cd faissx

# Install in development mode
pip install -e .
```

## Configuration

The server can be configured programmatically or via command-line arguments.

### Programmatic Configuration

```python
from faissx import server

server.configure(
    port=45678,                      # Port to listen on
    bind_address="0.0.0.0",         # Address to bind to
    data_dir="/path/to/data",       # Directory for persistent storage
    auth_keys={"key1": "tenant1"},  # API keys and their tenant IDs
    enable_auth=True,               # Enable authentication
)

server.run()
```

### Command-line Configuration

```bash
# Basic configuration
faissx.server run --port 45678 --data-dir ./data

# With authentication
faissx.server run --enable-auth --auth-keys "key1:tenant1,key2:tenant2"

# Using an auth file
faissx.server run --enable-auth --auth-file /path/to/auth.json
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `port` | 45678 | Port to listen on |
| `bind_address` | "0.0.0.0" | Network interface to bind to |
| `data_dir` | None | Directory for persistent storage (None = in-memory only) |
| `auth_keys` | {} | Dictionary mapping API keys to tenant IDs |
| `auth_file` | None | Path to JSON file containing API keys |
| `enable_auth` | False | Whether to enable authentication |

### Environment Variables

- `FAISSX_DATA_DIR`: Directory for persistent storage
- `FAISSX_PORT`: Port to listen on
- `FAISSX_ENABLE_AUTH`: Whether to enable authentication (0/1)

## Running the Server

### Basic Usage

```bash
# Start with default settings (in-memory, no auth)
faissx.server run

# Start with persistence
faissx.server run --data-dir ./data

# Start with authentication
faissx.server run --enable-auth --auth-keys "key1:tenant1,key2:tenant2"
```

### Server Modes

#### In-Memory Mode

When run without a `data_dir`, the server operates in in-memory mode. All indices are stored in RAM and are lost when the server is restarted.

```bash
faissx.server run
```

#### Persistent Mode

With a `data_dir` specified, the server persists indices to disk for durability across restarts.

```bash
faissx.server run --data-dir ./data
```

## API Protocol

The FAISSx server uses a binary protocol based on ZeroMQ and msgpack. The protocol is request-response based with the following structure:

### Request Format

```python
{
    "action": "action_name",      # Required: Operation to perform
    "api_key": "your_api_key",    # Optional: Authentication key
    "tenant_id": "your_tenant",   # Optional: Tenant identifier
    # Action-specific parameters
}
```

### Response Format

```python
{
    "success": True/False,         # Operation success status
    "error": "error_message",      # Present only on failure
    # Action-specific results
}
```

### Supported Actions

| Action | Description | Parameters | Response |
|--------|-------------|------------|----------|
| `ping` | Check server availability | None | `{"success": true}` |
| `create_index` | Create a new index | `index_id`, `dimension`, `index_type` | `{"success": true, "index_id": "..."}` |
| `add_vectors` | Add vectors to an index | `index_id`, `vectors` | `{"success": true, "count": n}` |
| `search` | Search for similar vectors | `index_id`, `query_vectors`, `k` | `{"success": true, "distances": [...], "indices": [...]}` |
| `get_index_stats` | Get index statistics | `index_id` | `{"success": true, "dimension": n, "count": m, ...}` |
| `list_indexes` | List available indices | None | `{"success": true, "indexes": [...]}` |

## Security

### Authentication

The server supports API key-based authentication for multi-tenant deployments.

#### Enabling Authentication

```bash
# From command line
faissx.server run --enable-auth --auth-keys "key1:tenant1,key2:tenant2"

# Using an auth file
faissx.server run --enable-auth --auth-file /path/to/auth.json
```

#### Auth File Format

```json
{
    "api_key_1": "tenant_id_1",
    "api_key_2": "tenant_id_2"
}
```

### Tenant Isolation

Each tenant has isolated storage, preventing cross-tenant data access. All vector operations are scoped to the tenant's namespace.

## Docker Deployment

### Building the Docker Image

```bash
# From the server directory
./build_docker.sh

# Or manually
docker build -t muxi/faissx:latest .
```

### Running with Docker

```bash
# Basic run
docker run -p 45678:45678 muxi/faissx:latest

# With persistent storage
docker run -p 45678:45678 -v /path/to/data:/data \
  -e FAISSX_DATA_DIR=/data muxi/faissx:latest

# With authentication
docker run -p 45678:45678 \
  -v /path/to/auth.json:/auth.json \
  -e FAISSX_ENABLE_AUTH=1 \
  -e FAISSX_AUTH_FILE=/auth.json \
  muxi/faissx:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3'
services:
  faissx:
    image: muxi/faissx:latest
    ports:
      - "45678:45678"
    volumes:
      - ./data:/data
      - ./auth.json:/auth.json
    environment:
      - FAISSX_DATA_DIR=/data
      - FAISSX_ENABLE_AUTH=1
      - FAISSX_AUTH_FILE=/auth.json
```

## Performance

The FAISSx server is designed for high performance:

- **Binary Protocol**: Efficient data serialization with msgpack
- **ZeroMQ**: Low-latency, persistent connections
- **Native FAISS**: Direct integration with FAISS for optimized vector operations
- **Memory Mapping**: Efficient handling of large indices when using persistent storage

### Tuning for Performance

- Use in-memory mode for highest performance
- Configure FAISS indices appropriate for your use case (flat indices for small datasets, IVF/HNSW for larger datasets)
- Run the server on machines with sufficient memory for your indices
- Consider GPU acceleration for large-scale deployments

## Testing

### Running Tests

```bash
# From the project root
pytest faissx/server/tests

# With coverage
pytest faissx/server/tests --cov=faissx.server
```

### Load Testing

For load testing and benchmarking, use the included benchmark scripts:

```bash
python benchmarks/server_benchmark.py
```

## Troubleshooting

### Common Issues

#### Connection Refused

```
ConnectionRefused: Connection refused when connecting to server
```

- Check that the server is running
- Verify the port is not blocked by a firewall
- Ensure the server is binding to the correct network interface

#### Authentication Errors

```
RuntimeError: Authentication failed: Invalid API key
```

- Ensure you're using a valid API key
- Check that authentication is enabled on the server
- Verify the tenant ID matches the API key configuration

#### Memory Issues

```
MemoryError: Unable to allocate memory for index
```

- Reduce the index size or use a machine with more memory
- Consider using disk-based indices with the `data_dir` option
- Use more efficient index types (IVF/HNSW instead of flat indices)

### Logging

To enable detailed logging:

```bash
# Set logging level to DEBUG
faissx.server run --log-level DEBUG

# Log to a file
faissx.server run --log-file /path/to/server.log
```

## License

FAISSx is licensed under the [Apache 2.0 license](../LICENSE).

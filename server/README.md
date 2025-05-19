# FAISS Proxy Server: ZeroMQ Edition

A high-performance vector database proxy server using ZeroMQ for ultra-low latency communication.

## Features

- High-performance binary protocol using ZeroMQ
- Efficient vector serialization with numpy arrays
- Metadata handling with msgpack
- Multi-tenant support with API key authentication
- Persistent connections for reduced latency
- Deployable as Docker container or standalone Python package
- Drop-in FAISS API compatibility (when used with the client library)

## Architecture

The server uses a simple and efficient architecture:

- **ZeroMQ REP Socket**: Handles request/reply pattern with clients
- **Protocol Layer**: Serializes/deserializes messages with numpy and msgpack
- **FAISS Manager**: Core vector operations and index management
- **Authentication**: API key validation and tenant isolation

## Installation

### Option 1: PyPI Package (Recommended)

```bash
# Install the package
pip install faiss-proxy

# Run the server with Python API
python -c "from faiss_proxy import server; server.configure(port=5555); server.run()"
```

### Option 2: Docker Container

```bash
# Pull the image
docker pull muxi/faiss-proxy:latest

# Run with default configuration
docker run -p 5555:5555 -v /path/to/data:/data muxi/faiss-proxy:latest

# Run with custom configuration
docker run -p 5555:5555 \
  -v /path/to/data:/data \
  -e FAISS_DATA_DIR=/data \
  -e FAISS_API_KEYS=key1:tenant1,key2:tenant2 \
  muxi/faiss-proxy:latest
```

## Configuration Options

The server can be configured using environment variables or the Python API:

| Variable            | Python Config     | Description                           | Default          |
|---------------------|-------------------|---------------------------------------|------------------|
| FAISS_DATA_DIR      | data_dir          | Directory to store FAISS indexes      | None (in-memory) |
| FAISS_PORT          | port              | Port to listen on                     | 45678            |
| FAISS_BIND_ADDRESS  | bind_address      | Address to bind to                    | 0.0.0.0          |
| FAISS_AUTH_KEYS     | auth_keys         | API key:tenant mapping                | {} (no keys)     |
| FAISS_AUTH_FILE     | auth_file         | Path to JSON file with API keys       | None             |
| FAISS_ENABLE_AUTH   | enable_auth       | Enable authentication                 | False            |

> Note: You can provide API keys either inline with `auth_keys` or from a JSON file with `auth_file`.
> The JSON file should have the format `{"api_key1": "tenant1", "api_key2": "tenant2"}`.
> Only one authentication method can be used at a time.

### Python API Configuration

```python
from faiss_proxy import server

# Configure the server with inline API keys
server.configure(
    port=45678,
    bind_address="0.0.0.0",
    data_dir="/path/to/data",  # omit for in-memory indices
    auth_keys={"key1": "tenant1", "key2": "tenant2"},
    enable_auth=True
)

# Alternatively, load API keys from a JSON file
# server.configure(
#     port=45678,
#     bind_address="0.0.0.0",
#     auth_file="/path/to/auth_keys.json",
#     enable_auth=True
# )

# Run the server
server.run()
```

### Docker Container Configuration

```bash
# Pull the image
docker pull muxi/faiss-proxy:latest

# Run with default configuration
docker run -p 45678:45678 muxi/faiss-proxy:latest

# Run with custom configuration
docker run -p 45678:45678 \
  -v /path/to/data:/data \
  -v /path/to/auth.json:/auth.json \
  -e FAISS_PORT=45678 \
  -e FAISS_BIND_ADDRESS=0.0.0.0 \
  -e FAISS_DATA_DIR=/data \
  -e FAISS_AUTH_KEYS=key1:tenant1,key2:tenant2 \
  -e FAISS_AUTH_FILE=/auth.json \
  -e FAISS_ENABLE_AUTH=true \
  muxi/faiss-proxy:latest

# Note: You should use either FAISS_AUTH_KEYS or FAISS_AUTH_FILE, not both
```

## Protocol

The ZeroMQ message protocol uses a binary format:

1. **Header**: Msgpack-encoded dictionary with operation, authentication, and parameters
2. **Vector Data**: Raw numpy array bytes (present for add and search operations)
3. **Metadata**: Msgpack-encoded structured data (IDs, filters, etc.)

### Supported Operations

- `create_index`: Create a new FAISS index
- `add_vectors`: Add vectors to an index
- `search`: Search for similar vectors
- `delete_vector`: Delete a vector from an index
- `delete_index`: Delete an index
- `get_index_stats`: Get index metadata
- `list_indexes`: List all available indexes
- `ping`: Check server availability

## Client Integration

To connect to the server, use the FAISS Proxy client library for a drop-in replacement for FAISS:

```python
import faiss_proxy as faiss

# Configure the client
faiss.configure(zmq_url="tcp://localhost:5555", api_key="your-key")

# Use like normal FAISS
index = faiss.IndexFlatL2(128)
index.add(vectors)
D, I = index.search(query_vectors, k=10)
```

## Docker Files

The repository includes two Dockerfile options:

1. `Dockerfile` - Standard Python-based container
2. `Dockerfile.pypy` - PyPy-based container for potential performance improvements

## Development

For development purposes, dependencies are specified in `dev-requirements.txt`.

```bash
# Install development dependencies
pip install -r server/dev-requirements.txt

# Run tests
pytest
```

## Performance Considerations

The ZeroMQ-based implementation provides significant performance improvements over HTTP/REST:

- Binary protocol with minimal overhead
- Persistent connections to reduce latency
- Efficient serialization of vectors using numpy arrays
- No need for JSON encoding/decoding of large vector data

This makes it particularly suitable for high-throughput vector operations in production environments.

## Security Notes

- The server supports API keys for authentication
- Communication is not encrypted by default - use in secure networks or behind a proxy

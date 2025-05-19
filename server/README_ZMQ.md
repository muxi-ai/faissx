# FAISS Proxy Server: ZeroMQ Edition

This is the ZeroMQ-based implementation of the FAISS Proxy server, providing high-performance vector operations with minimal overhead.

## Features

- High-performance binary protocol using ZeroMQ
- Efficient vector serialization with numpy arrays
- Metadata handling with msgpack
- Multi-tenant support with API key authentication
- Persistent connections for reduced latency
- Drop-in FAISS API compatibility (when used with the client library)

## Architecture

The server uses a simple and minimal architecture:

- **ZeroMQ REP Socket**: Handles request/reply pattern with clients
- **Protocol Layer**: Serializes/deserializes messages with numpy and msgpack
- **FAISS Manager**: Core vector operations and index management
- **Authentication**: API key validation and tenant isolation

## Getting Started

### Prerequisites

- Python 3.8+
- FAISS library (`faiss-cpu` or `faiss-gpu`)
- ZeroMQ library
- Numpy
- Msgpack

### Installation

1. Clone the repository
2. Install the requirements:

```bash
pip install -r server/requirements.txt
```

### Running the Server

```bash
cd server
python run.py
```

By default, the server will bind to `0.0.0.0:5555`. You can configure this with environment variables:

```bash
FAISS_BIND_ADDRESS=127.0.0.1 FAISS_PORT=5556 python run.py
```

### Configuration Options

The server can be configured with the following environment variables:

- `FAISS_BIND_ADDRESS`: Address to bind to (default: `0.0.0.0`)
- `FAISS_PORT`: Port to listen on (default: `5555`)
- `FAISS_DATA_DIR`: Directory to store index data (default: `./data`)
- `FAISS_PROXY_API_KEYS`: Comma-separated list of API keys and tenant IDs (format: `key1:tenant1,key2:tenant2`)

## Protocol

The ZeroMQ message protocol uses a binary format:

1. **Header Size Preamble**: Msgpack-encoded sizes of all parts
2. **Header**: Msgpack-encoded dictionary with operation, authentication, and parameters
3. **Vector Data**: Raw numpy array bytes (present for add and search operations)
4. **Metadata**: Msgpack-encoded structured data (IDs, filters, etc.)

### Supported Operations

- `create_index`: Create a new FAISS index
- `add_vectors`: Add vectors to an index
- `search`: Search for similar vectors
- `delete_vector`: Delete a vector from an index
- `delete_index`: Delete an index
- `get_index_info`: Get index metadata

See the `protocol.py` file for details on message formats.

## Testing

A simple test client is provided to verify server functionality:

```bash
python test_zmq_client.py
```

This will test basic operations like:
- Creating an index
- Adding vectors
- Searching for similar vectors
- Getting index information

## Docker

You can build and run the server using Docker:

```bash
# Build the image
docker build -t faiss-proxy-zmq -f server/Dockerfile .

# Run the container
docker run -p 5555:5555 -v ./data:/app/data faiss-proxy-zmq
```

## Client Library

For complete FAISS API compatibility, use the FAISS Proxy client library. It provides a drop-in replacement for the FAISS library that transparently communicates with the server.

See the `client/` directory for details.

## Performance Considerations

The ZeroMQ-based implementation provides significant performance improvements over HTTP/REST:

- Binary protocol with minimal overhead
- Persistent connections to reduce latency
- Efficient serialization of vectors using numpy arrays
- No need for JSON encoding/decoding of large vector data

This makes it particularly suitable for high-throughput vector operations in production environments.

## Security Notes

- The server uses API keys for authentication
- Communication is not encrypted by default - use a secure network or add TLS

## Next Steps

See the `NEXT_STEPS.md` file for upcoming features and improvements.

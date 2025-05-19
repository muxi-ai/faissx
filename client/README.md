# FAISSx Client

This directory contains the FAISSx client implementation and utilities.

## Overview

The FAISSx client provides a drop-in replacement for FAISS that connects to a FAISSx server via ZeroMQ. This allows you to use FAISS in a client-server architecture where the server manages the indices and performs the vector operations.

## Features

- Drop-in replacement for FAISS IndexFlatL2
- ZeroMQ-based communication for high performance
- Support for authentication
- Simple API for creating indices, adding vectors, and searching

## Usage

```python
from faissx import client as faiss

# Configure the client
faiss.configure(
    server="tcp://localhost:45678",  # ZeroMQ server address
    api_key="your-api-key",          # Optional API key for authentication
    tenant_id="your-tenant-id"       # Optional tenant ID for multi-tenant isolation
)

# Use it like regular FAISS
index = faiss.IndexFlatL2(128)
index.add(vectors)
distances, indices = index.search(query_vectors, k=10)
```

## Environment Variables

You can configure the client using environment variables:

- `FAISSX_SERVER`: ZeroMQ server address (default: `tcp://localhost:45678`)
- `FAISSX_API_KEY`: API key for authentication
- `FAISSX_TENANT_ID`: Tenant ID for multi-tenant isolation
- `FAISSX_FALLBACK_TO_LOCAL`: Whether to fall back to local FAISS if the server is unavailable (default: `1`)

## Examples

See the `simple_client.py` script for a complete example of using the FAISSx client.

## Testing

To run the client tests:

```bash
./run_tests.sh
```

Make sure the FAISSx server is running before running the tests.

## Installation

```bash
pip install faissx
```

## Supported FAISS Features

Currently supported FAISS features:

- IndexFlatL2 (more index types coming soon)
- Basic vector add operations
- Vector search operations

## Limitations

Current limitations:

- Not all FAISS index types are supported yet
- Some advanced FAISS operations may not be available
- No GPU support yet (coming in future releases)

## License

MIT

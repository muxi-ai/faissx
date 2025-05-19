# FAISSx Client Library

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/muxi-ai/faissx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A true drop-in replacement for FAISS with optional remote execution capabilities.

## Overview

The FAISSx client provides:

1. **True drop-in replacement** for FAISS - simply change your import statement
2. **Dual execution modes**:
   - Local mode: Uses local FAISS library (default)
   - Remote mode: Uses a FAISSx server via ZeroMQ (activated by calling `configure()`)
3. **Identical API** to the original FAISS library
4. **High-performance binary protocol** for efficient remote vector operations

## Installation

```bash
# Install from PyPI
pip install faissx

# For development
git clone https://github.com/muxi-ai/faissx.git
cd faissx
pip install -e .
```

## Usage

### Local Mode (Default)

By default, the client uses your local FAISS installation with no configuration needed:

```python
# Just change the import - everything else stays the same
from faissx import client as faiss
import numpy as np

# Do FAISS stuff...
dimension = 128
index = faiss.IndexFlatL2(dimension)
vectors = np.random.random((100, dimension)).astype('float32')
index.add(vectors)
D, I = index.search(np.random.random((1, dimension)).astype('float32'), k=5)
```

### Remote Mode

To use a remote FAISSx server, add a call to `configure()` before creating any indices:

```python
from faissx import client as faiss
import numpy as np

# Connect to a remote FAISSx server
faiss.configure(
    server="tcp://localhost:45678",  # ZeroMQ server address
    api_key="test-key-1",            # API key for authentication
    tenant_id="tenant-1"             # Tenant ID for multi-tenant isolation
)

# After configure(), all operations use the remote server
dimension = 128
index = faiss.IndexFlatL2(dimension)
vectors = np.random.random((100, dimension)).astype('float32')
index.add(vectors)
D, I = index.search(np.random.random((1, dimension)).astype('float32'), k=5)
```

**Important**: When `configure()` is called, the client will always use the remote server for all operations. If the server connection fails, operations will fail - there is no automatic fallback to local mode.

## Configuration

### Environment Variables

You can configure the client using environment variables:

- `FAISSX_SERVER`: ZeroMQ server address (default: `tcp://localhost:45678`)
- `FAISSX_API_KEY`: API key for authentication
- `FAISSX_TENANT_ID`: Tenant ID for multi-tenant isolation

### Programmatic Configuration

```python
from faissx import client as faiss

# Configure the client programmatically
faiss.configure(
    server="tcp://your-server:45678",
    api_key="your-api-key",
    tenant_id="your-tenant-id"
)
```

## Supported FAISS Features

The FAISSx client currently supports:

| Feature | Status | Notes |
|---------|--------|-------|
| IndexFlatL2 | ✅ | Fully supported |
| Vector Addition | ✅ | Identical to FAISS |
| Vector Search | ✅ | Identical to FAISS |
| Index Reset | ✅ | Clears the index |
| Other Index Types | 🔄 | Coming soon |

## API Reference

### Main Functions

#### `configure(server=None, api_key=None, tenant_id=None)`

Configures the client to use a remote FAISSx server.

- **server**: ZeroMQ server address (e.g., "tcp://localhost:45678")
- **api_key**: API key for authentication
- **tenant_id**: Tenant ID for multi-tenant isolation

### IndexFlatL2 Class

```python
class IndexFlatL2:
    def __init__(self, d: int):
        """
        Initialize the index with specified dimension.

        Args:
            d (int): Vector dimension for the index
        """

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)
        """

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.

        Args:
            x (np.ndarray): Query vectors, shape (n, d)
            k (int): Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Distances array of shape (n, k)
                - Indices array of shape (n, k)
        """

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
```

## Performance Considerations

When using the remote mode:

1. **Data Transfer**: Large vector operations involve transferring data over the network. The ZeroMQ protocol minimizes overhead but network latency applies.

2. **Connection Handling**: The client maintains a persistent connection to the server for efficient operations.

3. **Serialization**: Vectors are serialized using msgpack for efficient binary transfer.

## Advanced Usage

### Error Handling

```python
from faissx import client as faiss

try:
    faiss.configure(server="tcp://non-existent-server:45678")
    index = faiss.IndexFlatL2(128)
except RuntimeError as e:
    print(f"Connection error: {e}")
    # Handle the error or fall back to local FAISS
    import faiss as local_faiss
    index = local_faiss.IndexFlatL2(128)
```

### Working with Existing FAISS Code

Since FAISSx is a drop-in replacement, you can easily switch between local FAISS and remote FAISSx by changing imports:

```python
# Original code using local FAISS
import faiss
index = faiss.IndexFlatL2(128)

# Switch to FAISSx (local mode)
from faissx import client as faiss
index = faiss.IndexFlatL2(128)

# Switch to FAISSx (remote mode)
from faissx import client as faiss
faiss.configure(server="tcp://localhost:45678")
index = faiss.IndexFlatL2(128)
```

## Examples

Check out the example scripts in the repository:

- [Simple Client](../examples/simple_client.py): Basic usage of the client
- [Remote Search Example](../examples/remote_search.py): Using remote search operations

## Troubleshooting

### Connection Issues

If you're having trouble connecting to the server:

1. Ensure the server is running: `nc -z localhost 45678`
2. Check your firewall settings
3. Verify you've provided the correct API key and tenant ID if authentication is enabled

### Vector Dimension Mismatch

If you get dimension mismatch errors, ensure:

1. Your index was created with the correct dimension
2. All vectors have the same dimension as the index
3. All vectors are properly converted to float32 type

## License

FAISSx is licensed under the [Apache 2.0 license](./LICENSE).

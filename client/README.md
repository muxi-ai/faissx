# FAISSx Client Library

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/muxi-ai/faissx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A true drop-in replacement for FAISS that extends Facebook's vector similarity search library with optional remote execution capabilities.

## Quick Start

```python
# Just change your import - everything else stays the same!
from faissx import client as faiss  # ← replace "import faiss" with this
import numpy as np

# Your existing FAISS code works unchanged
dimension = 128
index = faiss.IndexFlatL2(dimension)
vectors = np.random.random((100, dimension)).astype('float32')
index.add(vectors)
D, I = index.search(np.random.random((1, dimension)).astype('float32'), k=5)

# Optional: Connect to a remote FAISSx server for distributed search
faiss.configure(server="tcp://localhost:45678", api_key="your-key", tenant_id="your-tenant")
# Now all operations use the remote server
```

## Overview

The FAISSx client provides:

1. **True drop-in replacement** for FAISS - simply change your import statement
2. **Dual execution modes**:
   - Local mode: Uses local FAISS library (default)
   - Remote mode: Uses a FAISSx server via ZeroMQ (activated by calling `configure()`)
3. **Identical API** to the original FAISS library
4. **High-performance binary protocol** for efficient remote vector operations
5. **Optimized implementations** with robust fallback strategies
6. **Vector caching** for enhanced retrieval and reconstruction
7. **Batched processing** for large vector operations

## When to Use FAISSx Client

Choose FAISSx over vanilla FAISS when you need:

- **Microservices architecture**: Multiple services sharing the same vector search without duplicating large indices
- **Multi-language support**: Access FAISS from JavaScript, Go, Java, or other languages via the network API
- **Resource optimization**: Centralize memory-intensive vector operations instead of loading indices in every application instance
- **Team collaboration**: Share trained indices between data science teams and production applications
- **Container deployments**: Stateless applications that can't maintain large in-memory indices
- **Enterprise features**: Authentication, multi-tenancy, and centralized monitoring for vector search

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

## Recent Optimizations

FAISSx client has been significantly optimized in recent versions:

### Vector Reconstruction and Caching

- Multiple fallback methods for retrieving vectors
- Vector caching for improved performance
- Batched operations for large vector sets
- Robust error handling for remote operations

### Enhanced Index Implementations

- Optimized IndexPQ with comprehensive training strategies
- Improved IndexIVFScalarQuantizer with better error recovery
- Enhanced index modification capabilities (merging, splitting)
- Detailed performance logging for all operations

### Persistence Layer Improvements

- Better handling for both local and remote modes
- Special handling for IDMap and IDMap2 classes
- More robust vector reconstruction for saved indices
- Optimized file formats for different index types

### Error Recovery

- Improved error recovery with graceful fallbacks
- Automatic retries with exponential backoff
- Better handling of server limitations
- Informative logging for troubleshooting

## Supported FAISS Features

The FAISSx client currently supports:

| Feature | Status | Notes |
|---------|--------|-------|
| IndexFlatL2 | ✅ | Fully supported |
| IndexIVFFlat | ✅ | Fully supported with parameter tuning |
| IndexHNSWFlat | ✅ | Fully supported |
| IndexPQ | ✅ | Optimized with robust fallbacks |
| IndexIVFPQ | ✅ | Fully supported |
| IndexScalarQuantizer | ✅ | Fully supported |
| IndexIVFScalarQuantizer | ✅ | Optimized with error recovery |
| IndexIDMap/IDMap2 | ✅ | Fully supported with special persistence |
| Vector Addition | ✅ | Identical to FAISS |
| Vector Search | ✅ | Identical to FAISS with batching |
| Range Search | ✅ | Supported with server fallbacks |
| Index Reset | ✅ | Clears the index (recreates in remote mode) |
| Vector Reconstruction | ✅ | Multiple fallback methods |
| Index Persistence | ✅ | Optimized read_index/write_index support |
| Index Modification | ✅ | Merge and split with batched operations |
| Parameter Controls | ✅ | Fine-grained tuning for performance |

## Index Selection Guide

Choose the right index type for your use case:

### **Exact Search (Small to Medium datasets)**
- **`IndexFlatL2`**: Best for exact L2 distance search, up to ~1M vectors
- **`IndexFlatIP`**: Best for exact inner product search (cosine similarity)

### **Approximate Search (Large datasets)**
- **`IndexIVFFlat`**: Good balance of speed and accuracy for 1M+ vectors
- **`IndexHNSWFlat`**: Excellent query performance, higher memory usage
- **`IndexIVFPQ`**: Best memory efficiency for very large datasets (10M+ vectors)

### **Specialized Use Cases**
- **`IndexPQ`**: Memory-constrained environments, good compression
- **`IndexScalarQuantizer`**: Good compression with better accuracy than PQ
- **`IndexIDMap/IDMap2`**: When you need to maintain your own vector IDs
- **Binary indices**: For binary embeddings (e.g., hashing-based methods)

### **Performance vs Memory Trade-offs**
```python
# High accuracy, high memory
index = faiss.IndexFlatL2(dimension)

# Good balance for most use cases
index = faiss.IndexIVFFlat(dimension, nlist=100)
index.nprobe = 10  # Adjust for speed vs accuracy

# Maximum memory efficiency
index = faiss.IndexIVFPQ(dimension, nlist=100, m=8, nbits=8)
```

## API Reference

### Main Functions

#### `configure(server=None, api_key=None, tenant_id=None)`

Configures the client to use a remote FAISSx server.

- **server**: ZeroMQ server address (e.g., "tcp://localhost:45678")
- **api_key**: API key for authentication
- **tenant_id**: Tenant ID for multi-tenant isolation

### Common Index Methods

Most index classes implement these standard methods:

```python
def add(self, x: np.ndarray) -> None:
    """Add vectors to the index."""

def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Search for k nearest neighbors for each query vector."""

def reset(self) -> None:
    """Reset the index to its initial state."""

def reconstruct(self, i: int) -> np.ndarray:
    """Reconstruct vector i from the index."""

def reconstruct_n(self, i0: int, ni: int) -> np.ndarray:
    """Reconstruct ni vectors from the index, starting from i0."""

def train(self, x: np.ndarray) -> None:
    """Train the index on the given vectors."""

def set_parameter(self, name: str, value: Any) -> None:
    """Set a parameter for the index."""
```

### Index Modification Utilities

```python
from faissx.client.modification import merge_indices, split_index

# Merge multiple indices
merged_index = merge_indices([index1, index2, index3])

# Split an index into multiple parts
split_indices = split_index(index, 3)  # Split into 3 parts
```

## Performance Considerations

When using the remote mode:

1. **Data Transfer**: Large vector operations involve transferring data over the network. The ZeroMQ protocol minimizes overhead but network latency applies.

2. **Connection Handling**: The client maintains a persistent connection to the server for efficient operations.

3. **Serialization**: Vectors are serialized using msgpack for efficient binary transfer.

4. **Batched Operations**: Large vector sets are automatically processed in batches to optimize memory usage and network transfer.

5. **Vector Caching**: Reconstructed vectors can be cached to improve performance of subsequent operations.

## Advanced Usage

### Error Handling

```python
from faissx import client as faiss
from faissx.client.recovery import with_retry, configure_recovery

# Configure recovery behavior
configure_recovery(max_retries=5, initial_backoff=1.0, backoff_factor=2.0)

try:
    faiss.configure(server="tcp://your-server:45678")

    # Use automatic retry for operations
    index = faiss.IndexFlatL2(128)
    with_retry(index.add, vectors)

except RuntimeError as e:
    print(f"Connection error: {e}")
    # Handle the error appropriately
```

### Memory Management

```python
from faissx.client.optimization import memory_manager

# Set memory usage options
memory_manager.set_option('max_memory_usage_mb', 1024)  # 1GB limit
memory_manager.set_option('use_memory_mapping', True)  # Use mmap for large indices
memory_manager.set_option('cache_vectors', True)  # Enable vector caching
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

- [Simple Client](../examples/simple_client.py): Basic usage of the client with server connectivity
- [Server Example](../examples/server_example.py): How to configure and run the FAISSx server
- [IVF Index Example](../examples/ivf_index_example.py): Performance comparison of different index types
- [HNSW and PQ Example](../examples/hnsw_and_pq_example.py): Advanced index types with memory/performance trade-offs
- [Index Modification Example](../examples/index_modification_example.py): Merging and splitting indices
- [Batch Operations Example](../examples/batch_operations_example.py): Performance optimization with batched operations
- [Range Search Example](../examples/range_search_example.py): Radius-based vector search
- [Recovery Example](../examples/recovery_example.py): Error handling and retry strategies

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

### Operational Considerations

When using remote mode, be aware of these considerations:

1. **Network latency**: Operations involve network communication which adds latency compared to local mode
2. **Graceful degradation**: The client implements fallback strategies for error recovery (e.g., returning zero vectors if reconstruction fails)
3. **Specialized index types**: Some advanced index types like `IndexIDMap2` may have limited server-side support
4. **Binary indices**: Full server-side support for binary indices is still in development

The FAISSx server implements all core FAISS operations including vector reconstruction, index reset, and range search. Most limitations you encounter will be related to network issues rather than missing server functionality.

## License

FAISSx is licensed under the [Apache 2.0 license](../LICENSE).

> [!NOTE]
> FAISSx depends on [FAISS](https://github.com/facebookresearch/faiss), which is licensed under the MIT License.

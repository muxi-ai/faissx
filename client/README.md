# FAISSx Client

A drop-in replacement for FAISS with remote execution capabilities

## Overview

FAISSx Client is a Python library that provides a compatible API with FAISS but delegates all operations to a remote FAISSx server. This allows applications to use FAISS in a distributed environment without changing their code significantly.

## Installation

```bash
pip install faissx
```

## Usage

### Basic Usage

```python
# Standard FAISS import replaced with faissx import
import faissx as faiss

# Create an index - transparently creates remote index
index = faiss.IndexFlatL2(128)

# Add vectors - transparently sends to remote service
import numpy as np
vectors = np.random.random((100, 128)).astype('float32')
index.add(vectors)

# Search - transparently queries remote service
query_vectors = np.random.random((10, 128)).astype('float32')
D, I = index.search(query_vectors, k=5)
```

### Configuration

```python
# Configure once at application startup
import faissx

faissx.configure(
    api_url="http://faiss-service:8000",
    api_key="your-api-key",
    tenant_id="your-tenant-id"
)
```

### Environment Variables

You can also configure FAISSx using environment variables:

```bash
export faissx_API_URL="http://faiss-service:8000"
export faissx_API_KEY="your-api-key"
export faissx_TENANT_ID="your-tenant-id"
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

# FAISS Proxy Server

A lightweight proxy service for FAISS vector operations in multi-server deployments.

## Overview

FAISS Proxy Server provides a simple REST API for storing, retrieving, and searching vector embeddings. It enables sharing FAISS indices across multiple applications with tenant isolation. The service is designed to be:

- **Simple**: Focused on core vector operations
- **Fast**: Optimized for vector search performance
- **Scalable**: Supports multiple tenants and indices
- **Secure**: API key authentication and tenant isolation

## Features

- Create and manage multiple FAISS indices
- Store vectors with associated metadata
- Perform vector similarity search with metadata filtering
- API key-based authentication
- Tenant isolation for multi-application deployments
- Simple persistence to disk
- Health and metrics endpoints for monitoring

## Quick Start

### Using Docker

```bash
# Build the Docker image
cd server
docker build -t faiss-proxy-server .

# Run the container
docker run -p 8000:8000 -v /path/to/data:/data faiss-proxy-server
```

### Using PyOxidizer (Standalone Executable)

```bash
# Install PyOxidizer
pip install pyoxidizer

# Build the executable
cd server
pyoxidizer build --release

# Run the executable
./build/x86_64-unknown-linux-gnu/release/install/faiss-proxy
```

### Setting API Keys

API keys can be set using an environment variable:

```bash
export FAISS_PROXY_API_KEYS="key1:tenant1,key2:tenant2"
```

Or by modifying the `auth.py` file for a development setup.

## API Usage

### Authentication

All API requests require an API key in the `X-API-Key` header:

```
X-API-Key: your-api-key
```

### Create an Index

```bash
curl -X POST "http://localhost:8000/v1/index" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key-1" \
  -d '{
    "name": "my-index",
    "dimension": 1536,
    "index_type": "IndexFlatL2",
    "tenant_id": "tenant-1"
  }'
```

### Add Vectors

```bash
curl -X POST "http://localhost:8000/v1/index/{index_id}/vectors" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key-1" \
  -d '{
    "vectors": [
      {
        "id": "vec1",
        "values": [0.1, 0.2, ...],
        "metadata": {
          "source": "document-1",
          "timestamp": 1645023489
        }
      }
    ]
  }'
```

### Search Vectors

```bash
curl -X GET "http://localhost:8000/v1/index/{index_id}/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key-1" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "k": 10,
    "filter": {
      "source": "document-1"
    }
  }'
```

### Delete a Vector

```bash
curl -X DELETE "http://localhost:8000/v1/index/{index_id}/vectors/{vector_id}" \
  -H "X-API-Key: test-key-1"
```

### Delete an Index

```bash
curl -X DELETE "http://localhost:8000/v1/index/{index_id}" \
  -H "X-API-Key: test-key-1"
```

## Technical Details

### Project Structure

```
server/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── index.py
│   │   └── vectors.py
│   └── utils/
│       ├── __init__.py
│       ├── auth.py
│       └── faiss_manager.py
├── requirements.txt
├── Dockerfile
└── pyoxidizer.bzl
```

### Data Persistence

FAISS indices and metadata are persisted to disk at:

- Docker: `/data/{tenant_id}/{index_id}/`
- Local: `./data/{tenant_id}/{index_id}/` (default)

The data directory can be configured using the `FAISS_DATA_DIR` environment variable.

## Configuration

Environment variables:

- `FAISS_DATA_DIR`: Directory for storing indices and metadata
- `FAISS_PROXY_API_KEYS`: Comma-separated list of API keys and tenant IDs (format: "key1:tenant1,key2:tenant2")

## Development

### Setup

```bash
# Create a virtual environment
cd server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the development server
uvicorn app.main:app --reload
```

### Testing

API documentation is available at http://localhost:8000/docs when running the development server.

## Limitations

- FAISS doesn't support direct deletion of vectors. When deleting vectors, only the metadata is removed, but the vector remains in the index.
- For production use, consider implementing proper deletion by rebuilding indices periodically.
- This implementation is optimized for simplicity, not for very large indices (millions of vectors).

## License

MIT

# FAISS Proxy

> Remote Vector Database Service

## Overview

FAISS Proxy is a lightweight microservice that provides remote access to FAISS vector indices. It enables multiple clients to store, retrieve, and search vector embeddings through a simple REST API. The proxy handles index management, tenant isolation, and basic authentication while leaving application-specific logic to the clients.

## Objectives

- Enable shared access to FAISS indices across multiple clients
- Provide tenant isolation for multi-application deployments
- Keep the implementation simple and general-purpose
- Offer a ready-to-deploy Docker container
- Support basic performance and scaling requirements
- Provide a drop-in replacement Python client library for seamless integration

## Functional Requirements

1. **Core Vector Operations**
   - Store vectors with associated metadata
   - Retrieve vectors by ID
   - Search for similar vectors with configurable parameters
   - Delete vectors from indices

2. **Index Management**
   - Create and delete indices
   - Configure index parameters (dimension, index type)
   - Support multiple indices per tenant
   - Automatic index optimization

3. **API Interface**
   - RESTful API for all operations
   - Simple authentication mechanism
   - Tenant identification for isolation

4. **Python Client Library**
   - Drop-in replacement for FAISS Python library
   - Support for all major FAISS index types
   - Compatible API with the original FAISS library
   - Transparent remote execution

## API Design

### Endpoints

1. **Index Management**
   - `POST /v1/index` - Create a new index
   - `GET /v1/index/{id}` - Get index information
   - `DELETE /v1/index/{id}` - Delete an index

2. **Vector Operations**
   - `POST /v1/index/{id}/vectors` - Add vectors to index
   - `GET /v1/index/{id}/search` - Search for similar vectors
   - `DELETE /v1/index/{id}/vectors/{vector_id}` - Delete a vector

3. **Advanced FAISS Operations**
   - `POST /v1/index/{id}/train` - Train an index (for quantized indices)
   - `GET /v1/index/{id}/ntotal` - Get total number of vectors
   - `POST /v1/index/{id}/reset` - Reset an index
   - `POST /v1/index/{id}/specialized_search` - Specialized search operations (range search, etc.)

4. **Administration**
   - `GET /v1/health` - Service health check
   - `GET /v1/metrics` - Performance metrics

### Data Models

1. **Index Creation**

```json
{
  "name": "my-index",
  "dimension": 1536,
  "index_type": "IndexFlatL2",
  "tenant_id": "client-123"
}
```

2. **Vector Addition**

```json
{
  "vectors": [
    {
      "id": "vec1",
      "values": [0.1, 0.2, ...],
      "metadata": {
        "source": "document-1",
        "timestamp": 1645023489
      }
    },
    // Additional vectors...
  ]
}
```

3. **Search Request**

```json
{
  "vector": [0.1, 0.2, ...],
  "k": 10,
  "filter": {
    "source": "document-1"
  }
}
```

4. **Search Response**

```json
{
  "results": [
    {
      "id": "vec1",
      "score": 0.85,
      "metadata": { "source": "document-1", "timestamp": 1645023489 }
    },
    // Additional results...
  ]
}
```

## Python Client Library

### Design Principles

1. **Drop-In Compatibility**
   - Match the FAISS API exactly for seamless migration
   - Transparently handle remote calls
   - Support typical FAISS usage patterns

2. **Authentication & Configuration**
   - Simple configuration for API endpoint and authentication
   - Environment variable support
   - Default to local FAISS if no remote configured

3. **Performance Optimization**
   - Batch operations where possible
   - Connection pooling
   - Optional local caching

### Example Usage

```python
# Standard FAISS import replaced with proxy import
import faiss_proxy as faiss

# Create an index - transparently creates remote index
index = faiss.IndexFlatL2(128)

# Add vectors - transparently sends to remote service
index.add(vectors)

# Search - transparently queries remote service
D, I = index.search(query_vectors, k=10)
```

### Configuration

```python
# Configure once at application startup
import faiss_proxy

faiss_proxy.configure(
    api_url="http://faiss-service:8000",
    api_key="your-api-key",
    tenant_id="your-tenant-id"
)

# Then use as normal FAISS
import faiss_proxy as faiss
```

## Non-Functional Requirements

1. **Performance**
   - Support at least 100 vector searches per second
   - Search latency under 50ms for indices with up to 10,000 vectors
   - Support for at least 50 concurrent indices

2. **Security**
   - API key authentication
   - Strict tenant isolation
   - TLS for all communications

3. **Observability**
   - Basic logging of operations
   - Simple metrics for monitoring
   - Health check endpoint

## Implementation Plan

### Phase 1: Core Server Implementation (1 day)
1. Set up basic service structure and API endpoints
2. Implement FAISS integration for vector operations
3. Create index management system
4. Add basic authentication and tenant isolation

### Phase 2: Client Library Implementation (1 day)
1. Implement proxy classes for core FAISS types
2. Create transparent remote call mechanism
3. Add configuration and authentication
4. Support basic error handling

### Phase 3: Extended FAISS Capabilities (1-2 days)
1. Add support for additional index types
2. Implement training methods
3. Add specialized search operations
4. Support advanced FAISS features

### Phase 4: Testing & Packaging (1 day)
1. End-to-end testing
2. Performance testing
3. Package server as Docker container
4. Package client as PyPI package

## Project Structure

The project will be split into server and client components:

```
faiss-proxy/
├── server/           # FastAPI server implementation
│   ├── app/          # Server application code
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── client/           # Python client library
│   ├── faiss_proxy/  # Package directory
│   ├── setup.py
│   ├── README.md
│   └── tests/        # Client tests
└── README.md         # Project overview
```

## Future Enhancements

1. **Advanced Features**
   - Support for additional FAISS index types
   - GPU acceleration via remote execution
   - Distributed indices across multiple servers

2. **Scaling**
   - Multi-node support
   - Sharding for very large indices
   - Read replicas for search-heavy workloads

## Documentation

1. **Server Documentation**
   - Docker-based setup instructions
   - API reference

2. **Client Library Documentation**
   - Installation instructions
   - API reference
   - Migration guide from FAISS

## Conclusion

FAISS Proxy provides a simple, general-purpose remote vector database service with a drop-in replacement client library. This enables seamless migration from local FAISS to a distributed deployment, making it ideal for scaling applications like MUXI that require consistent vector operations across multiple instances.

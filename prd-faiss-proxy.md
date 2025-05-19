# FAISS Proxy (0MQ Edition)

> Ultra-Fast Remote Vector Database Service (0MQ-based)

## Drop-In Replacement Behavior

**Seamless Local/Remote Switching:**

The `faiss_proxy` client library is designed to be a true drop-in replacement for the original FAISS library. The only difference is whether you call `faiss.configure()`:

- **Remote Mode:**

  ```python
  import faiss_proxy as faiss
  faiss.configure("tcp://remote-server:5555")
  # All FAISS operations are transparently executed on the remote server

  index = faiss.IndexFlatL2(128)
  index.add(vectors)
  D, I = index.search(query, k=10)
  ```

- **Local Mode (Default):**

  ```python
  import faiss_proxy as faiss
  # No faiss.configure() called, so all operations use the local FAISS library

  index = faiss.IndexFlatL2(128)
  index.add(vectors)
  D, I = index.search(query, k=10)
  ```

**Key Points:**

- If `faiss.configure()` is not called, `faiss_proxy` will automatically use the local FAISS implementation.
- The API and behavior are identical to the original FAISS library.
- No need to change any other code—just swap the import and (optionally) call `faiss.configure()` to use remote.
- This allows you to use the same codebase for both local and remote vector search, with zero friction.

## Overview

FAISS Proxy is a high-performance microservice that provides remote access to FAISS vector indices using ZeroMQ (0MQ) for communication. It enables multiple clients to store, retrieve, and search vector embeddings through a persistent, binary protocol. The proxy handles index management, tenant isolation, and authentication, while maximizing throughput and minimizing latency—ideal for large-scale, real-time vector search workloads.

## Objectives

- Enable shared access to FAISS indices across multiple clients
- Provide tenant isolation for multi-application deployments
- Achieve maximum speed and throughput (no HTTP/REST overhead)
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

3. **Binary Protocol Interface (0MQ)**
   - All operations use persistent 0MQ sockets
   - Messages are sent as binary blobs (numpy arrays, msgpack for metadata)
   - Simple authentication mechanism (API key in message header)
   - Tenant identification for isolation

4. **Python Client Library**
   - Drop-in replacement for FAISS Python library
   - Support for all major FAISS index types
   - Compatible API with the original FAISS library
   - Transparent remote execution over 0MQ

## Protocol Design

### Message Structure

All client-server communication happens over a persistent 0MQ REQ/REP socket. Each message consists of:

- **Header** (msgpack):
  - `operation`: string (e.g., `create_index`, `add_vectors`, `search`, ...)
  - `api_key`: string
  - `tenant_id`: string
  - `index_id`: string (if applicable)
  - `request_id`: string (for tracing)
- **Payload** (binary):
  - For vectors: raw numpy array bytes
  - For metadata: msgpack

### Example Operations

1. **Create Index**
   - Header: `{operation: "create_index", dimension: 1536, index_type: "IndexFlatL2", ...}`
   - Payload: none
   - Response: `{status: "ok", index_id: "..."}`

2. **Add Vectors**
   - Header: `{operation: "add_vectors", index_id: "...", ...}`
   - Payload: [vectors as numpy bytes] + [metadata as msgpack]
   - Response: `{status: "ok", added_count: N, failed_count: M}`

3. **Search**
   - Header: `{operation: "search", index_id: "...", k: 10, ...}`
   - Payload: [query vector as numpy bytes] + [optional filter as msgpack]
   - Response: `{status: "ok", results: [ ... ]}` (results as msgpack)

4. **Delete Vector**
   - Header: `{operation: "delete_vector", index_id: "...", vector_id: "..."}`
   - Payload: none
   - Response: `{status: "ok"}`

## Python Client Library

### Design Principles

1. **Drop-In Compatibility**
   - Match the FAISS API exactly for seamless migration
   - Transparently handle remote calls over 0MQ
   - Support typical FAISS usage patterns

2. **Authentication & Configuration**
   - Simple configuration for 0MQ endpoint and authentication
   - Environment variable support
   - Default to local FAISS if no remote configured

3. **Performance Optimization**
   - Batch operations by default
   - Persistent socket connection (no reconnect per operation)
   - Optional local caching

### Example Usage

```python
# Standard FAISS import replaced with proxy import
import faiss_proxy as faiss

# Configure once at application startup
faiss_proxy.configure(
    zmq_url="tcp://faiss-service:5555",
    api_key="your-api-key",
    tenant_id="your-tenant-id"
)

# Create an index - transparently creates remote index
index = faiss.IndexFlatL2(128)

# Add vectors - transparently sends to remote service
index.add(vectors)

# Search - transparently queries remote service
D, I = index.search(query_vectors, k=10)
```

## Server Implementation

- Runs a 0MQ REP socket (e.g., `tcp://*:5555`)
- Handles incoming binary messages, dispatches to FAISS manager
- Uses numpy for vector serialization, msgpack for metadata
- Supports multi-tenant isolation and API key authentication
- Can be packaged as a Docker container

## Non-Functional Requirements

1. **Performance**
   - Support at least 1,000 vector searches per second (10x HTTP baseline)
   - Search latency under 10ms for indices with up to 10,000 vectors
   - Support for at least 100 concurrent indices

2. **Security**
   - API key authentication in message header
   - Strict tenant isolation
   - (Optional) CurveZMQ or TLS for encrypted communication

3. **Observability**
   - Basic logging of operations
   - Simple metrics for monitoring
   - Health check operation via protocol

## Implementation Plan

### Phase 1: Core Server Implementation
1. Set up 0MQ REP socket and message protocol
2. Implement FAISS integration for vector operations
3. Create index management system
4. Add basic authentication and tenant isolation

### Phase 2: Client Library Implementation
1. Implement proxy classes for core FAISS types
2. Create transparent remote call mechanism over 0MQ
3. Add configuration and authentication
4. Support basic error handling

### Phase 3: Extended FAISS Capabilities
1. Add support for additional index types
2. Implement training methods
3. Add specialized search operations
4. Support advanced FAISS features

### Phase 4: Testing & Packaging
1. End-to-end testing
2. Performance testing
3. Package server as Docker container
4. Package client as PyPI package

## Project Structure

The project is split into server and client components:

```
faiss-proxy/
├── server/           # 0MQ server implementation
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
   - 0MQ protocol reference

2. **Client Library Documentation**
   - Installation instructions
   - API reference
   - Migration guide from FAISS

## Conclusion

FAISS Proxy (0MQ Edition) provides a blazing-fast, general-purpose remote vector database service with a drop-in replacement client library. By leveraging 0MQ and binary protocols, it enables seamless migration from local FAISS to a distributed, high-performance deployment—ideal for applications demanding maximum speed and scalability.


# FAISS Proxy (0MQ Edition)

> A blazing-fast proxy service for FAISS vector operations in multi-server deployments, using ZeroMQ (0MQ) for maximum performance.

## Overview

FAISS Proxy provides a remote vector database service with a drop-in replacement Python client library. It enables sharing FAISS indices across multiple applications with tenant isolation, making it ideal for scaling AI applications that require vector search capabilities. Communication is handled via a persistent, binary 0MQ protocol for ultra-low latency and high throughput.

## Drop-In Replacement Behavior

**Seamless Local/Remote Switching:**

The `faiss_proxy` client library is a true drop-in replacement for the original FAISS library. The only difference is whether you call `faiss_proxy.configure()`:

- **Remote Mode:**

  ```python
  import faiss_proxy as faiss
  faiss_proxy.configure(zmq_url="tcp://remote-server:5555", api_key="your-key", tenant_id="your-tenant")
  # All FAISS operations are transparently executed on the remote server

  index = faiss.IndexFlatL2(128)
  index.add(vectors)
  D, I = index.search(query, k=10)
  ```
- **Local Mode (Default):**

  ```python
  import faiss_proxy as faiss
  # No configure() called, so all operations use the local FAISS library

  index = faiss.IndexFlatL2(128)
  index.add(vectors)
  D, I = index.search(query, k=10)
  ```
- If `faiss_proxy.configure()` is not called, `faiss_proxy` will automatically use the local FAISS implementation.
- The API and behavior are identical to the original FAISS library.
- No need to change any other code—just swap the import and (optionally) call `configure()` to use remote.

## Project Structure

This repository is organized into two main components:

- **Server**: A 0MQ service that provides a binary protocol for FAISS operations
- **Client**: A Python library that mimics the FAISS API but delegates operations to the server when configured

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

## Server Features

- Create and manage multiple FAISS indices
- Store vectors with associated metadata
- Perform vector similarity search with metadata filtering
- API key-based authentication (in message header)
- Tenant isolation for multi-application deployments
- Simple persistence to disk
- Health and metrics operations via protocol
- Ultra-fast binary protocol over persistent 0MQ sockets

## Client Features

- Drop-in replacement for FAISS Python library
- Transparent remote execution over 0MQ
- Compatible API with the original FAISS library
- Simple configuration via environment variables or code
- Seamless fallback to local FAISS if not configured for remote

## Getting Started

### Server Deployment

See [server/README.md](server/README.md) for detailed instructions on deploying the server.

### Client Installation

```bash
# From PyPI (coming soon)
pip install faiss-proxy

# From source
cd client
pip install -e .
```

See [client/README.md](client/README.md) for detailed instructions on using the client library.

## License

Apache License 2.0

# FAISS Proxy

A lightweight proxy service for FAISS vector operations in multi-server deployments.

## Overview

FAISS Proxy provides a remote vector database service with a drop-in replacement Python client library. It enables sharing FAISS indices across multiple applications with tenant isolation, making it ideal for scaling AI applications that require vector search capabilities.

## Project Structure

This repository is organized into two main components:

- **Server**: A FastAPI service that provides a REST API for FAISS operations
- **Client**: A Python library that mimics the FAISS API but delegates operations to the server

```
faiss-proxy/
├── server/           # FastAPI server implementation
│   ├── app/          # Server application code
│   ├── requirements.txt
│   ├── Dockerfile
│   └── pyoxidizer.bzl
└── client/           # Python client library
    ├── faiss_proxy/  # Package directory
    ├── setup.py
    ├── README.md
    └── tests/        # Client tests
```

## Server Features

- Create and manage multiple FAISS indices
- Store vectors with associated metadata
- Perform vector similarity search with metadata filtering
- API key-based authentication
- Tenant isolation for multi-application deployments
- Simple persistence to disk
- Health and metrics endpoints for monitoring

## Client Features

- Drop-in replacement for FAISS Python library
- Transparent remote execution
- Compatible API with the original FAISS library
- Simple configuration via environment variables or code

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

## Integration with MUXI

FAISS Proxy is designed to work seamlessly with MUXI. See [client/README.md](client/README.md) for instructions on integrating with MUXI.

## License

MIT

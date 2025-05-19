# Changelog

All notable changes to FAISSx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1]

Initial release of FAISSx, a high-performance vector database proxy using FAISS and ZeroMQ.

### Added

#### Project Infrastructure
- Project renamed from FAISS-Proxy to FAISSx
- Directory structure reorganized (faissx, client, server, examples, data)
- Build system configured (setup.py, MANIFEST.in)
- Documentation updated
- Basic Docker deployment

#### Server Implementation
- ZeroMQ server application structure
- Authentication with API keys
- FAISS manager for vector operations
- Binary protocol for CRUD operations on indices
- Vector addition and search operations
- Tenant isolation for multi-application deployments
- Docker container setup
- Comprehensive server documentation with API protocol details

#### Client Implementation
- Client package structure
- Configuration management
- Remote API client using ZeroMQ
- IndexFlatL2 implementation with API parity to FAISS
- Documentation for client usage
- Drop-in replacement behavior for seamless FAISS integration
- Test suite for client functionality

### Documentation
- Main README with project overview and quick start guides
- Server README with architecture and API details
- Client README with detailed usage examples
- Implementation status tracker
- Contributing guidelines

## [Unreleased]

### Coming Soon
- Support for additional FAISS index types (IndexIVFFlat, IndexHNSW, IndexPQ)
- Index training endpoints
- Specialized search operations (range search, etc.)
- Proper deletion through index rebuilding
- Benchmarking tools
- Additional client-side features and FAISS compatibility
- Error recovery and reconnection
- Metadata filtering
- Batch operations support

# Changelog

## 0.0.2

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


## 0.0.1

Initial release of FAISSx, a high-performance vector database proxy using FAISS and ZeroMQ.

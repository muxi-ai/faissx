# Changelog

## 0.0.3

### Added

#### Advanced Features
- Implemented IndexIDMap and IndexIDMap2 for custom vector IDs
- Added factory pattern (index_factory) for creating indices from string descriptions
- Implemented direct index persistence (write_index/read_index)
- Added index modification capabilities (merging and splitting indices)
- Created optimization controls with fine-grained parameters:
  - Search parameters: nprobe, efSearch, k_factor
  - Training parameters: n_iter, min_points_per_centroid
  - HNSW specific parameters: efConstruction
  - Batch operation parameters: batch_size
  - Quality vs speed tradeoff parameters
- Implemented memory management options for efficient resource usage:
  - Memory mapping for large indices
  - Memory usage limits and tracking
  - Index caching with configurable thresholds
  - Automatic unloading of unused indices
  - I/O buffer size controls
- Added error recovery and reconnection capabilities with automatic retries and exponential backoff:
  - Configurable retry attempts and backoff strategy
  - Automatic reconnection on network failures
  - Event callbacks for disconnect/reconnect events
  - Connection monitoring with health checks
  - Manual and automatic recovery options

#### Core Improvements
- Enhanced modular architecture with consistent interfaces
- Extended example suite with comprehensive feature demonstrations
- Improved documentation for advanced usage patterns
- Added unit tests for new functionality

### Changed
- Refactored index implementations for better performance
- Improved memory efficiency for large indices
- Enhanced error handling and validation

---

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

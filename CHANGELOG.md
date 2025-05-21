# Changelog

## 0.0.3

### Added

#### Core Index Implementation
- Created optimized implementations of core index types:
  - IndexPQ with robust vector extraction and fallbacks
  - IndexIVFScalarQuantizer with improved training strategies
  - IndexIDMap and IndexIDMap2 for custom vector IDs
  - Modification module with batched vector operations
- Implemented vector caching across all index implementations
- Added batched processing for large vector operations
- Enhanced vector reconstruction with multiple fallback strategies

#### Advanced Features
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

#### Error Handling and Resilience
- Added robust error handling for server limitations
- Implemented error recovery and reconnection capabilities:
  - Configurable retry attempts and backoff strategy
  - Automatic reconnection on network failures
  - Event callbacks for disconnect/reconnect events
  - Connection monitoring with health checks
  - Manual and automatic recovery options
- Fixed client connection issues with more robust error handling
- Improved handling when get_client() connection fails

#### Testing and Documentation
- Created comprehensive test suite for all optimized implementations
- Added test scripts for remote mode functionality
- Extended example suite with comprehensive feature demonstrations
- Improved documentation for advanced usage patterns

### Changed
- Organized project structure for better maintainability:
  - Moved all tests to logical locations (client/tests and server/tests)
  - Created dedicated indices/ directory for index-specific tests
  - Removed obsolete utility scripts and duplicate tests
  - Consolidated documentation into a dedicated notes/ directory
- Refactored index implementations for better performance
- Improved memory efficiency for large indices
- Enhanced documentation with updated README files
- Consolidated development notes and technical documentation
- Improved handling of non-empty base indices for IndexIDMap types
- Fixed environment variable handling for server port configuration

### Fixed
- Addressed server-side limitations with client-side workarounds
- Improved recovery from common error conditions
- Enhanced compatibility for different server implementations
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

# Changelog

## 0.0.3 (General Availability)

### New Features

#### Index Types and Transformations
- **IndexPreTransform**: Modular vector transformation framework with standardized API
  - L2NormTransform for unit-length normalization
  - PCATransform for dimensionality reduction
  - RemapDimensionsTransform for feature selection
- **Binary Vector Support**: Comprehensive binary index implementations
  - BinaryIndex base class with efficient Hamming distance calculations
  - IndexBinaryFlat for exact binary vector search
  - IndexBinaryIVF for fast approximate search with inverted file structure
  - IndexBinaryHash for hash-based binary vector lookup
- **Core Index Optimizations**: Enhanced implementations of key index types
  - IndexPQ with robust vector extraction and fallback strategies
  - IndexIVFScalarQuantizer with improved training strategies
  - IndexIDMap and IndexIDMap2 for custom vector IDs with batched operations

#### Advanced Operations
- **Factory Pattern**: Create indices from string descriptions using `index_factory`
- **Index Persistence**: Direct index save/load with `write_index`/`read_index`
- **Index Modification**: Merge and split indices with optimized vector operations
- **Vector Reconstruction**: Perfect floating-point precision reconstruction
  - Single vector reconstruction (`reconstruct`)
  - Batch reconstruction (`reconstruct_n`) for efficiency
- **Vector Caching**: Intelligent caching across all index implementations
- **Batched Processing**: Optimized handling of large vector operations

#### Optimization Controls
- **Search Parameters**: Fine-grained control over nprobe, efSearch, k_factor
- **Training Parameters**: Configurable n_iter, min_points_per_centroid
- **HNSW Parameters**: efConstruction and search-time controls
- **Memory Management**:
  - Memory mapping for large indices
  - Usage limits and tracking with configurable thresholds
  - Automatic unloading of unused indices
  - I/O buffer size controls

### Authentication & Security
- **Multi-Tenant Authentication**: Production-ready authentication system
  - Support for CLI auth-keys format ("key1:tenant1,key2:tenant2")
  - JSON auth-file format support
  - Complete tenant data isolation
  - Comprehensive logging of authentication events

### Reliability & Error Handling
- **Connection Resilience**: Robust error handling and recovery
  - Configurable retry attempts with exponential backoff
  - Automatic reconnection on network failures
  - Connection health monitoring with event callbacks
  - Graceful degradation for server connectivity issues
- **Enhanced Error Recovery**: Improved handling of common error conditions
- **Server Compatibility**: Better compatibility across different server implementations

### Bug Fixes
- **Authentication**: Fixed critical server-side authentication enforcement
- **Vector Reconstruction**: Resolved architectural issues affecting precision
- **API Consistency**: Fixed method inconsistencies in IVF-PQ implementation
- **Training Behavior**: Corrected scalar quantizer training to match FAISS behavior
- **Environment Variables**: Fixed server port configuration handling

### Documentation & Testing
- **Comprehensive Test Suite**: Full coverage of optimized implementations
- **Enhanced Examples**: Extended example suite demonstrating advanced features
- **API Documentation**: Improved documentation for advanced usage patterns
- **Authentication Tests**: Complete test coverage for multi-tenant scenarios

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

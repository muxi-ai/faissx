# Implementation Plan: Binary Indices and IndexPreTransform

This document outlines the implementation plan for adding the remaining FAISS index types to FAISSx for 100% compatibility.

## 1. Binary Indices Implementation

Binary indices work with binary vectors (typically uint8 arrays) and use Hamming distance for similarity calculations.

### File Structure

Create the following new files:
- `faissx/client/indices/binary_flat.py` - Implementation of IndexBinaryFlat
- `faissx/client/indices/binary_ivf.py` - Implementation of IndexBinaryIVF
- `faissx/client/indices/binary_hash.py` - Implementation of IndexBinaryHash
- `faissx/client/tests/indices/test_binary_flat.py` - Tests for binary flat
- `faissx/client/tests/indices/test_binary_ivf.py` - Tests for binary IVF

### Key Components

1. **Base Binary Index Class**:
   - Create a common base class for binary indices
   - Implement Hamming distance calculation for binary vectors
   - Adapt vector storage for binary format (uint8 instead of float32)

2. **Index-Specific Implementations**:
   - IndexBinaryFlat: Basic binary vector index with exact search
   - IndexBinaryIVF: Inverted file index for binary vectors
   - IndexBinaryHash: Hash-based binary index

3. **Server-Side Additions**:
   - Add binary index support to server protocol
   - Implement Hamming distance calculation on server
   - Add binary vector operations to index manager

4. **Testing**:
   - Write comprehensive tests comparing against original FAISS
   - Test edge cases specific to binary vectors
   - Validate both local and remote modes

## 2. IndexPreTransform Implementation

IndexPreTransform applies transformations to vectors before indexing them.

### File Structure

Create the following new files:
- `faissx/client/indices/pre_transform.py` - Implementation of IndexPreTransform
- `faissx/client/transforms.py` - Vector transformation utilities
- `faissx/client/tests/indices/test_pre_transform.py` - Tests for IndexPreTransform

### Key Components

1. **Vector Transformations**:
   - Implement PCA transform
   - Implement L2 normalization
   - Implement vector chunking
   - Support chaining multiple transforms

2. **IndexPreTransform Class**:
   - Create wrapper for applying transforms before using base index
   - Implement reverse transforms for reconstructing original vectors
   - Support serialization of transform parameters

3. **Server-Side Additions**:
   - Add transform support to server protocol
   - Implement transformation operations on server
   - Store transform parameters with indices

4. **Testing**:
   - Test transformations with various index types
   - Verify reconstruction with transforms
   - Compare with original FAISS implementation

## Development Workflow

1. Study original FAISS implementation of these index types
2. Implement client-side code first with local-only support
3. Add server protocol extensions for remote support
4. Implement server-side functionality
5. Write comprehensive tests for both local and remote modes
6. Update documentation with new index types

## Resources Needed

- Original FAISS source code for reference
- Test datasets for binary vectors
- Examples of transform usage from FAISS documentation

## Success Criteria

- All binary indices pass tests against reference FAISS implementation
- IndexPreTransform works with all supported base indices
- Remote mode operates identically to local mode for new indices
- Documentation is complete for all new index types

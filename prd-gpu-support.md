# FAISSx Enhancement: GPU Acceleration Support

## Overview

This document outlines the plan for implementing GPU acceleration in the FAISSx server component using FAISS GPU indices. The enhancement will allow FAISSx to leverage NVIDIA GPUs for significantly faster vector operations while maintaining the existing API contract with clients.

## Problem Statement

Vector similarity search operations are computationally intensive and can become a bottleneck as index sizes and query volumes increase. While FAISS provides GPU acceleration through CUDA, this capability is not currently exposed in the FAISSx server. Adding GPU support would provide:

1. Significantly faster search operations (often 5-10x)
2. Higher throughput for concurrent queries
3. Ability to handle larger indices efficiently
4. Reduced CPU load on server machines

## Goals

1. Enable GPU acceleration for compatible index types in FAISSx server
2. Maintain backward compatibility with existing clients
3. Implement automatic fallback to CPU when GPU is unavailable
4. Add configuration options for GPU resource management
5. Create benchmarks to measure performance improvements

## Non-Goals

1. Adding GPU support for index types not supported by FAISS GPU
2. Modifying the client-side library (it already works in local mode with GPU if available)
3. Supporting non-CUDA GPU platforms (e.g., AMD, Apple Metal)
4. Creating a new API for GPU-specific operations

## Success Criteria

1. 3-10x speedup for search operations on supported index types
2. Automatic GPU detection and usage with no client changes required
3. Graceful fallback to CPU implementation when GPU is unavailable
4. Proper resource management to prevent GPU memory exhaustion
5. Full compatibility with existing client implementations

## Technical Approach

### 1. GPU Detection and Configuration

Add GPU detection at server startup and provide configuration options:

```python
class FaissServer:
    def __init__(self, config):
        # GPU configuration
        self.use_gpu = config.get("use_gpu", True)
        self.gpu_resources = None

        # Initialize GPU resources if available and enabled
        if self.use_gpu:
            try:
                import faiss.contrib.gpu
                ngpus = faiss.get_num_gpus()
                if ngpus > 0:
                    self.gpu_resources = [faiss.StandardGpuResources()
                                         for i in range(ngpus)]
                    self.gpu_indices = {}  # Track GPU indices by ID
                    print(f"Initialized {ngpus} GPUs for FAISS operations")
                else:
                    print("No GPUs detected, falling back to CPU")
            except ImportError:
                print("GPU support not available in FAISS installation")
                self.use_gpu = False
```

### 2. GPU Index Creation

Modify the index creation code to use GPU when available:

```python
def create_index(self, index_id, dimension, index_type="L2"):
    """Create a new index with GPU support if available"""
    if index_id in self.indexes:
        return {"success": False, "error": f"Index {index_id} already exists"}

    try:
        # Parse index parameters and type
        index_params = self._parse_index_params(index_type)
        main_type = index_params["main_type"]

        # Create CPU index first
        cpu_index = self._create_cpu_index(dimension, main_type, index_params)

        # If GPU is available and this index type supports GPU
        if self.use_gpu and self.gpu_resources and self._supports_gpu(main_type):
            try:
                # Choose GPU (simple round-robin allocation)
                gpu_idx = len(self.gpu_indices) % len(self.gpu_resources)
                res = self.gpu_resources[gpu_idx]

                # Create GPU index version
                gpu_index = faiss.index_cpu_to_gpu(res, gpu_idx, cpu_index)

                # Store both versions (GPU used for operations, CPU for persistence)
                self.gpu_indices[index_id] = gpu_index
                self.cpu_indices[index_id] = cpu_index
                index = gpu_index
                using_gpu = True
            except Exception as e:
                # Log error and fall back to CPU
                print(f"GPU index creation failed: {e}, falling back to CPU")
                index = cpu_index
                using_gpu = False
        else:
            # Use CPU version
            index = cpu_index
            using_gpu = False

        # Store dimensions and other metadata
        self.dimensions[index_id] = dimension

        return {
            "success": True,
            "index_id": index_id,
            "dimension": dimension,
            "type": index_type,
            "using_gpu": using_gpu
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 3. GPU-Compatible Index Types

Not all FAISS index types support GPU acceleration. We'll implement support for these index types:

- **Fully GPU compatible**:
  - IndexFlatL2 / IndexFlatIP
  - IndexIVFFlat
  - IndexIVFPQ (with limitations)

- **Partially compatible** (some operations on GPU, some on CPU):
  - IndexIVFScalarQuantizer
  - HNSW (search only)

- **CPU only**:
  - Other specialized indices

```python
def _supports_gpu(self, index_type):
    """Check if an index type supports GPU acceleration"""
    fully_supported = ["L2", "IP", "IVF"]
    partially_supported = ["IVFSQ", "HNSW"]

    if any(x in index_type for x in fully_supported):
        return True
    if any(x in index_type for x in partially_supported):
        return True
    return False
```

### 4. Persistence Management

Since GPU indices can't be directly serialized, we need to maintain CPU versions for persistence:

```python
def save_index(self, index_id):
    """Save index to disk"""
    if index_id not in self.dimensions:
        return {"success": False, "error": f"Index {index_id} does not exist"}

    try:
        # Always save the CPU version
        index_to_save = self.cpu_indices.get(index_id, self.indexes.get(index_id))

        if not index_to_save:
            return {"success": False, "error": "Index not found"}

        # If this is a GPU index, we need to copy back to CPU first
        if index_id in self.gpu_indices:
            # Convert GPU index back to CPU for saving
            index_to_save = faiss.index_gpu_to_cpu(self.gpu_indices[index_id])

        # Continue with normal save logic...
        path = self._get_index_path(index_id)
        faiss.write_index(index_to_save, path)

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 5. Dynamic GPU Memory Management

Implement memory monitoring and management to avoid out-of-memory errors:

```python
def _check_gpu_memory(self, gpu_idx, required_mb=None):
    """Check if GPU has enough free memory"""
    try:
        # This requires pycuda or similar to implement
        free_memory = self._get_gpu_free_memory(gpu_idx)

        if required_mb and free_memory < required_mb * 1024 * 1024:
            return False

        # If memory is critically low, move an index to CPU
        if free_memory < 500 * 1024 * 1024:  # Less than 500MB free
            self._offload_index_to_cpu()

        return True
    except Exception:
        # If monitoring fails, assume it's okay
        return True

def _offload_index_to_cpu(self):
    """Move least recently used GPU index to CPU"""
    # Find least recently accessed index
    lru_index_id = self._find_lru_gpu_index()

    if lru_index_id:
        # Move from GPU to CPU
        cpu_index = faiss.index_gpu_to_cpu(self.gpu_indices[lru_index_id])
        self.indexes[lru_index_id] = cpu_index
        del self.gpu_indices[lru_index_id]
        del self.cpu_indices[lru_index_id]
```

### 6. Configuration Options

Add server configuration options for GPU usage:

```python
# Server configuration (in config.py or similar)
GPU_CONFIG_OPTIONS = {
    "use_gpu": {
        "type": bool,
        "default": True,
        "description": "Enable GPU acceleration if available"
    },
    "gpu_memory_fraction": {
        "type": float,
        "default": 0.8,
        "description": "Fraction of GPU memory to use (0.0-1.0)"
    },
    "preferred_gpu_indices": {
        "type": list,
        "default": [0, 1, 2, 3],  # Order of GPU preference
        "description": "Preferred order of GPUs to use"
    },
    "max_indices_per_gpu": {
        "type": int,
        "default": 5,
        "description": "Maximum number of indices to store on each GPU"
    }
}
```

## Implementation Phases

### Phase 1: Basic GPU Support

1. Implement GPU resource initialization
2. Add support for FlatL2/FlatIP indices on GPU
3. Implement proper persistence with CPU copies
4. Add basic configuration options
5. Create benchmarking tools

### Phase 2: Advanced GPU Support

1. Add support for IVF indices on GPU
2. Implement dynamic memory management
3. Add multi-GPU support
4. Create monitoring endpoints for GPU utilization
5. Improve configuration options

### Phase 3: Optimization and Edge Cases

1. Optimize for different index sizes and types
2. Add advanced memory management strategies
3. Implement hybrid operations for partially supported indices
4. Create comprehensive benchmarking suite
5. Add documentation and usage guidelines

## Backward Compatibility

All changes will be transparent to clients:

1. Existing API endpoints remain unchanged
2. Clients will automatically benefit from GPU acceleration
3. Server can run with or without GPU hardware
4. Persistence format remains compatible with previous versions

## Testing Strategy

1. Unit tests for GPU index creation and operations
2. Performance comparisons with CPU-only implementation
3. Memory management stress tests
4. Failover scenarios (GPU error, out of memory, etc.)
5. Multi-client concurrent access patterns

## Benchmark Methodology

Create a benchmarking suite to measure performance gain:

1. Compare search times for different index types and sizes
2. Test with varying number of query vectors
3. Measure throughput under concurrent client load
4. Compare memory usage patterns
5. Test with different GPU hardware configurations

Example benchmark scenarios:
- 1M vectors, dimensions: 128, 384, 768, 1536
- Batch sizes: 1, 10, 100, 1000
- Index types: Flat, IVF, HNSW
- Concurrent clients: 1, 5, 10, 50

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| High memory usage on GPU | Implement dynamic offloading to CPU |
| Not all index types support GPU | Clearly document support limitations |
| GPU errors during runtime | Graceful fallback to CPU implementation |
| Performance varies by GPU model | Benchmark on representative hardware |
| CUDA compatibility issues | Test on different CUDA versions |

## Requirements

Server-side requirements:
1. NVIDIA GPU with CUDA support
2. FAISS built with GPU support
3. Adequate GPU memory for indices

## Future Considerations

1. Support for newer GPU architectures and features
2. Optimized hybrid CPU/GPU operations
3. Dynamic workload balancing across multiple GPUs
4. Support for other GPU platforms (AMD, Intel) if FAISS adds support
5. Integration with GPU monitoring and management tools

## Timeline

- Phase 1: 2 weeks
- Phase 2: 3 weeks
- Phase 3: 2 weeks
- Testing and benchmarking: 2 weeks
- Documentation and release: 1 week
- Total: ~10 weeks

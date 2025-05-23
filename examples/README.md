# FAISSx Examples

This directory contains various examples demonstrating different aspects of FAISSx functionality.

## Available Examples

### Basic Usage
- **`simple_client.py`** - Basic client usage connecting to a remote server
- **`server_example.py`** - Shows how to configure and run a FAISSx server

### Index Types
- **`ivf_index_example.py`** - Performance comparison of Flat vs IVF indices
- **`hnsw_and_pq_example.py`** - Advanced indices (HNSW, PQ) with memory and performance analysis

### Performance & Optimization
- **`batch_operations_example.py`** - Performance optimization with batched operations
- **`optimization_example.py`** - Various optimization techniques and performance tuning

### Advanced Operations
- **`index_modification_example.py`** - Index merge and split operations
- **`recovery_example.py`** - Error recovery, reconnection, and resilience features

## Running the Examples

### Prerequisites

1. **Start the FAISSx server**:
   ```bash
   faissx.server run --port 45678
   ```

2. **Verify server is running**:
   ```bash
   lsof -i :45678
   ```

### Running Examples

Navigate to the examples directory and run any example:

```bash
cd examples

# Basic usage
python simple_client.py
python server_example.py

# Index types
python ivf_index_example.py
python hnsw_and_pq_example.py

# Performance
python batch_operations_example.py
python optimization_example.py

# Advanced operations
python index_modification_example.py
python recovery_example.py
```

## Example Requirements

All examples require:
- FAISSx server running on port 45678
- Python packages: `numpy`, `time`, `logging`
- Some examples may require additional packages as noted in their docstrings

## Getting Help

Each example includes detailed comments and documentation. Run any example to see its output and learn about the demonstrated functionality.

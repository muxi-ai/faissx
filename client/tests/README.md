# FAISSx Client Tests

This directory contains tests for the FAISSx client library.

## Directory Structure

- `indices/` - Tests for specific index implementations
  - `test_pq_optimized.py` - Tests for Product Quantization (PQ) indices
  - `test_ivf_sq_optimized.py` - Tests for IVF Scalar Quantizer indices
  - `test_ivf_pq_optimized.py` - Tests for IVF PQ indices
  - `test_scalar_quantizer.py` - Tests for Scalar Quantizer indices
  - `test_scalar_quantizer_optimized.py` - Tests for optimized Scalar Quantizer indices
  - `test_flat_index.py` - Tests for Flat indices
  - `test_hnsw_modes.py` - Tests for HNSW indices in different modes

- `test_optimized_implementations.py` - Comprehensive tests for optimized implementations
- `test_optimized_client.py` - Tests for the optimized client implementation
- `test_io.py` - Tests for index persistence functionality
- `test_idmap_simple.py` - Tests for IDMap indices
- `test_timeout_comprehensive.py` - Comprehensive timeout tests
- `test_timeout_global.py` - Global timeout tests
- `test_timeout.py` - Basic timeout tests
- `test_dynamic_timeout.py` - Dynamic timeout tests
- `direct_remote_test.py` - Tests for direct remote mode
- `simple.py` - Simple client test

## How to Run Tests

To run all client tests:

```bash
cd client/tests
pytest
```

To run a specific test:

```bash
cd client/tests
pytest test_io.py
```

To run tests for a specific index:

```bash
cd client/tests
pytest indices/test_pq_optimized.py
```

## Remote Testing

Some tests require a running FAISSx server. To run these tests:

1. Start the server:
   ```bash
   faissx.server run
   ```

2. Run the tests with the `FAISSX_SERVER` environment variable:
   ```bash
   FAISSX_SERVER=tcp://localhost:45678 pytest
   ```

## Test Groups

- **Index Tests**: Tests for specific index implementations (in the `indices/` directory)
- **Optimization Tests**: Tests for optimized implementations (`test_optimized_implementations.py`, `test_optimized_client.py`)
- **Persistence Tests**: Tests for saving and loading indices (`test_io.py`)
- **Timeout Tests**: Tests for timeout handling (`test_timeout*.py`)
- **Mode Tests**: Tests for different operation modes (local, remote with/without auth)

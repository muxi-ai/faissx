# FAISSx Local Mode Tests

This directory contains tests for verifying that FAISSx works properly as a drop-in replacement for FAISS in local mode.

## Purpose

These tests ensure that the FAISSx client can seamlessly replace FAISS with minimal or no code changes when used with:

```python
from faissx import client as faiss
```

The tests verify functionality when no remote server is configured, causing FAISSx to operate in local mode where it directly delegates to the original FAISS library.

## Test Structure

### Core Tests

- **test_index_flat.py**: Tests `IndexFlatL2` for basic vector insertion and search
- **test_index_ivf_flat.py**: Tests `IndexIVFFlat` for inverted file index operations
- **test_index_idmap.py**: Tests `IndexIDMap` and `IndexIDMap2` for custom vector IDs
- **test_factory.py**: Tests `index_factory` for creating various index types
- **test_persistence.py**: Tests saving and loading indices with `write_index`/`read_index`

### Helper Modules

- **fix_imports.py**: Provides constants and imports necessary for tests
- **run_all_tests.py**: Script to run all tests in sequence with proper reporting
- **__main__.py**: Makes the package runnable with `python -m`

## Running Tests

Run all tests with:

```bash
python run_all_tests.py
```

Run a specific test module:

```bash
python -m unittest test_index_flat
```

## Implementation Notes

1. These tests focus on verifying API compatibility between FAISSx and FAISS
2. Some advanced index types are currently skipped due to implementation complexity
3. Inner product metrics are implemented using a custom `IndexFlatIP` class
4. Environment variables are cleared before each test to ensure local mode is used

## Test Status

| Test Category | Status | Notes |
|--------------|--------|-------|
| IndexFlatL2  | ✅ Pass | All core functionality tested |
| IndexFactory | ✅ Pass | Basic factory tests pass |
| IndexIVFFlat | ✅ Pass | All tests now passing properly |
| IndexIDMap   | ✅ Pass | Basic tests pass, some advanced features skipped |
| Persistence  | ✅ Pass | Basic tests pass with custom implementation |

## Future Work

- Implement any remaining advanced features
- Add more comprehensive error handling tests
- Add benchmarking to compare performance

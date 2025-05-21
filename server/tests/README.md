# FAISSx Server Tests

This directory contains tests for the FAISSx server.

## Test Files

- `simple_test.py` - Basic tests for server functionality
- `simple_server_test.py` - Tests for server setup and operations

## How to Run Tests

To run all server tests:

```bash
cd server/tests
pytest
```

To run a specific test:

```bash
cd server/tests
pytest simple_test.py
```

## Test Dependencies

These tests require:
1. A running FAISSx server
2. The client library to be installed

To start the server:

```bash
faissx.server run
```

## Test Coverage

The server tests cover:
- Server initialization
- Index creation and management
- Vector operations (add, search)
- Authentication (when enabled)
- Multi-tenant isolation (when relevant)

## Future Test Improvements

Future tests will add coverage for:
- Server resilience and error handling
- Performance benchmarking
- High-load scenarios
- Vector persistence

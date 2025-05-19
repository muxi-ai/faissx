# Server Directory Reorganization

We've restructured the server directory to better organize the code and separate core functionality from deployment files. We've also shifted to a binary-only distribution strategy, removing pip installation support and unifying our Docker and standalone distribution around a single binary artifact.

## Directory Structure

```
server/
├── src/                        # Core package
│   ├── __init__.py            # Package initialization
│   ├── auth.py                # Authentication
│   ├── faiss_core.py          # FAISS operations
│   ├── protocol.py            # ZeroMQ protocol
│   ├── run.py                 # Server implementation
│   └── test_zmq_client.py     # Test client
├── tests/                      # Tests
│   ├── test_zmq/              # ZeroMQ tests
│   │   └── test_basic.py      # Basic functionality tests
│   └── conftest_zmq.py        # Test fixtures
├── Dockerfile                  # Multi-stage Docker build
├── dev-requirements.txt        # Developer dependencies
├── pyoxidizer.bzl              # PyOxidizer configuration
├── test_client.py              # Test client entrypoint
└── run_zmq_tests.sh            # Test runner
```

## Changes Made

1. **Created package structure**:
   - Moved core files to `server/` directory
   - Added package `__init__.py`
   - Updated imports to use the new package structure

2. **Simplified entrypoints**:
   - Removed top-level run.py script
   - Using direct module execution for development
   - Single binary artifact for all production deployments

3. **Updated distribution strategy**:
   - Removed pip/setup.py installation option
   - Renamed requirements.txt to dev-requirements.txt for development only
   - Focus on consistent binary distribution via PyOxidizer
   - Implemented multi-stage Docker build that uses the same binary

4. **Updated test structure**:
   - Adjusted test path imports
   - Updated test runner to use the new structure

## Running the Server

```bash
# Option 1: Using direct module execution (development)
python -m src.run

# Option 2: Using Docker (runs the binary in Alpine)
docker build -t faiss-proxy-server -f Dockerfile .
docker run -p 5555:5555 faiss-proxy-server

# Option 3: Using PyOxidizer binary directly
pyoxidizer build --release
./build/*/release/install/faiss-proxy-server
```

## Running the Test Client

```bash
# Option 1: Using the entrypoint script
./test_client.py

# Option 2: Using Python module
python -m src.test_zmq_client
```

## Running Tests

```bash
./run_zmq_tests.sh
```

## Docker Build

```bash
docker build -t faiss-proxy-server -f Dockerfile .
```

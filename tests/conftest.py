#!/usr/bin/env python3
#
# Pytest configuration and fixtures for FAISSx tests
# https://github.com/muxi-ai/faissx
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pytest configuration and fixtures for FAISSx tests
"""

import logging
import os
import socket
import subprocess
import sys
import time

import pytest

from faissx.client.client import FaissXClient


def _free_port():
    """Ask the OS for a free TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _port_open(port):
    """Return True if something is listening on 127.0.0.1:port."""
    with socket.socket() as sock:
        sock.settimeout(0.2)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


# Resolve the test server address once, at import time, so test modules can
# read it at module level via FAISSX_TEST_SERVER. An externally provided
# FAISSX_SERVER wins; otherwise the session-scoped fixture below starts a
# server on a dynamically allocated port (hermetic - never reuses whatever
# happens to be listening on the default port). Deliberately NOT exported as
# FAISSX_SERVER: the client library reads that variable and would silently
# switch every local-mode test into remote mode.
_EXTERNAL_SERVER = os.environ.get("FAISSX_SERVER")
TEST_SERVER_PORT = _free_port()
TEST_SERVER_ADDRESS = _EXTERNAL_SERVER or f"tcp://127.0.0.1:{TEST_SERVER_PORT}"
os.environ["FAISSX_TEST_SERVER"] = TEST_SERVER_ADDRESS

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Index types for parametrized testing
INDEX_TYPES = [
    # Basic types
    "L2",         # Flat L2 index
    "IP",         # Flat IP index
    "BINARY_FLAT",  # Binary flat index

    # Quantization types
    "PQ4x2",      # Product Quantization with 4 subquantizers, 2 bits each
    "PQ4",        # Product Quantization with 4 subquantizers

    # IVF types
    "IVF16",       # IVF index with 16 clusters
    "IVF16_IP",    # IVF index with IP distance
    "IVF4_SQ8",    # IVF index with 4 clusters and 8-bit scalar quantization

    # Transformation types
    "OPQ4_8,L2",   # OPQ transformation + L2 index
    "PCA4,L2",     # PCA transformation + L2 index
    "NORM,L2",     # L2 normalization + L2 index

    # HNSW types
    "HNSW32",      # HNSW index with 32 neighbors per node
    "HNSW16_IP",   # HNSW index with IP distance

    # ID mapping types
    "IDMap:L2",    # IDMap with L2 flat index
    "IDMap2:L2",   # IDMap2 with L2 flat index
]


@pytest.fixture(scope="session", autouse=True)
def faissx_server():
    """
    Session-scoped FAISSx server for tests that need the remote API.

    Starts the current source tree's server as a subprocess on a dynamically
    allocated port and exposes it via the FAISSX_SERVER environment variable.
    When FAISSX_SERVER was already set externally, that server is used instead
    and nothing is started. Yields the server address.
    """
    if _EXTERNAL_SERVER:
        yield _EXTERNAL_SERVER
        return

    address = TEST_SERVER_ADDRESS

    process = subprocess.Popen(
        [
            sys.executable, "-m", "faissx.server.cli", "run",
            "--port", str(TEST_SERVER_PORT),
            "--log-level", "WARNING",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        if _port_open(TEST_SERVER_PORT):
            break
        if process.poll() is not None:
            raise RuntimeError("FAISSx test server exited during startup")
        time.sleep(0.1)
    else:
        process.terminate()
        raise RuntimeError("FAISSx test server did not start listening in time")

    yield address

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture(autouse=True)
def _reset_client_singleton():
    """
    Reset the global client singleton after every test.

    Tests that call faiss.configure() switch the shared singleton to remote
    mode; without a reset, later tests that expect local mode inherit that
    state and hit the network instead.
    """
    yield
    from faissx.client import client as client_module
    if client_module._client is not None:
        try:
            client_module._client.disconnect()
        except Exception:
            pass
        client_module._client = None


@pytest.fixture(scope="function")
def client(faissx_server):
    """
    Create a FaissXClient instance connected to a server.

    Uses the server at FAISSX_SERVER when that environment variable is set;
    otherwise connects to the session-scoped test server. Tests that need the
    raw client API (create_index, add_vectors, search, ...) require this
    connection - those methods are remote-only.
    """
    client = FaissXClient()
    client.configure(server=faissx_server)

    yield client

    # Cleanup after test
    try:
        client.disconnect()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture(params=INDEX_TYPES)
def index_type(request):
    """
    Parametrized fixture for testing different index types.
    """
    return request.param

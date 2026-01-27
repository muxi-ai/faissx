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

import pytest

from faissx.client.client import FaissXClient

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


@pytest.fixture(scope="function")
def client():
    """
    Create a FaissXClient instance for testing.

    This fixture checks for FAISSX_SERVER environment variable to determine
    whether to run in local or remote mode.
    """
    client = FaissXClient()

    # Configure client based on environment
    server_addr = os.environ.get('FAISSX_SERVER')
    if server_addr:
        # Remote mode - connect to specified server
        client.configure(server=server_addr)
        client.connect()
    else:
        # Local mode - use embedded server
        pass  # Client will run in local mode by default

    yield client

    # Cleanup after test
    try:
        if hasattr(client, 'close'):
            client.close()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture(params=INDEX_TYPES)
def index_type(request):
    """
    Parametrized fixture for testing different index types.
    """
    return request.param

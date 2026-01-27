#!/usr/bin/env python3
#
# Unified interface for LLM providers using OpenAI format
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
FAISSx Package - High-performance vector database proxy using ZeroMQ

This package provides a complete solution for distributed vector search operations
using Facebook AI Similarity Search (FAISS) over ZeroMQ for high-performance communication.

Key components:

1. Server Module (faissx.server):
   - Standalone service that manages FAISS indices
   - Multi-tenant isolation for shared deployments
   - Authentication with API keys
   - Persistent storage for indices
   - Binary protocol for efficient data transfer
   - Command-line interface for easy deployment

2. Client Module (faissx.client):
   - Drop-in replacement for FAISS with identical API
   - Transparent remote execution of vector operations
   - Support for standard FAISS index types
   - Efficient binary serialization of vector data
   - Authentication and tenant isolation
   - Local fallback capabilities

3. Protocol:
   - Zero-copy binary messaging for maximum performance
   - MessagePack-based serialization for structured data
   - Optimized for large vector datasets

The package can be used in both standalone server mode and as a client library.
"""

import os

# Import key client components for easy access
# These imports are at the module level instead of the end of the file
# to ensure they're available in the global namespace
from .client.index import (  # noqa: F401
    IndexFlatL2,
    IndexHNSWFlat,
    IndexIDMap,
    IndexIDMap2,
    IndexIVFFlat,
    IndexIVFPQ,
    IndexPQ,
    IndexScalarQuantizer,
)

# Define public API
__all__ = [
    'configure',
    'IndexFlatL2',
    'IndexIVFFlat',
    'IndexHNSWFlat',
    'IndexPQ',
    'IndexIVFPQ',
    'IndexScalarQuantizer',
    'IndexIDMap',
    'IndexIDMap2',
]


def get_version() -> str:
    """
    Read and return the package version from the .version file.

    Returns:
        str: The current version of the package

    Note:
        The .version file should be located in the same directory as this file.
        The version string is stripped of any whitespace to ensure clean formatting.
    """
    version_file = os.path.join(os.path.dirname(__file__), ".version")
    with open(version_file, encoding="utf-8") as f:
        return f.read().strip()


# Initialize package version from .version file
__version__ = get_version()

# Package metadata for distribution and documentation
__author__ = "Ran Aroussi"  # Primary package author
__license__ = "Apache-2.0"  # Open source license
__url__ = "https://github.com/muxi-ai/faissx"  # Source code repository

# Global configuration settings
_API_URL = "tcp://localhost:45678"  # Default server URL
_API_KEY = None  # Default: No API key (for local development)
_TENANT_ID = "default"  # Default tenant ID for multi-tenant deployments
_CONNECTION_TIMEOUT = 10000  # Default connection timeout in milliseconds
_REQUEST_TIMEOUT = 30000  # Default request timeout in milliseconds


def configure(
    url: str = None,
    api_key: str = None,
    tenant_id: str = None,
    connection_timeout: int = None,
    request_timeout: int = None
) -> None:
    """
    Configure the FAISSx client for remote server connection.

    This function should be called before using any FAISSx client functionality
    to establish the connection parameters for the remote FAISS server.

    Args:
        url (str, optional): ZeroMQ URL for the FAISS server
            (default: "tcp://localhost:45678")
        api_key (str, optional): API key for server authentication
            (default: None, no authentication)
        tenant_id (str, optional): Tenant ID for multi-tenant deployments
            (default: "default")
        connection_timeout (int, optional): Connection timeout in milliseconds
            (default: 10000)
        request_timeout (int, optional): Request timeout in milliseconds
            (default: 30000)

    Example:
        >>> import faissx
        >>> faissx.configure(url="tcp://faiss-server:45678", api_key="my-api-key")
        >>> from faissx.client.index import IndexFlatL2
        >>> index = IndexFlatL2(128)  # Now uses remote server
    """
    global _API_URL, _API_KEY, _TENANT_ID, _CONNECTION_TIMEOUT, _REQUEST_TIMEOUT

    if url is not None:
        _API_URL = url
    if api_key is not None:
        _API_KEY = api_key
    if tenant_id is not None:
        _TENANT_ID = tenant_id
    if connection_timeout is not None:
        _CONNECTION_TIMEOUT = connection_timeout
    if request_timeout is not None:
        _REQUEST_TIMEOUT = request_timeout

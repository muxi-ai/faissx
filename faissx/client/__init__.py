#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
FAISSx Client Module
A drop-in replacement for FAISS with remote execution capabilities

This module provides a client interface for FAISSx that implements the
FAISS API, with two distinct modes of operation:

1. Local Mode (Default): Uses the local FAISS library, making it a true
   drop-in replacement without any configuration required.

2. Remote Mode: Activated by calling configure(), this mode executes
   operations remotely via ZeroMQ, offering enhanced scalability.

Key features include:

1. True drop-in API compatibility with the original FAISS library
2. Local FAISS by default for simplicity with no configuration required
3. Optional remote execution of vector operations when configured
4. Environment-based or programmatic configuration
5. Support for authentication and multi-tenant isolation
6. Index implementations that mirror the behavior of native FAISS indices

Users can switch between local FAISS processing and remote FAISSx execution
by simply adding a configure() call before creating any indices.
"""

import os
import sys
from typing import Optional

# Import core client functionality
from .client import configure, FaissXClient, get_client

# Define a ScalarQuantizer class to match FAISS API
class ScalarQuantizer:
    """
    Provides the same scalar quantizer types as in FAISS.
    """
    QT_8bit = 1  # 8 bits per component
    QT_4bit = 2  # 4 bits per component
    QT_8bit_uniform = 3  # 8 bits per component, uniform
    QT_4bit_uniform = 4  # 4 bits per component, uniform
    QT_fp16 = 5  # half-precision float
    QT_8bit_direct = 6  # 8 bits direct embedded
    QT_6bit = 7  # 6 bits per component

# Import index classes directly to expose them at the module level
from .indices import (
    IndexFlatL2,
    IndexIVFFlat,
    IndexHNSWFlat,
    IndexPQ,
    IndexIVFPQ,
    IndexScalarQuantizer,
    IndexIVFScalarQuantizer,
    IndexIDMap,
    IndexIDMap2,
    index_factory,
    write_index,
    read_index
)

# # Global configuration variables with environment variable fallbacks
# # These control the client's connection and authentication settings
# _API_URL: Optional[str] = os.environ.get(
#     "FAISSX_SERVER", ""  # Default to empty, which enables local mode
# )
# _API_KEY: Optional[str] = os.environ.get("FAISSX_API_KEY", "")  # API key for authentication
# _TENANT_ID: Optional[str] = os.environ.get("FAISSX_TENANT_ID", "")  # Tenant ID for multi-tenancy

# Attempt to import local FAISS for local mode
try:
    # Import FAISS locally for local mode when remote server is not configured
    import faiss as _local_faiss
except ImportError:
    _local_faiss = None
    print(
        "Warning: Local FAISS not found, only remote mode will be available", file=sys.stderr
    )

# Define public API - symbols that can be imported directly from the module
__all__ = [
    "configure",  # Client configuration function
    "FaissXClient",  # Main client class
    "get_client",  # Client instance getter
    "IndexFlatL2",  # L2 distance index implementation
    "IndexIVFFlat",  # IVF flat index implementation
    "IndexHNSWFlat",  # HNSW flat index implementation
    "IndexPQ",  # PQ index implementation
    "IndexIVFPQ",  # IVF PQ index implementation
    "IndexScalarQuantizer",  # Scalar quantizer index implementation
    "IndexIVFScalarQuantizer",  # IVF Scalar quantizer index implementation
    "IndexIDMap",  # ID mapping index
    "IndexIDMap2",  # Extended ID mapping index
    "index_factory",  # Factory function for creating indices
    "write_index",  # Write index to disk
    "read_index",  # Read index from disk
    "ScalarQuantizer",  # Scalar quantizer constants
]

# Set version to match local FAISS version for compatibility
# Fallback to a default version if local FAISS is not available
__version__ = _local_faiss.__version__ if _local_faiss else "1.7.0"

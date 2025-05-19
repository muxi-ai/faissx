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
FAISSx - A drop-in replacement for FAISS with remote execution capabilities via ZeroMQ

This module provides a client interface for FAISSx, allowing vector operations to be
executed remotely via ZeroMQ. It includes fallback capabilities to local FAISS when needed.
"""

import os
import sys
from typing import Optional

# Module version
__version__ = "0.1.0"

# Global configuration variables with environment variable fallbacks
# These control the client's connection and authentication settings
_API_URL: Optional[str] = os.environ.get("FAISSX_SERVER", "tcp://localhost:45678")  # Default to localhost
_API_KEY: Optional[str] = os.environ.get("FAISSX_API_KEY", "")  # API key for authentication
_TENANT_ID: Optional[str] = os.environ.get("FAISSX_TENANT_ID", "")  # Tenant ID for multi-tenancy
_FALLBACK_TO_LOCAL: bool = os.environ.get("FAISSX_FALLBACK_TO_LOCAL", "1") == "1"  # Enable local FAISS fallback by default

# Attempt to import local FAISS for fallback capabilities
try:
    # Import FAISS locally to enable fallback mode when remote server is unavailable
    import faiss as _local_faiss
except ImportError:
    _local_faiss = None
    if _FALLBACK_TO_LOCAL:
        print(
            "Warning: Local FAISS not found, can't use fallback mode", file=sys.stderr
        )
        _FALLBACK_TO_LOCAL = False

# Import core client functionality
from .client import configure, FaissXClient, get_client
from .index import (
    IndexFlatL2,  # L2 distance-based index implementation
    # Add other index types as they are implemented
)

# Define public API - symbols that can be imported directly from the module
__all__ = [
    "configure",  # Client configuration function
    "FaissXClient",  # Main client class
    "get_client",  # Client instance getter
    "IndexFlatL2",  # L2 distance index implementation
    # Add other exported symbols as they are implemented
]

# Set version to match local FAISS version for compatibility
# Fallback to a default version if local FAISS is not available
__version__ = _local_faiss.__version__ if _local_faiss else "1.7.0"

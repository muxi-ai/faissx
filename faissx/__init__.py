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
    with open(version_file, "r", encoding="utf-8") as f:
        return f.read().strip()


# Initialize package version from .version file
__version__ = get_version()

# Package metadata for distribution and documentation
__author__ = "Ran Aroussi"  # Primary package author
__license__ = "Apache-2.0"  # Open source license
__url__ = "https://github.com/muxi-ai/faissx"  # Source code repository

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
FAISSx package - High-performance vector database proxy using ZeroMQ.

This package provides both server and client implementations for using
FAISS (Facebook AI Similarity Search) over ZeroMQ for high performance.

The package consists of:
- Server: A standalone service that manages FAISS indices and handles vector operations
- Client: A Python client library that provides a FAISS-compatible interface
- Protocol: A ZeroMQ-based communication protocol for client-server interaction
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Core Module Bridge
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
FAISSx Core Compatibility Module

This module re-exports functionality from faiss_core.py to maintain
compatibility with existing code that imports from .core
"""

# Re-export core functionality needed by server.py
from .faiss_core import IndexID

# Define missing types
import numpy as np
from typing import Dict, List, Any, Optional, Union, TypedDict

# Type definitions for vector data
VectorData = Union[List[float], np.ndarray]  # Vector data as list or numpy array
QueryParams = Dict[str, Any]  # Parameters for search queries

# Results structure for search operations
class SearchResult(TypedDict):
    """Type definition for search results"""
    indices: List[int]
    distances: List[float]


# Constants
DEFAULT_PORT = 45678
DEFAULT_HOST = "localhost"
DEFAULT_K = 10
DEFAULT_TIMEOUT = 30


# Helper function that might be expected from core
def create_index_from_type(index_type, dimension, metric="L2", metadata=None):
    """
    Create an index from a type string specification.

    Args:
        index_type: Type of index to create
        dimension: Vector dimension
        metric: Distance metric (L2 or IP)
        metadata: Optional metadata for the index

    Returns:
        FAISS index instance
    """
    # Import here to avoid circular imports
    from .transformations import create_index_from_type as create_from_transform

    try:
        # Try the implementation from transformations
        return create_from_transform(index_type, dimension, metric, metadata)
    except Exception:
        # Fall back to a simpler implementation
        import faiss

        if index_type == "L2" or index_type == "FLAT":
            return faiss.IndexFlatL2(dimension), {"type": "FLAT", "metric": "L2"}
        elif index_type == "IP":
            return faiss.IndexFlatIP(dimension), {"type": "FLAT", "metric": "IP"}
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

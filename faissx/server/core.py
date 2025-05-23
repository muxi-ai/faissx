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

This module serves as a compatibility bridge between the main server components and the
core FAISS functionality. It re-exports essential functionality from faiss_core.py while
providing additional type definitions and helper functions needed by the server.

Key Responsibilities:
- Type definitions for vector operations and search results
- Constants for default server configuration
- Helper functions for index creation and management
- Compatibility layer for legacy code importing from .core

Design Rationale:
This module exists to maintain a clean separation between core FAISS operations and
server-specific functionality while providing backward compatibility for existing
code that expects to import from .core.

Usage:
    from faissx.server.core import VectorData, SearchResult, create_index_from_type
"""

# Standard library imports
# NumPy is essential for efficient vector operations and FAISS integration
import numpy as np  # Numerical operations and array handling

# Type hints improve code readability and enable static analysis
# Multi-line import for better readability and easier maintenance
from typing import (
    Dict,       # Dictionary type for key-value mappings
    List,       # List type for sequences
    Any,        # Any type for flexible typing when specific type is unknown
    Union,      # Union type for values that can be one of several types
    TypedDict,  # TypedDict for structured dictionary types with known keys
    Tuple,      # Tuple type for fixed-length sequences
)  # Type hints for better code safety

# Internal imports - Re-export core functionality needed by server.py
# IndexID is imported from faiss_core to maintain module boundaries
from .faiss_core import IndexID  # Index identifier type from core module

# Module exports definition for explicit public API control
# __all__ defines what gets imported when using "from module import *"
# This provides a clean public interface and prevents accidental imports
__all__ = [
    "IndexID",                 # Re-exported from faiss_core for compatibility
    "VectorData",              # Type alias for vector data representation
    "QueryParams",             # Type alias for search query parameters
    "IndexMetadata",           # Type alias for index metadata dictionaries
    "SearchResult",            # TypedDict for structured search results
    "DEFAULT_PORT",            # Default network port for server communication
    "DEFAULT_HOST",            # Default hostname for client connections
    "DEFAULT_K",               # Default number of nearest neighbors to return
    "DEFAULT_TIMEOUT",         # Default timeout for operations
    "create_index_from_type",  # Factory function for index creation
]

# Type definitions for vector data and operations
# These types provide clear interfaces for vector operations throughout the system
# Type aliases improve code readability and make it easier to change implementations

# VectorData represents the two primary ways vectors are passed to the system
# List[float] is convenient for simple cases and JSON serialization
# np.ndarray is more efficient for large datasets and mathematical operations
VectorData = Union[List[float], np.ndarray]  # Vector data as list or numpy array

# QueryParams allows flexible search configuration through key-value pairs
# Common parameters include: k (number of results), nprobe (search quality), etc.
QueryParams = Dict[str, Any]  # Parameters for search queries

# IndexMetadata stores configuration and status information about indices
# Typical contents: type, dimension, metric, training status, etc.
IndexMetadata = Dict[str, Any]  # Metadata associated with indices


# Results structure for search operations
# This TypedDict ensures consistent result format across all search operations
# Using TypedDict provides type safety while maintaining dictionary-like behavior
class SearchResult(TypedDict):
    """
    Type definition for search results returned by vector similarity operations.

    This structure standardizes the format of search results across all vector
    similarity operations, ensuring consistent interfaces for client code.

    Attributes:
        indices: List of indices of the most similar vectors found in the dataset
        distances: List of distance scores corresponding to each found vector

    Important Notes:
        - Both lists have the same length and correspond element-wise
        - indices[i] represents the dataset index with distance distances[i]
        - Distance interpretation depends on metric (lower=similar for L2)
        - Indices refer to positions in the original vector dataset

    Example:
        {
            'indices': [42, 17, 93],
            'distances': [0.1, 0.3, 0.7]
        }
        This means vector at index 42 has distance 0.1 (most similar).
    """

    # Index values pointing to vectors in the original dataset
    # These are zero-based positions that can be used to retrieve original vectors
    indices: List[int]  # Indices of similar vectors in the original dataset

    # Distance values corresponding to each index
    # Lower values indicate higher similarity for L2 distance
    # Higher values indicate higher similarity for Inner Product distance
    distances: List[float]  # Distance scores (interpretation depends on metric used)


# Configuration constants for server operations
# These constants provide sensible defaults for server configuration and operations
# Using constants improves maintainability and prevents magic numbers throughout code

# Network configuration constants
# Port 45678 is chosen to avoid conflicts with common services
# This port is in the user/private range and unlikely to conflict with system services
DEFAULT_PORT = 45678  # Default ZeroMQ port for server communication

# Host configuration for client connections
# "localhost" restricts connections to local machine for security
# Change to "0.0.0.0" to accept connections from any interface
DEFAULT_HOST = "localhost"  # Default host for client connections

# Search operation defaults
# K=10 provides a good balance between result quality and performance
# Most applications find 10 results sufficient for similarity search
DEFAULT_K = 10  # Default number of nearest neighbors to return

# Timeout configuration for robust operation
# 30 seconds allows for complex operations while preventing indefinite waits
# This timeout applies to both network operations and index operations
DEFAULT_TIMEOUT = 30  # Default timeout in seconds for operations


# Helper functions for index creation and management
# These functions provide a convenient interface for creating FAISS indices


def create_index_from_type(
    index_type: str, dimension: int, metric: str = "L2", metadata: IndexMetadata = None
) -> Tuple[Any, IndexMetadata]:
    """
    Create a FAISS index from a type string specification.

    This function serves as a factory for creating FAISS indices with proper
    configuration. It implements a two-tier approach: first attempting to use
    the full implementation from transformations module for advanced features,
    then falling back to a simpler implementation for basic index types.

    Design Pattern:
        This factory pattern abstracts index creation complexity and provides
        a unified interface for all index types while maintaining extensibility.

    Supported Index Types:
        - "L2" or "FLAT": Flat index with L2 (Euclidean) distance
        - "IP": Flat index with Inner Product distance
        - Additional types supported via transformations module

    Args:
        index_type: String specifying the type of index (e.g., "L2", "IP", "IVF100")
        dimension: Number of dimensions in vector space (must be positive integer)
        metric: Distance metric ("L2" for Euclidean, "IP" for Inner Product)
        metadata: Optional metadata dictionary to associate with the index

    Returns:
        Tuple containing:
            - FAISS index instance ready for use
            - Metadata dictionary with index configuration information

    Raises:
        ValueError: If the specified index_type is not supported
        TypeError: If dimension is not a positive integer

    Example:
        >>> index, info = create_index_from_type("L2", 128)
        >>> index, info = create_index_from_type("IVF100", 64, metric="IP")
    """
    # Input validation phase
    # Validate dimension parameter to prevent FAISS errors downstream
    # FAISS requires positive integer dimensions for proper index construction
    if not isinstance(dimension, int) or dimension <= 0:
        raise TypeError("Dimension must be a positive integer")

    # Initialize metadata if not provided
    # Creating a new dict prevents mutation of default argument
    # This follows Python best practices for mutable default arguments
    if metadata is None:
        metadata = {}

    # Dynamic import to avoid circular dependencies
    # Lazy importing prevents module initialization order issues
    # This pattern allows optional dependencies without breaking core functionality
    from .transformations import create_index_from_type as create_from_transform

    try:
        # Primary path: attempt advanced index creation
        # The transformations module provides full-featured index creation
        # This supports complex index types like IVF, PQ, HNSW, etc.
        # Pass all parameters to maintain full functionality
        return create_from_transform(index_type, dimension, metric, metadata)

    except Exception:
        # Fallback path: provide basic index types for reliability
        # This ensures the function works even if transformations module fails
        # Catches all exceptions to be robust against import or implementation errors
        # Only basic flat indices are supported in fallback mode
        import faiss

        # Handle basic flat index types with appropriate distance metrics
        # Flat indices provide exact search but may be slower for large datasets
        if index_type in ("L2", "FLAT"):
            # Create L2 (Euclidean distance) flat index
            # L2 distance: sqrt(sum((a[i] - b[i])^2)) - standard similarity metric
            # Flat means exhaustive search through all vectors
            index = faiss.IndexFlatL2(dimension)

            # Build comprehensive metadata for the created index
            # Include all relevant configuration information for debugging and monitoring
            index_metadata = {"type": "FLAT", "metric": "L2", "dimension": dimension}
            index_metadata.update(metadata)  # Merge with user-provided metadata
            return index, index_metadata

        elif index_type == "IP":
            # Create Inner Product flat index
            # IP distance: sum(a[i] * b[i]) - useful for normalized vectors
            # Higher IP values indicate greater similarity (opposite of L2)
            index = faiss.IndexFlatIP(dimension)

            # Build comprehensive metadata for the created index
            # Ensure consistency with L2 index metadata structure
            index_metadata = {"type": "FLAT", "metric": "IP", "dimension": dimension}
            index_metadata.update(metadata)  # Merge with user-provided metadata
            return index, index_metadata

        else:
            # Unsupported index type in fallback mode
            # Provide comprehensive error message to help users understand limitations
            # List supported fallback types and suggest using transformations module
            raise ValueError(
                f"Unsupported index type: {index_type}. "
                f"Supported types: L2, FLAT, IP (or use transformations module for advanced types)"
            )

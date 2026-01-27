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
Core index implementations and utilities for FAISSx.

This module organizes and exports various FAISS index implementations and utility
functions. It provides a clean public API for accessing different types of vector
indices and their supporting operations.
"""

# Core index implementations - organized by type
# Factory and utility functions for index management
from .factory import index_factory  # Index creation from string descriptions
from .flat import IndexFlatL2  # Basic L2 distance-based index
from .hnsw_flat import IndexHNSWFlat  # Hierarchical navigable small world graph

# ID mapping implementations for custom vector identification
from .id_map import IndexIDMap, IndexIDMap2  # Basic and extended ID mapping
from .io import read_index, write_index  # Index persistence operations
from .ivf_flat import IndexIVFFlat  # Inverted file index with flat vectors
from .ivf_pq import IndexIVFPQ  # Combined IVF and PQ for large-scale search
from .ivf_scalar_quantizer import IndexIVFScalarQuantizer  # Scalar quantization
from .modification import merge_indices, split_index  # Index modification utilities

# Quantization-based indices for memory-efficient vector storage
from .pq import IndexPQ  # Product quantization for memory efficiency
from .scalar_quantizer import IndexScalarQuantizer  # Scalar quantization

# Public API exports - organized by category for clear documentation
__all__ = [
    # Core indices - basic vector search implementations
    'IndexFlatL2',  # Exact L2 distance search
    'IndexIVFFlat',  # Approximate search with inverted file structure
    'IndexHNSWFlat',  # Graph-based approximate search

    # Quantization indices - memory-efficient implementations
    'IndexPQ',  # Product quantization for reduced memory usage
    'IndexIVFPQ',  # Combined IVF and PQ for large datasets
    'IndexScalarQuantizer',  # Simple scalar quantization
    'IndexIVFScalarQuantizer',  # IVF with scalar quantization

    # ID mapping - custom vector identification
    'IndexIDMap',  # Basic ID mapping wrapper
    'IndexIDMap2',  # Extended ID mapping with additional features

    # Utilities - index management operations
    'index_factory',  # Create indices from string descriptions
    'write_index',  # Save index to disk
    'read_index',  # Load index from disk
    'merge_indices',  # Combine multiple indices
    'split_index',  # Split index into multiple parts
]

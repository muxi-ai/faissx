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
FAISSx Index Implementations Package

This package provides client-side implementations of FAISS index classes that support both local
and remote operation modes. Local mode uses FAISS directly, while remote mode communicates with
the FAISSx server for distributed vector operations.
"""

# Core index implementations
from .flat import IndexFlatL2  # Basic L2 distance-based index
from .ivf_flat import IndexIVFFlat  # Inverted file index with flat vectors
from .hnsw_flat import IndexHNSWFlat  # Hierarchical navigable small world graph
from .pq import IndexPQ  # Product quantization for memory efficiency
from .ivf_pq import IndexIVFPQ  # Combined IVF and PQ for large-scale search
from .scalar_quantizer import IndexScalarQuantizer  # Scalar quantization
from .ivf_scalar_quantizer import IndexIVFScalarQuantizer  # Scalar quantization

# ID mapping implementations for custom vector identification
# IndexIDMap: Basic ID mapping functionality
# IndexIDMap2: Extended with vector update capabilities
from .id_map import IndexIDMap, IndexIDMap2

# Factory function for creating indices from string descriptions
# Compatible with FAISS's index_factory interface
from .factory import index_factory

# Index persistence operations
# write_index: Save index to disk
# read_index: Load index from disk
from .io import write_index, read_index

# Index modification utilities
# merge_indices: Combine multiple indices
# split_index: Divide index into smaller parts
from .modification import merge_indices, split_index

# Public API exports
__all__ = [
    'IndexFlatL2',  # Basic L2 distance index
    'IndexIVFFlat',  # IVF with flat vectors
    'IndexHNSWFlat',  # HNSW graph index
    'IndexPQ',  # Product quantization
    'IndexIVFPQ',  # IVF with product quantization
    'IndexScalarQuantizer',  # Scalar quantization
    'IndexIVFScalarQuantizer',  # Scalar quantization
    'IndexIDMap',  # Basic ID mapping
    'IndexIDMap2',  # Extended ID mapping
    'index_factory',  # Index creation factory
    'write_index',  # Index persistence
    'read_index',  # Index loading
    'merge_indices',  # Index combination
    'split_index',  # Index division
]

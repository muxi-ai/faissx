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
This package contains the FAISSx index implementations.

Each class provides a client-side implementation of a FAISS index class that
can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).
"""

from .flat import IndexFlatL2
from .ivf_flat import IndexIVFFlat
from .hnsw_flat import IndexHNSWFlat
from .pq import IndexPQ
from .ivf_pq import IndexIVFPQ
from .scalar_quantizer import IndexScalarQuantizer
# IndexIDMap provides custom IDs for vectors
# IndexIDMap2 extends it with vector update functionality
from .id_map import IndexIDMap, IndexIDMap2
# Factory function for creating indices from string descriptions (FAISS-compatible)
from .factory import index_factory
# Index persistence functions
from .io import write_index, read_index
# Index modification functions (merging and splitting)
from .modification import merge_indices, split_index

__all__ = [
    'IndexFlatL2',
    'IndexIVFFlat',
    'IndexHNSWFlat',
    'IndexPQ',
    'IndexIVFPQ',
    'IndexScalarQuantizer',
    'IndexIDMap',
    'IndexIDMap2',
    'index_factory',
    'write_index',
    'read_index',
    'merge_indices',
    'split_index',
]

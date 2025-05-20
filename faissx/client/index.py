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
FAISSx Index Implementation Module

This module provides client-side implementations of FAISS index classes that
communicate with a remote FAISSx service via ZeroMQ. Key features include:

- Drop-in replacements for FAISS index types
- Identical API signatures to the original FAISS implementations
- Transparent remote execution of add, search, and other vector operations
- Local-to-server index mapping to maintain consistent vector references
- Automatic conversion of data types and array formats for ZeroMQ transport
- Support for all standard FAISS index operations with server delegation

Each index class matches the behavior of its FAISS counterpart while sending
the actual computational work to the FAISSx server.
"""

# For backward compatibility, make the index classes available directly from this module
from .indices.flat import IndexFlatL2
from .indices.ivf_flat import IndexIVFFlat
from .indices.hnsw_flat import IndexHNSWFlat
from .indices.pq import IndexPQ
from .indices.ivf_pq import IndexIVFPQ
from .indices.scalar_quantizer import IndexScalarQuantizer
from .indices.id_map import IndexIDMap, IndexIDMap2

# Define what's available via import
__all__ = [
    'IndexFlatL2',
    'IndexIVFFlat',
    'IndexHNSWFlat',
    'IndexPQ',
    'IndexIVFPQ',
    'IndexScalarQuantizer',
    'IndexIDMap',
    'IndexIDMap2'
]

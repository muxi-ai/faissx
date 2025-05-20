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
FAISSx index_factory implementation.

This module provides a FAISS-compatible index factory that parses index descriptor strings
and creates the corresponding index objects. It supports the same syntax as the original
FAISS implementation for seamless compatibility.

Example usage:
    index = index_factory(128, "Flat")  # Creates a flat index
    index = index_factory(128, "IVF100,Flat")  # Creates an IVF index with a flat quantizer
    index = index_factory(128, "IVF100,PQ16")  # Creates an IVF index with PQ
    index = index_factory(128, "HNSW32")  # Creates an HNSW index
"""

import re
import logging
from typing import Any, Optional

import faiss

# Import all supported index types
from .flat import IndexFlatL2
from .ivf_flat import IndexIVFFlat
from .hnsw_flat import IndexHNSWFlat
from .pq import IndexPQ
from .ivf_pq import IndexIVFPQ
from .scalar_quantizer import IndexScalarQuantizer
from .id_map import IndexIDMap, IndexIDMap2

# Configure module-level logger
logger = logging.getLogger(__name__)


def index_factory(d: int, description: str, metric: Optional[int] = None) -> Any:
    """
    Parse an index description string and create the corresponding index.

    This function implements a FAISS-compatible index factory that creates vector indices
    based on a description string. It supports various index types including Flat, IVF,
    HNSW, PQ, and their combinations.

    Args:
        d: Dimensionality of the vectors to be indexed
        description: Index description string using FAISS syntax (e.g., "Flat", "IVF100,Flat")
        metric: Distance metric to use (default: faiss.METRIC_L2)

    Returns:
        A FAISSx index object corresponding to the description

    Raises:
        ValueError: If the description is malformed or unsupported
    """
    # Set default metric to L2 if not specified
    if metric is None:
        metric = faiss.METRIC_L2

    # Remove all whitespace from description for consistent parsing
    description = re.sub(r"\s+", "", description)

    # Handle IDMap and IDMap2 wrappers
    # These allow mapping between arbitrary IDs and sequential indices
    if description.startswith("IDMap") or description.startswith("IDMap2"):
        is_idmap2 = description.startswith("IDMap2")
        # Extract the sub-index description after the IDMap prefix
        sub_description = (
            description[len("IDMap2,"):] if is_idmap2 else description[len("IDMap,"):]
        )
        # Recursively create the sub-index
        sub_index = index_factory(d, sub_description, metric)
        # Wrap with appropriate IDMap type
        return IndexIDMap2(sub_index) if is_idmap2 else IndexIDMap(sub_index)

    # Handle Flat index - simplest case with no compression
    if description == "Flat":
        if metric == faiss.METRIC_L2:
            return IndexFlatL2(d)
        else:
            # Currently only L2 distance is supported for Flat indices
            raise ValueError(f"Metric {metric} not supported for Flat index")

    # Handle HNSW (Hierarchical Navigable Small World) index
    # Format: HNSW<M> where M is the number of connections per layer
    hnsw_match = re.match(r"HNSW(\d+)", description)
    if hnsw_match:
        m = int(hnsw_match.group(1))
        return IndexHNSWFlat(d, m, metric)

    # Handle IVF (Inverted File) indices
    # Format: IVF<nlist>,<quantizer> where nlist is number of clusters
    ivf_match = re.match(r"IVF(\d+),(\w+)", description)
    if ivf_match:
        nlist = int(ivf_match.group(1))
        coarse_quantizer_type = ivf_match.group(2)

        # Handle IVF with Flat quantizer
        if coarse_quantizer_type == "Flat":
            coarse_quantizer = IndexFlatL2(d)
            return IndexIVFFlat(coarse_quantizer, d, nlist, metric)

        # Handle IVF with Product Quantization
        # Format: PQ<M>[x<nbits>] where M is subquantizers, nbits is bits per subquantizer
        pq_match = re.match(r"PQ(\d+)(x\d+)?", coarse_quantizer_type)
        if pq_match:
            m = int(pq_match.group(1))  # Number of subquantizers
            nbits = 8  # Default bits per subquantizer

            # Parse optional bits per subquantizer specification
            if pq_match.group(2):
                nbits = int(pq_match.group(2)[1:])

            coarse_quantizer = IndexFlatL2(d)
            return IndexIVFPQ(coarse_quantizer, d, nlist, m, nbits, metric)

    # Handle Scalar Quantizer
    # Format: SQ<nbits> where nbits is bits per dimension
    sq_match = re.match(r"SQ(\d+)", description)
    if sq_match:
        return IndexScalarQuantizer(d, metric)

    # Handle direct Product Quantization (without IVF)
    # Format: PQ<M>[x<nbits>] where M is subquantizers, nbits is bits per subquantizer
    pq_match = re.match(r"PQ(\d+)(x\d+)?", description)
    if pq_match:
        m = int(pq_match.group(1))  # Number of subquantizers
        nbits = 8  # Default bits per subquantizer

        # Parse optional bits per subquantizer specification
        if pq_match.group(2):
            nbits = int(pq_match.group(2)[1:])

        return IndexPQ(d, m, nbits, metric)

    # If no matching index type was found, raise an error
    raise ValueError(f"Unsupported index description: {description}")

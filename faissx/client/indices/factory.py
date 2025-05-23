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

The factory pattern allows for concise index creation using standardized string descriptors
instead of directly instantiating index classes with numerous parameters. This approach
matches FAISS's original API for easier migration between libraries.

Supported index types and descriptor formats:

- Flat: "Flat"
  Simple exhaustive search index (most accurate but slowest)
  Example: index_factory(128, "Flat")

- HNSW: "HNSW<M>"
  Hierarchical Navigable Small World (fast graph-based approximate search)
  <M> is the number of connections per node (higher = better accuracy, more memory)
  Example: index_factory(128, "HNSW32")  # 32 connections per node

- IVF: "IVF<nlist>,<quantizer>"
  Inverted File Index (clustering-based approximate search)
  <nlist> is the number of clusters (typically sqrt(dataset size))
  <quantizer> is the storage type within clusters (Flat, PQ, etc.)
  Example: index_factory(128, "IVF100,Flat")  # 100 clusters with flat storage

- PQ: "PQ<m>[x<nbits>]"
  Product Quantization (vector compression for memory efficiency)
  <m> is the number of subquantizers (must divide vector dimension)
  <nbits> is bits per subquantizer (default: 8)
  Example: index_factory(128, "PQ16")  # 16 subquantizers with 8 bits each
  Example: index_factory(128, "PQ16x4")  # 16 subquantizers with 4 bits each

- SQ: "SQ<nbits>"
  Scalar Quantization (simple compression technique)
  <nbits> is bits per component (typically 8)
  Example: index_factory(128, "SQ8")  # 8 bits per component

- IDMap/IDMap2: "IDMap,<index>" or "IDMap2,<index>"
  ID mapping wrappers for custom vector IDs
  <index> is any other index type
  Example: index_factory(128, "IDMap,Flat")  # Flat index with ID mapping
  Example: index_factory(128, "IDMap2,HNSW32")  # HNSW index with updatable ID mapping

- Combined formats:
  Example: index_factory(128, "IVF100,PQ16")  # IVF with PQ quantization
  Example: index_factory(128, "IDMap2,IVF100,PQ16")  # IDMap2 wrapper for IVF-PQ
"""

import re
import logging
from typing import Optional, Tuple, Union, TypeVar, Pattern, Match

import faiss

# Import all supported index types
from .flat import IndexFlatL2
from .ivf_flat import IndexIVFFlat
from .hnsw_flat import IndexHNSWFlat
from .pq import IndexPQ
from .ivf_pq import IndexIVFPQ
from .scalar_quantizer import IndexScalarQuantizer
from .id_map import IndexIDMap, IndexIDMap2
from .base import FAISSxBaseIndex

# Configure module-level logger
logger = logging.getLogger(__name__)

# Define a generic type for index classes
T = TypeVar('T', bound=FAISSxBaseIndex)

# Define constants for metrics
try:
    METRIC_L2 = faiss.METRIC_L2
    METRIC_INNER_PRODUCT = faiss.METRIC_INNER_PRODUCT
except ImportError:
    # Fallback values if FAISS is not available
    METRIC_L2 = 0
    METRIC_INNER_PRODUCT = 1

# Precompile regex patterns for better performance
# Pattern for IDMap or IDMap2 prefixes, e.g., "IDMap,Flat" or "IDMap2,HNSW32"
# Group 1: The map type ("IDMap" or "IDMap2")
# Group 2: The wrapped index description
PATTERN_IDMAP: Pattern[str] = re.compile(r"^(IDMap2?),(.*)")

# Pattern for HNSW indices, e.g., "HNSW32" where 32 is the M parameter
# Group 1: The M parameter value (number of connections per node)
PATTERN_HNSW: Pattern[str] = re.compile(r"^HNSW(\d+)$")

# Pattern for IVF indices, e.g., "IVF100,Flat" or "IVF100,PQ16"
# Group 1: The nlist parameter (number of clusters)
# Group 2: The quantizer type ("Flat", "PQ16", etc.)
PATTERN_IVF: Pattern[str] = re.compile(r"^IVF(\d+),(\w+)$")

# Pattern for Product Quantization, e.g., "PQ16" or "PQ16x8"
# Group 1: The m parameter (number of subquantizers)
# Group 2: Optional bit width specification (e.g., "x8" for 8 bits)
PATTERN_PQ: Pattern[str] = re.compile(r"^PQ(\d+)(x\d+)?$")

# Pattern for Scalar Quantization, e.g., "SQ8"
# Group 1: The number of bits
PATTERN_SQ: Pattern[str] = re.compile(r"^SQ(\d+)$")

# Union type for all supported index instances
IndexType = Union[
    IndexFlatL2,
    IndexIVFFlat,
    IndexHNSWFlat,
    IndexPQ,
    IndexIVFPQ,
    IndexScalarQuantizer,
    IndexIDMap,
    IndexIDMap2
]


def _create_flat_index(d: int, metric: int) -> IndexFlatL2:
    """
    Create a flat index with the specified dimension and metric.

    A flat index stores vectors in their original form and performs exhaustive search
    by computing distances between the query and all indexed vectors. This provides
    exact nearest neighbor search at the cost of linear scaling with dataset size.

    Args:
        d: Vector dimension
        metric: Distance metric (only L2 supported)

    Returns:
        IndexFlatL2: A flat index instance

    Raises:
        ValueError: If metric is not L2
    """
    if metric != METRIC_L2:
        raise ValueError(f"Metric {metric} not supported for Flat index. Only L2 is supported.")
    return IndexFlatL2(d)


def _create_hnsw_index(d: int, m: int, metric: int) -> IndexHNSWFlat:
    """
    Create an HNSW (Hierarchical Navigable Small World) index.

    HNSW is a graph-based approach for approximate nearest neighbor search
    that builds a navigable small world graph for efficient traversal.
    It offers excellent search performance with good accuracy and scales
    well for high-dimensional data.

    Args:
        d: Vector dimension
        m: Number of connections per layer (higher = better accuracy but more memory)
        metric: Distance metric (L2 or inner product)

    Returns:
        IndexHNSWFlat: An HNSW index instance
    """
    return IndexHNSWFlat(d, m, metric)


def _create_ivf_flat_index(
    d: int, nlist: int, metric: int, expected_training_size: int = 1000
) -> IndexIVFFlat:
    """
    Create an IVF (Inverted File) index with flat quantizer.

    IVF indices partition the vector space into clusters and only search within
    the most relevant clusters, reducing search complexity for large datasets.
    The flat quantizer means vectors within clusters are stored in raw form.

    Args:
        d: Vector dimension
        nlist: Number of clusters (typically sqrt(n) where n is dataset size)
        metric: Distance metric
        expected_training_size: Expected number of training vectors (used for safety checks)

    Returns:
        IndexIVFFlat: An IVF index with flat quantizer
    """
    # Use safety check to prevent clustering crashes
    safe_nlist = _safe_nlist_for_training_size(nlist, expected_training_size)

    # Create a flat index to use as the coarse quantizer for clustering
    coarse_quantizer = IndexFlatL2(d)
    return IndexIVFFlat(coarse_quantizer, d, safe_nlist, metric)


def _safe_nlist_for_training_size(nlist: int, expected_training_size: int = 1000) -> int:
    """
    Calculate a safe nlist value based on expected training size.

    FAISS requires approximately 39 * nlist training points for stable clustering,
    but we use a more conservative estimate to prevent crashes.
    This function ensures we don't create IVF indices that will crash during training.

    Args:
        nlist: Requested number of clusters
        expected_training_size: Expected number of training vectors

    Returns:
        Safe nlist value that won't cause clustering to crash
    """
    # Use a more conservative estimate: 100 points per cluster to ensure stability
    # This is higher than FAISS's minimum of 39 to provide a safety margin
    min_points_per_cluster = 100
    max_safe_nlist = max(1, expected_training_size // min_points_per_cluster)

    if nlist > max_safe_nlist:
        safe_nlist = max_safe_nlist
        logger.warning(
            f"Reducing nlist from {nlist} to {safe_nlist} to prevent clustering crash. "
            f"Using conservative estimate of {min_points_per_cluster} points per cluster. "
            f"Original FAISS minimum would be {nlist * 39} training points for nlist={nlist}."
        )
        return safe_nlist

    return nlist


def _create_ivf_pq_index(
    d: int, nlist: int, m: int, nbits: int, metric: int, expected_training_size: int = 1000
) -> IndexIVFPQ:
    """
    Create an IVF index with product quantization.

    This combines IVF clustering with Product Quantization (PQ) for storage efficiency.
    PQ compresses vectors by splitting them into subvectors and quantizing each separately,
    significantly reducing memory usage at a small cost to accuracy.

    Args:
        d: Vector dimension
        nlist: Number of clusters
        m: Number of subquantizers (must divide d)
        nbits: Bits per subquantizer (typically 8)
        metric: Distance metric
        expected_training_size: Expected number of training vectors (used for safety checks)

    Returns:
        IndexIVFPQ: An IVF index with product quantization

    Raises:
        ValueError: If m does not divide d evenly
    """
    # Validate that m divides d evenly
    if d % m != 0:
        raise ValueError(f"Number of subquantizers (m={m}) must divide dimension (d={d}) evenly")

    # Use safety check to prevent clustering crashes
    safe_nlist = _safe_nlist_for_training_size(nlist, expected_training_size)

    coarse_quantizer = IndexFlatL2(d)
    return IndexIVFPQ(coarse_quantizer, d, safe_nlist, m, nbits, metric)


def _parse_pq_params(description: str) -> Tuple[int, int]:
    """
    Parse Product Quantization parameters from description string.

    Parses descriptors like "PQ16" or "PQ16x8" to extract the number of subquantizers
    and bits per subquantizer. If the bit width is not specified, defaults to 8 bits.

    Args:
        description: PQ description string (e.g., "PQ16" or "PQ16x8")

    Returns:
        Tuple[int, int]: Number of subquantizers and bits per subquantizer

    Raises:
        ValueError: If description format is invalid
    """
    pq_match: Optional[Match[str]] = PATTERN_PQ.match(description)
    if not pq_match:
        raise ValueError(f"Invalid PQ description: {description}. Expected format: PQ<m>[x<bits>]")

    m = int(pq_match.group(1))  # Extract number of subquantizers
    nbits = 8  # Default bits per subquantizer

    if pq_match.group(2):  # Check for optional bits parameter
        nbits = int(pq_match.group(2)[1:])

    return m, nbits


def index_factory(
    d: int, description: str, metric: Optional[int] = None, expected_training_size: int = 1000
) -> IndexType:
    """
    Create a FAISS-compatible index from a description string.

    This factory function parses an index description string and creates the corresponding
    index object with appropriate parameters. It serves as a drop-in replacement for FAISS's
    own index_factory function, supporting the same syntax and index types.

    Args:
        d: Vector dimension
        description: Index description string. Supported formats include:
            - "Flat": Simple flat index (exhaustive search)
            - "HNSW<M>": Hierarchical Navigable Small World (e.g., "HNSW32")
            - "IVF<nlist>,<quantizer>": Inverted file index (e.g., "IVF100,Flat")
            - "PQ<m>[x<bits>]": Product quantization (e.g., "PQ16" or "PQ16x8")
            - "SQ<bits>": Scalar quantization (e.g., "SQ8")
            - "IDMap,<index>": ID mapping wrapper (e.g., "IDMap,Flat")
            - "IDMap2,<index>": ID mapping with compact storage (e.g., "IDMap2,Flat")
        metric: Distance metric (default: faiss.METRIC_L2)
        expected_training_size: Expected number of training vectors (for IVF safety checks)

    Returns:
        IndexType: A FAISSx index instance corresponding to the description

    Raises:
        ValueError: If description is malformed or unsupported
    """
    metric = metric or METRIC_L2
    logger.debug(f"Using metric: {metric}")

    # Remove whitespace for consistent parsing
    description = re.sub(r"\s+", "", description)
    logger.debug(f"Creating index with dimension {d} and description: {description}")

    try:
        # Handle IDMap and IDMap2 wrappers
        idmap_match: Optional[Match[str]] = PATTERN_IDMAP.match(description)
        if idmap_match:
            is_idmap2 = idmap_match.group(1) == "IDMap2"
            sub_description = idmap_match.group(2)
            logger.debug(
                f"Creating {'IDMap2' if is_idmap2 else 'IDMap'} wrapper around {sub_description}"
            )

            try:
                # Recursively create the underlying index
                sub_index = index_factory(d, sub_description, metric, expected_training_size)
            except Exception as e:
                raise ValueError(f"Failed to create sub-index for {description}: {e}")

            # Wrap the sub-index with the appropriate ID mapping layer
            return IndexIDMap2(sub_index) if is_idmap2 else IndexIDMap(sub_index)

        # Handle Flat index
        if description == "Flat":
            logger.debug(f"Creating Flat index with dimension {d}")
            return _create_flat_index(d, metric)

        # Handle HNSW index
        hnsw_match: Optional[Match[str]] = PATTERN_HNSW.match(description)
        if hnsw_match:
            m = int(hnsw_match.group(1))
            logger.debug(f"Creating HNSW index with M={m}")
            return _create_hnsw_index(d, m, metric)

        # Handle IVF indices
        ivf_match: Optional[Match[str]] = PATTERN_IVF.match(description)
        if ivf_match:
            nlist = int(ivf_match.group(1))
            coarse_quantizer_type = ivf_match.group(2)
            logger.debug(
                f"Creating IVF index with nlist={nlist}, quantizer={coarse_quantizer_type}"
            )

            if coarse_quantizer_type == "Flat":
                logger.debug("Using Flat quantizer for IVF index")
                return _create_ivf_flat_index(d, nlist, metric, expected_training_size)

            if PATTERN_PQ.match(coarse_quantizer_type):
                m, nbits = _parse_pq_params(coarse_quantizer_type)
                logger.debug(
                    f"Using PQ quantizer for IVF index with m={m}, nbits={nbits}"
                )
                return _create_ivf_pq_index(d, nlist, m, nbits, metric, expected_training_size)

        # Handle Scalar Quantizer
        sq_match: Optional[Match[str]] = PATTERN_SQ.match(description)
        if sq_match:
            logger.debug("Creating Scalar Quantizer index")
            return IndexScalarQuantizer(d, metric)

        # Handle direct Product Quantization
        pq_match: Optional[Match[str]] = PATTERN_PQ.match(description)
        if pq_match:
            m, nbits = _parse_pq_params(description)
            logger.debug(f"Creating PQ index with m={m}, nbits={nbits}")
            return IndexPQ(d, m, nbits, metric)

        # If we get here, the description is not supported
        raise ValueError(
            f"Unsupported index description: {description}. "
            f"See documentation for supported formats."
        )

    except ValueError:
        # Re-raise ValueError directly
        raise
    except Exception as e:
        # Wrap other exceptions with more context
        raise ValueError(
            f"Error creating index from description '{description}': {str(e)}"
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server HNSW Module
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
FAISSx Server HNSW Module

This module provides enhanced controls for HNSW (Hierarchical Navigable Small World)
indices, allowing fine-grained parameter tuning and efficient construction.
"""

import time
import faiss
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("faissx.server")


# Default parameters for HNSW construction
DEFAULT_M = 32  # Number of connections per layer
DEFAULT_EF_CONSTRUCTION = 200  # Size of the dynamic list for nearest neighbors during construction
DEFAULT_EF_SEARCH = 64  # Size of the dynamic list for nearest neighbors during search


def create_hnsw_index(
    dimension: int,
    metric_type: str = "L2",
    hnsw_params: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create an HNSW index with optimized parameters.

    Args:
        dimension: Dimension of vectors
        metric_type: Distance metric type ('L2' or 'IP')
        hnsw_params: HNSW-specific parameters

    Returns:
        tuple: (index, index_info)
    """
    # Set default parameters
    params = {
        "M": DEFAULT_M,
        "efConstruction": DEFAULT_EF_CONSTRUCTION,
        "efSearch": DEFAULT_EF_SEARCH,
        "seed": None
    }

    # Update with provided parameters
    if hnsw_params:
        params.update(hnsw_params)

    # Set metric type
    faiss_metric_type = faiss.METRIC_L2
    if metric_type.upper() == "IP":
        faiss_metric_type = faiss.METRIC_INNER_PRODUCT

    # Create the index
    index = faiss.IndexHNSWFlat(dimension, params["M"], faiss_metric_type)

    # Configure HNSW parameters
    index.hnsw.efConstruction = params["efConstruction"]
    index.hnsw.efSearch = params["efSearch"]

    # Set random seed if provided
    if params["seed"] is not None:
        index.hnsw.level_generator_seed = params["seed"]

    # Prepare index info
    index_info = {
        "type": "IndexHNSWFlat",
        "dimension": dimension,
        "metric_type": metric_type,
        "hnsw": {
            "M": params["M"],
            "efConstruction": params["efConstruction"],
            "efSearch": params["efSearch"],
            "seed": params["seed"]
        }
    }

    return index, index_info


def optimize_hnsw_params(
    sample_vectors: np.ndarray,
    metric_type: str = "L2",
    target_qps: Optional[float] = None,
    target_recall: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize HNSW parameters based on sample data and target requirements.

    Args:
        sample_vectors: Sample vectors for optimization
        metric_type: Distance metric type ('L2' or 'IP')
        target_qps: Target queries per second (None for maximum recall)
        target_recall: Target recall (None for maximum QPS)

    Returns:
        dict: Optimized parameters
    """
    if target_qps is None and target_recall is None:
        # Default to targeting high recall
        target_recall = 0.95

    n_vectors, dimension = sample_vectors.shape

    # Create parameter combinations to test
    m_values = [16, 32, 64]
    ef_construction_values = [100, 200, 400]
    ef_search_values = [32, 64, 128, 256]

    best_params = {"M": DEFAULT_M, "efConstruction": DEFAULT_EF_CONSTRUCTION, "efSearch": DEFAULT_EF_SEARCH}
    best_score = -1  # Higher is better

    # Set metric type
    faiss_metric_type = faiss.METRIC_L2
    if metric_type.upper() == "IP":
        faiss_metric_type = faiss.METRIC_INNER_PRODUCT

    # Create a ground truth index for comparison
    gt_index = faiss.IndexFlatL2(dimension)
    gt_index.add(sample_vectors)

    # Test parameter combinations
    results = []
    for m in m_values:
        for ef_construction in ef_construction_values:
            # Create and train index
            index = faiss.IndexHNSWFlat(dimension, m, faiss_metric_type)
            index.hnsw.efConstruction = ef_construction

            # Add vectors and measure build time
            start_time = time.time()
            index.add(sample_vectors)
            build_time = time.time() - start_time

            # Test search performance with different efSearch values
            for ef_search in ef_search_values:
                index.hnsw.efSearch = ef_search

                # Measure search time and recall
                k = 10  # Number of neighbors to search for

                # Create queries from a subset of vectors
                n_queries = min(100, n_vectors)
                query_vectors = sample_vectors[:n_queries].copy()

                # Get ground truth results
                gt_distances, gt_indices = gt_index.search(query_vectors, k)

                # Measure search performance
                start_time = time.time()
                distances, indices = index.search(query_vectors, k)
                search_time = time.time() - start_time

                # Calculate recall
                recall = 0.0
                for i in range(n_queries):
                    gt_set = set(gt_indices[i])
                    result_set = set(indices[i])
                    if len(gt_set) > 0:
                        recall += len(gt_set.intersection(result_set)) / len(gt_set)
                recall /= n_queries

                # Calculate QPS
                qps = n_queries / search_time

                # Score this parameter combination based on target
                if target_recall is not None:
                    # Optimize for recall with minimal latency
                    if recall >= target_recall:
                        score = 1.0 / search_time  # Higher QPS is better
                    else:
                        score = recall / target_recall  # Partial credit for recall
                elif target_qps is not None:
                    # Optimize for QPS with maximum recall
                    if qps >= target_qps:
                        score = recall  # Higher recall is better
                    else:
                        score = (qps / target_qps) * recall  # Partial credit for QPS

                # Save result
                result = {
                    "M": m,
                    "efConstruction": ef_construction,
                    "efSearch": ef_search,
                    "build_time": build_time,
                    "search_time": search_time,
                    "qps": qps,
                    "recall": recall,
                    "score": score
                }
                results.append(result)

                # Update best parameters if better
                if score > best_score:
                    best_score = score
                    best_params = {
                        "M": m,
                        "efConstruction": ef_construction,
                        "efSearch": ef_search
                    }

    # Return best parameters along with performance metrics
    return {
        "optimized_params": best_params,
        "results": results,
        "target_qps": target_qps,
        "target_recall": target_recall
    }


def get_hnsw_stats(index: Any) -> Dict[str, Any]:
    """
    Get detailed statistics and parameters for an HNSW index.

    Args:
        index: HNSW index

    Returns:
        dict: HNSW statistics and parameters
    """
    if not isinstance(index, faiss.IndexHNSW):
        raise TypeError("Index is not an HNSW index")

    stats = {
        "ntotal": index.ntotal,
        "dimension": index.d,
        "hnsw": {
            "M": index.hnsw.M,
            "efConstruction": index.hnsw.efConstruction,
            "efSearch": index.hnsw.efSearch,
            "level_generator_seed": index.hnsw.level_generator_seed,
            "upper_beam": index.hnsw.upper_beam,
            "max_level": index.hnsw.max_level
        }
    }

    # Get storage statistics if possible
    if hasattr(index, "storage_info"):
        stats["storage_info"] = index.storage_info()

    return stats


def update_hnsw_params(index: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update HNSW index parameters.

    Args:
        index: HNSW index
        params: Parameters to update

    Returns:
        dict: Updated parameters
    """
    if not isinstance(index, faiss.IndexHNSW):
        raise TypeError("Index is not an HNSW index")

    updated = {}

    # Update efSearch
    if "efSearch" in params:
        index.hnsw.efSearch = params["efSearch"]
        updated["efSearch"] = params["efSearch"]

    # Some parameters can only be updated on an empty index
    if index.ntotal == 0:
        # Update efConstruction
        if "efConstruction" in params:
            index.hnsw.efConstruction = params["efConstruction"]
            updated["efConstruction"] = params["efConstruction"]

        # Update seed
        if "seed" in params:
            index.hnsw.level_generator_seed = params["seed"]
            updated["seed"] = params["seed"]
    else:
        # Log warning for parameters that can't be updated
        for param_name in ["efConstruction", "seed"]:
            if param_name in params:
                logger.warning(f"Warning: Cannot update {param_name} on a non-empty index")

    # M cannot be updated after initialization
    if "M" in params:
        logger.warning("Warning: Cannot update M parameter after index creation")

    return updated


def incremental_construction(
    index: Any,
    vectors: np.ndarray,
    batch_size: int = 10000,
    efConstruction: Optional[int] = None,
    callback = None
) -> Dict[str, Any]:
    """
    Add vectors to an HNSW index in batches for more efficient construction.

    Args:
        index: HNSW index
        vectors: Vectors to add
        batch_size: Number of vectors to add in each batch
        efConstruction: Optional temporary efConstruction value
        callback: Optional callback function to report progress

    Returns:
        dict: Construction results
    """
    if not isinstance(index, faiss.IndexHNSW):
        raise TypeError("Index is not an HNSW index")

    n_vectors = vectors.shape[0]

    # Save original efConstruction
    original_ef = index.hnsw.efConstruction

    # Set temporary efConstruction if provided
    if efConstruction is not None:
        index.hnsw.efConstruction = efConstruction

    start_time = time.time()

    # Add vectors in batches
    n_batches = (n_vectors + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, n_vectors)

        batch_vectors = vectors[batch_start:batch_end]
        index.add(batch_vectors)

        # Call progress callback if provided
        if callback:
            progress = (i + 1) / n_batches
            callback(progress, batch_end, n_vectors)

    total_time = time.time() - start_time

    # Restore original efConstruction
    if efConstruction is not None:
        index.hnsw.efConstruction = original_ef

    return {
        "added_vectors": n_vectors,
        "construction_time": total_time,
        "vectors_per_second": n_vectors / total_time,
        "efConstruction": index.hnsw.efConstruction
    }


def configure_hnsw_for_accuracy(index: Any) -> Dict[str, Any]:
    """
    Configure HNSW parameters for high accuracy.

    Args:
        index: HNSW index

    Returns:
        dict: Updated parameters
    """
    if not isinstance(index, faiss.IndexHNSW):
        raise TypeError("Index is not an HNSW index")

    # High accuracy settings
    if index.ntotal == 0:
        # These can only be set on an empty index
        index.hnsw.efConstruction = 400

    # This can be set anytime
    index.hnsw.efSearch = 256

    return {
        "efConstruction": index.hnsw.efConstruction,
        "efSearch": index.hnsw.efSearch,
        "M": index.hnsw.M
    }


def configure_hnsw_for_speed(index: Any) -> Dict[str, Any]:
    """
    Configure HNSW parameters for high speed.

    Args:
        index: HNSW index

    Returns:
        dict: Updated parameters
    """
    if not isinstance(index, faiss.IndexHNSW):
        raise TypeError("Index is not an HNSW index")

    # High speed settings
    if index.ntotal == 0:
        # These can only be set on an empty index
        index.hnsw.efConstruction = 100

    # This can be set anytime
    index.hnsw.efSearch = 32

    return {
        "efConstruction": index.hnsw.efConstruction,
        "efSearch": index.hnsw.efSearch,
        "M": index.hnsw.M
    }


def calculate_memory_usage(index: Any) -> Dict[str, Any]:
    """
    Calculate memory usage for an HNSW index.

    Args:
        index: HNSW index

    Returns:
        dict: Memory usage statistics
    """
    if not isinstance(index, faiss.IndexHNSW):
        raise TypeError("Index is not an HNSW index")

    ntotal = index.ntotal
    dimension = index.d

    # Calculate approximate memory usage
    # Each vector uses 4 bytes per dimension
    vector_memory = ntotal * dimension * 4

    # HNSW graph structure
    # Each node has M links per level, and the number of levels is logarithmic
    # Links are stored as 32-bit integers (4 bytes)
    # Average number of levels per node is proportional to log(n)
    avg_levels = max(1, min(5, int(np.log2(ntotal + 1) / 5)))
    graph_memory = ntotal * index.hnsw.M * avg_levels * 4

    # Additional overhead
    overhead = ntotal * 32  # Rough estimate for various bookkeeping

    total_memory = vector_memory + graph_memory + overhead

    return {
        "total_bytes": total_memory,
        "vector_bytes": vector_memory,
        "graph_bytes": graph_memory,
        "overhead_bytes": overhead,
        "bytes_per_vector": total_memory / ntotal if ntotal > 0 else 0
    }

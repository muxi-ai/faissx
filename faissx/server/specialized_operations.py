#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Specialized Operations
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
FAISSx Server Specialized Operations

This module implements advanced FAISS operations not covered
in the core API to ensure full compatibility with FAISS-CPU.
"""

import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)


def merge_indices(server, target_index_id, source_index_ids):
    """
    Merge multiple source indices into a target index.

    This is an optimized implementation that handles more index types and
    provides better error reporting and validation.

    Args:
        server: FaissIndex server instance
        target_index_id: ID of the target index to merge into
        source_index_ids: List of source index IDs to merge from

    Returns:
        dict: Response indicating success or failure
    """
    if not source_index_ids:
        return {
            "success": False,
            "error": "No source indices provided for merging"
        }

    # Check if the target index exists
    if target_index_id not in server.indexes:
        return {
            "success": False,
            "error": f"Target index {target_index_id} not found"
        }

    # Check if all source indices exist
    for src_id in source_index_ids:
        if src_id not in server.indexes:
            return {
                "success": False,
                "error": f"Source index {src_id} not found"
            }

    target_index = server.indexes[target_index_id]
    target_dim = server.dimensions[target_index_id]

    # Track vectors added from each source
    merged_stats = {}

    try:
        # Process each source index
        for src_id in source_index_ids:
            src_index = server.indexes[src_id]
            src_dim = server.dimensions[src_id]
            src_ntotal_before = target_index.ntotal

            # Check dimension compatibility
            if src_dim != target_dim:
                return {
                    "success": False,
                    "error": (
                        f"Dimension mismatch: target dimension is {target_dim}, "
                        f"but source index {src_id} has dimension {src_dim}"
                    )
                }

            # Handle different index types
            if isinstance(target_index, faiss.IndexFlat):
                # For flat indices we can use direct merging
                if isinstance(src_index, faiss.IndexFlat):
                    try:
                        # Extract vectors from source index
                        vectors = extract_vectors(src_index, 0, src_index.ntotal)
                        # Add to target index
                        target_index.add(vectors)
                        merged_stats[src_id] = {
                            "vectors_added": src_index.ntotal,
                            "type": "direct"
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Error merging flat indices: {str(e)}"
                        }
                else:
                    # Different index types - extract and add
                    vectors = extract_vectors(src_index, 0, src_index.ntotal)
                    target_index.add(vectors)
                    merged_stats[src_id] = {
                        "vectors_added": src_index.ntotal,
                        "type": "extract_add"
                    }

            elif isinstance(target_index, faiss.IndexIVF):
                # For IVF indices, we need compatible clustering
                if isinstance(src_index, faiss.IndexIVF):
                    # Check for compatibility
                    if target_index.nlist != src_index.nlist:
                        # Different cluster counts - extract and add
                        vectors = extract_vectors(src_index, 0, src_index.ntotal)
                        target_index.add(vectors)
                        merged_stats[src_id] = {
                            "vectors_added": src_index.ntotal,
                            "type": "extract_add"
                        }
                    else:
                        # Try to use FAISS merge_from if available
                        try:
                            target_index.merge_from(src_index, 0)
                            merged_stats[src_id] = {
                                "vectors_added": src_index.ntotal,
                                "type": "merge_from"
                            }
                        except AttributeError:
                            # fall back to extract and add
                            vectors = extract_vectors(src_index, 0, src_index.ntotal)
                            target_index.add(vectors)
                            merged_stats[src_id] = {
                                "vectors_added": src_index.ntotal,
                                "type": "extract_add"
                            }
                else:
                    # Different index types - extract and add
                    vectors = extract_vectors(src_index, 0, src_index.ntotal)
                    target_index.add(vectors)
                    merged_stats[src_id] = {
                        "vectors_added": src_index.ntotal,
                        "type": "extract_add"
                    }

            elif isinstance(target_index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                # For IDMap indices, we need to handle IDs
                if isinstance(src_index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                    # Extract vectors and IDs if possible
                    try:
                        # Get the stored IDs and vectors
                        ids = []
                        vectors = []
                        for i in range(src_index.ntotal):
                            idx = src_index.id_map.at(i)
                            ids.append(idx)
                            vectors.append(src_index.reconstruct(idx))

                        if ids and vectors:
                            ids_np = np.array(ids, dtype=np.int64)
                            vectors_np = np.array(vectors, dtype=np.float32)
                            target_index.add_with_ids(vectors_np, ids_np)
                            merged_stats[src_id] = {
                                "vectors_added": len(ids),
                                "type": "add_with_ids"
                            }
                        else:
                            logger.warning(f"No IDs or vectors found in source index {src_id}")
                    except Exception as e:
                        logger.error(f"Error extracting IDs and vectors: {e}")
                        # Fall back to standard extraction
                        vectors = extract_vectors(src_index, 0, src_index.ntotal)
                        target_index.add(vectors)
                        merged_stats[src_id] = {
                            "vectors_added": src_index.ntotal,
                            "type": "extract_add"
                        }
                else:
                    # Source is not IDMap - just add vectors
                    vectors = extract_vectors(src_index, 0, src_index.ntotal)
                    target_index.add(vectors)
                    merged_stats[src_id] = {
                        "vectors_added": src_index.ntotal,
                        "type": "extract_add"
                    }

            else:
                # Other index types - extract and add
                vectors = extract_vectors(src_index, 0, src_index.ntotal)
                target_index.add(vectors)
                merged_stats[src_id] = {
                    "vectors_added": src_index.ntotal,
                    "type": "extract_add"
                }

            # Check that vectors were actually added
            src_ntotal_after = target_index.ntotal
            vectors_added = src_ntotal_after - src_ntotal_before
            merged_stats[src_id]["actual_added"] = vectors_added

            if vectors_added != merged_stats[src_id]["vectors_added"]:
                logger.warning(
                    f"Expected to add {merged_stats[src_id]['vectors_added']} "
                    f"vectors, but actually added {vectors_added}"
                )

        # Prepare success response
        return {
            "success": True,
            "message": (
                f"Successfully merged {len(source_index_ids)} indices into {target_index_id}, "
                f"new total: {target_index.ntotal} vectors"
            ),
            "ntotal": target_index.ntotal,
            "merged_stats": merged_stats
        }

    except Exception as e:
        return {"success": False, "error": f"Error merging indices: {str(e)}"}


def extract_vectors(index, start_idx=0, num_vectors=None):
    """
    Extract vectors from an index, with optimized handling for different index types.

    Args:
        index: FAISS index to extract vectors from
        start_idx: Starting index for extraction
        num_vectors: Number of vectors to extract (None for all remaining)

    Returns:
        numpy.ndarray: Extracted vectors as float32 array
    """
    if num_vectors is None:
        num_vectors = index.ntotal - start_idx

    if num_vectors <= 0:
        return np.array([], dtype=np.float32)

    # Check if this is a binary index
    is_binary = False
    for binary_class in [faiss.IndexBinaryFlat, faiss.IndexBinaryIVF, faiss.IndexBinaryHash]:
        if isinstance(index, binary_class):
            is_binary = True
            break

    # Handle binary indices differently
    if is_binary:
        # For binary indices, extraction is more complex
        # We need to reconstruct each vector individually
        dimension_bytes = index.d // 8
        binary_vectors = np.zeros((num_vectors, dimension_bytes), dtype=np.uint8)

        for i in range(num_vectors):
            idx = start_idx + i
            if idx < index.ntotal:
                try:
                    index.reconstruct(idx, binary_vectors[i])
                except:
                    # Some binary indices might not support reconstruction
                    logger.warning(f"Failed to reconstruct binary vector at index {idx}")

        return binary_vectors

    # For standard float indices
    # Check if index has reconstruct_n method for efficient extraction
    if hasattr(index, "reconstruct_n"):
        try:
            return index.reconstruct_n(start_idx, num_vectors)
        except Exception as e:
            logger.warning(f"reconstruct_n failed, falling back to individual reconstruction: {e}")

    # Fall back to reconstructing vectors one by one
    dimension = index.d
    vectors = np.zeros((num_vectors, dimension), dtype=np.float32)

    for i in range(num_vectors):
        idx = start_idx + i
        if idx < index.ntotal:
            try:
                vectors[i] = index.reconstruct(idx)
            except Exception as e:
                logger.warning(f"Failed to reconstruct vector at index {idx}: {e}")

    return vectors


def compute_clustering(server=None, vectors=None, n_clusters=None, metric_type=faiss.METRIC_L2, niter=25):
    """
    Compute k-means clustering on a set of vectors.

    This is useful for IVF index construction and data analysis.

    Args:
        server: FaissIndex server instance (optional, may be passed by action handler)
        vectors: Input vectors as numpy array
        n_clusters: Number of clusters to compute
        metric_type: Distance metric to use
        niter: Number of iterations for k-means

    Returns:
        dict: Response containing centroids and cluster assignments
    """
    try:
        # Check if we were called without vectors (happens when called via action handler)
        if vectors is None:
            return {"success": False, "error": "No vectors provided for clustering"}

        if n_clusters is None:
            return {"success": False, "error": "Number of clusters not specified"}

        # Convert to numpy array if needed
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        # Ensure we have float32 vectors
        vectors = vectors.astype(np.float32)

        # Get dimensions and validate
        n_vectors, dimension = vectors.shape
        if n_vectors < n_clusters:
            return {
                "success": False,
                "error": f"Number of vectors ({n_vectors}) must be >= number of clusters ({n_clusters})"
            }

        # Convert metric_type from string to FAISS constant if needed
        if isinstance(metric_type, str):
            if metric_type.upper() == "L2":
                metric_type = faiss.METRIC_L2
            elif metric_type.upper() in ["IP", "INNER_PRODUCT"]:
                metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                return {"success": False, "error": f"Unknown metric type: {metric_type}"}

        # Create kmeans object
        kmeans = faiss.Kmeans(dimension, n_clusters, niter=niter, verbose=False, gpu=False)
        kmeans.train(vectors)

        # Compute cluster assignments
        _, assignments = kmeans.index.search(vectors, 1)
        assignments = assignments.flatten()

        # Count vectors per cluster
        cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
        for c in assignments:
            cluster_sizes[c] += 1

        return {
            "success": True,
            "centroids": kmeans.centroids.tolist(),  # Convert to list for JSON serialization
            "assignments": assignments.tolist(),
            "cluster_sizes": cluster_sizes.tolist(),
            "n_clusters": n_clusters,
            "dimension": dimension,
            "obj": float(kmeans.obj[-1])  # Final objective value
        }

    except Exception as e:
        logger.exception(f"Error in compute_clustering: {e}")
        return {"success": False, "error": f"Error computing clusters: {str(e)}"}


def recluster_index(server, index_id, n_clusters=None, sample_ratio=0.5):
    """
    Re-cluster an index to optimize its structure.

    This is useful for optimizing IVF indices after they have changed significantly.

    Args:
        server: FaissIndex server instance
        index_id: ID of the index to recluster
        n_clusters: Number of clusters (None to use existing)
        sample_ratio: Ratio of vectors to sample for clustering

    Returns:
        dict: Response indicating success or failure
    """
    if index_id not in server.indexes:
        return {"success": False, "error": f"Index {index_id} not found"}

    index = server.indexes[index_id]

    # Only IVF indices can be reclustered
    if not isinstance(index, faiss.IndexIVF):
        return {
            "success": False,
            "error": f"Reclustering only supported for IVF indices, not {type(index).__name__}"
        }

    # Get vectors for clustering
    ntotal = index.ntotal
    if ntotal == 0:
        return {"success": False, "error": "Index is empty, cannot recluster"}

    # Determine number of clusters
    if n_clusters is None:
        n_clusters = index.nlist

    # Determine sample size
    sample_size = max(n_clusters * 10, int(ntotal * sample_ratio))
    sample_size = min(sample_size, ntotal)

    try:
        # Sample vectors for clustering
        indices = np.random.choice(ntotal, sample_size, replace=False)
        vectors = np.zeros((sample_size, index.d), dtype=np.float32)

        for i, idx in enumerate(indices):
            vectors[i] = index.reconstruct(idx)

        # Compute new clusters
        clustering_result = compute_clustering(vectors=vectors, n_clusters=n_clusters, metric_type=index.metric_type)
        if not clustering_result["success"]:
            return clustering_result

        # Create a new index with the same parameters but new clusters
        new_index = None

        if isinstance(index, faiss.IndexIVFFlat):
            # For IVFFlat we can create a direct replacement
            quantizer = faiss.IndexFlat(index.d, index.metric_type)
            quantizer.add(clustering_result["centroids"])
            new_index = faiss.IndexIVFFlat(quantizer, index.d, n_clusters, index.metric_type)

        elif isinstance(index, faiss.IndexIVFPQ):
            # For IVFPQ we need to preserve PQ parameters
            quantizer = faiss.IndexFlat(index.d, index.metric_type)
            quantizer.add(clustering_result["centroids"])
            m = index.pq.M
            nbits = index.pq.nbits
            new_index = faiss.IndexIVFPQ(quantizer, index.d, n_clusters, m, nbits, index.metric_type)

        elif isinstance(index, faiss.IndexIVFScalarQuantizer):
            # For IVFSQ we need to preserve SQ parameters
            quantizer = faiss.IndexFlat(index.d, index.metric_type)
            quantizer.add(clustering_result["centroids"])
            sq_type = index.sq_type
            new_index = faiss.IndexIVFScalarQuantizer(
                quantizer, index.d, n_clusters, sq_type, index.metric_type
            )

        if new_index is None:
            return {
                "success": False,
                "error": f"Reclustering not supported for {type(index).__name__}"
            }

        # Train the new index (quantizer is already trained)
        new_index.is_trained = True
        new_index.nprobe = index.nprobe

        # Add all vectors to the new index
        all_vectors = extract_vectors(index, 0, ntotal)
        new_index.add(all_vectors)

        # Replace the old index
        server.indexes[index_id] = new_index

        return {
            "success": True,
            "message": f"Index {index_id} reclustered with {n_clusters} clusters",
            "previous_nlist": index.nlist,
            "new_nlist": n_clusters,
            "ntotal": new_index.ntotal
        }

    except Exception as e:
        logger.exception(f"Error in recluster_index: {e}")
        return {"success": False, "error": f"Error reclustering index: {str(e)}"}


def hybrid_search(server, index_id, query_vectors, vector_weight=0.5,
                  metadata_filter=None, k=10, params=None):
    """
    Perform hybrid search combining vector similarity with metadata filtering.

    This allows for more expressive queries that combine semantic similarity
    with exact metadata constraints.

    Args:
        server: FaissIndex server instance
        index_id: ID of the index to search in
        query_vectors: Query vectors for similarity search
        vector_weight: Weight given to vector similarity vs metadata (0.0-1.0)
        metadata_filter: Filter expression for metadata
        k: Number of results to return
        params: Additional search parameters

    Returns:
        dict: Response containing search results
    """
    if index_id not in server.indexes:
        return {"success": False, "error": f"Index {index_id} not found"}

    try:
        # First, perform vector search
        search_result = server._search(index_id, query_vectors, k=k*2, params=params)
        if not search_result.get("success", False):
            return search_result

        # If no metadata filter, just return the vector search results
        if metadata_filter is None:
            return search_result

        # Extract results
        results = search_result.get("results", [])
        if not results:
            return search_result

        # Apply metadata filtering to each query result
        filtered_results = []

        for query_result in results:
            distances = query_result.get("distances", [])
            indices = query_result.get("indices", [])

            # Skip if no results
            if not indices:
                filtered_results.append({"distances": [], "indices": []})
                continue

            # Apply metadata filtering and rescoring
            filtered_distances = []
            filtered_indices = []

            for i, (dist, idx) in enumerate(zip(distances, indices)):
                # Skip invalid indices
                if idx < 0 or idx >= server.indexes[index_id].ntotal:
                    continue

                # Check metadata filter if defined
                if metadata_filter:
                    # TODO: Implement metadata filtering logic
                    # This is a placeholder for future implementation
                    pass

                # Add to filtered results
                filtered_distances.append(dist)
                filtered_indices.append(idx)

                # Stop once we have enough results
                if len(filtered_indices) >= k:
                    break

            # Add query result to filtered results
            filtered_results.append({
                "distances": filtered_distances,
                "indices": filtered_indices
            })

        # Return filtered results
        return {
            "success": True,
            "results": filtered_results,
            "num_queries": len(query_vectors),
            "k": k,
            "filtered": True,
            "vector_weight": vector_weight
        }

    except Exception as e:
        logger.exception(f"Error in hybrid_search: {e}")
        return {"success": False, "error": f"Error in hybrid search: {str(e)}"}


def batch_add_with_ids(server, index_id, vectors, ids, batch_size=1000):
    """
    Add vectors with IDs to an index in batches.

    Args:
        server: FaissIndex server instance
        index_id: ID of the index
        vectors: Vectors to add
        ids: IDs to associate with vectors
        batch_size: Batch size for adding

    Returns:
        dict: Response indicating success or failure
    """
    if index_id not in server.indexes:
        return {"success": False, "error": f"Index {index_id} not found"}

    index = server.indexes[index_id]

    # Check if this is an IDMap index
    if not isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
        return {
            "success": False,
            "error": f"add_with_ids requires an IDMap index, but {index_id} is {type(index).__name__}"
        }

    try:
        # Convert to numpy arrays
        vectors_np = np.array(vectors, dtype=np.float32)
        ids_np = np.array(ids, dtype=np.int64)

        # Check dimensions
        if vectors_np.shape[1] != server.dimensions[index_id]:
            return {
                "success": False,
                "error": (
                    f"Vector dimension mismatch: expected {server.dimensions[index_id]}, "
                    f"got {vectors_np.shape[1]}"
                )
            }

        # Check if vectors and IDs have same length
        if len(vectors_np) != len(ids_np):
            return {
                "success": False,
                "error": (
                    f"Number of vectors ({len(vectors_np)}) doesn't match "
                    f"number of IDs ({len(ids_np)})"
                )
            }

        # Add in batches
        total_vectors = len(vectors_np)
        for i in range(0, total_vectors, batch_size):
            batch_vectors = vectors_np[i:i+batch_size]
            batch_ids = ids_np[i:i+batch_size]
            index.add_with_ids(batch_vectors, batch_ids)

        return {
            "success": True,
            "ntotal": index.ntotal,
            "total": index.ntotal,
            "count": len(vectors_np),
            "message": f"Added {len(vectors_np)} vectors with IDs to index {index_id}"
        }

    except Exception as e:
        logger.exception(f"Error in batch_add_with_ids: {e}")
        return {"success": False, "error": f"Error adding vectors with IDs: {str(e)}"}


def optimize_index(server, index_id, optimization_level=1):
    """
    Optimize an index for better performance based on its current state.

    This automatically applies appropriate optimizations based on the index type.

    Args:
        server: FaissIndex server instance
        index_id: ID of the index to optimize
        optimization_level: Level of optimization (1=basic, 2=moderate, 3=aggressive)

    Returns:
        dict: Response indicating optimizations applied
    """
    if index_id not in server.indexes:
        return {"success": False, "error": f"Index {index_id} not found"}

    index = server.indexes[index_id]
    optimizations = []

    try:
        # Check if index needs training
        is_ivf = isinstance(index, faiss.IndexIVF)

        # If it's an IVF index that needs training and has no vectors
        if is_ivf and not index.is_trained and index.ntotal == 0:
            return {
                "success": False,
                "error": "Index requires training data before optimization"
            }

        # If it's an IVF index that needs training and has vectors, train it
        if is_ivf and not index.is_trained and index.ntotal > 0:
            # Get some vectors for training
            num_train = min(index.ntotal, 1000)  # Use up to 1000 vectors
            train_vectors = extract_vectors(index, 0, num_train)

            # Train the index
            index.train(train_vectors)
            optimizations.append("trained_index")

        # Optimize IVF indices
        if isinstance(index, faiss.IndexIVF):
            # Set nprobe based on optimization level and number of clusters
            if optimization_level == 1:
                nprobe = max(1, min(10, index.nlist // 10))
            elif optimization_level == 2:
                nprobe = max(1, min(50, index.nlist // 5))
            else:  # level 3
                nprobe = max(1, min(100, index.nlist // 2))

            index.nprobe = nprobe
            optimizations.append(f"set_nprobe_{nprobe}")

            # Basic parameter tuning
            if hasattr(index, "quantizer") and hasattr(index.quantizer, "efSearch"):
                # HNSW quantizer
                if optimization_level == 1:
                    index.quantizer.efSearch = 40
                elif optimization_level == 2:
                    index.quantizer.efSearch = 80
                else:
                    index.quantizer.efSearch = 120
                optimizations.append(f"set_efSearch_{index.quantizer.efSearch}")

        # Optimize HNSW indices
        elif isinstance(index, faiss.IndexHNSW):
            if hasattr(index, "efSearch"):
                if optimization_level == 1:
                    index.efSearch = 40
                elif optimization_level == 2:
                    index.efSearch = 80
                else:
                    index.efSearch = 120
                optimizations.append(f"set_efSearch_{index.efSearch}")

        # Optimize flat indices with direct access
        elif isinstance(index, faiss.IndexFlat):
            # Not much to optimize for flat indices
            optimizations.append("flat_index_no_optimization")

        # Return results
        return {
            "success": True,
            "message": f"Applied {len(optimizations)} optimizations to index {index_id}",
            "index_type": type(index).__name__,
            "optimizations": optimizations,
            "optimization_level": optimization_level
        }

    except Exception as e:
        logger.exception(f"Error in optimize_index: {e}")
        return {"success": False, "error": f"Error optimizing index: {str(e)}"}

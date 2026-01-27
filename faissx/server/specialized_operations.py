#!/usr/bin/env python3
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
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Constants for specialized operations
DEFAULT_CLUSTERING_ITERATIONS = 25
DEFAULT_MERGE_BATCH_SIZE = 1000
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_SAMPLE_RATIO = 0.5
MIN_TRAINING_VECTORS_FACTOR = 10
DEFAULT_NPROBE_FACTOR = 10

# Binary index class names for type checking
BINARY_INDEX_CLASSES = [
    faiss.IndexBinaryFlat,
    faiss.IndexBinaryIVF,
    faiss.IndexBinaryHash
]


def merge_indices(
    server: Any,
    target_index_id: str,
    source_index_ids: List[str]
) -> Dict[str, Any]:
    """
    Merge multiple source indices into a target index.

    This is an optimized implementation that handles more index types and
    provides better error reporting and validation. The function supports
    merging different FAISS index types by using appropriate strategies:
    - Direct merging for compatible index types (e.g., IndexFlat to IndexFlat)
    - Vector extraction and re-addition for incompatible types
    - Special handling for IDMap indices to preserve ID mappings
    - Automatic fallback mechanisms when direct merging fails

    The merge process preserves all vectors from source indices while maintaining
    the target index's configuration (quantization, clustering, etc.).

    Args:
        server: FaissIndex server instance containing the indices to merge
        target_index_id: ID of the target index to merge into. This index will
                        receive all vectors from the source indices.
        source_index_ids: List of source index IDs to merge from. All vectors
                         from these indices will be copied to the target.

    Returns:
        dict: Response indicating success or failure with the following structure:
            - success (bool): Whether the merge operation completed successfully
            - message (str): Human-readable description of the result
            - ntotal (int): Total number of vectors in target after merge
            - merged_stats (dict): Per-source statistics about the merge process
            - error (str, optional): Error description if success is False

    Implementation Strategy:
        The function uses different merge strategies based on index compatibility:
        1. For IndexFlat types: Direct vector extraction and addition
        2. For IndexIVF types: Checks cluster compatibility before merging
        3. For IDMap types: Preserves ID mappings during the merge process
        4. Fallback: Extract vectors and add them to target index

    Performance Considerations:
        - Large indices may require significant memory during vector extraction
        - Merge time scales with the total number of vectors being merged
        - IVF indices may need retraining after merge for optimal performance
    """
    # Input validation: Ensure we have source indices to merge
    if not source_index_ids:
        return {
            "success": False,
            "error": "No source indices provided for merging"
        }

    # Validate that the target index exists in the server
    if target_index_id not in server.indexes:
        return {
            "success": False,
            "error": f"Target index {target_index_id} not found"
        }

    # Validate that all source indices exist before starting the merge
    # This prevents partial merges that could leave the target in an inconsistent state
    for src_id in source_index_ids:
        if src_id not in server.indexes:
            return {
                "success": False,
                "error": f"Source index {src_id} not found"
            }

    # Get references to the target index and its dimension for validation
    target_index = server.indexes[target_index_id]
    target_dim = server.dimensions[target_index_id]

    # Track merge statistics for each source index
    # This provides detailed feedback about the merge process
    merged_stats = {}

    try:
        # Process each source index individually to enable partial recovery
        for src_id in source_index_ids:
            src_index = server.indexes[src_id]
            src_dim = server.dimensions[src_id]

            # Record the target's vector count before merging this source
            # This allows us to calculate how many vectors were actually added
            src_ntotal_before = target_index.ntotal

            # Dimension compatibility check: All indices must have same dimension
            # This is a fundamental requirement for vector operations
            if src_dim != target_dim:
                return {
                    "success": False,
                    "error": (
                        f"Dimension mismatch: target dimension is {target_dim}, "
                        f"but source index {src_id} has dimension {src_dim}"
                    )
                }

            # Handle different index types with specialized merge strategies
            # Each index type has different characteristics and optimal merge approaches
            if isinstance(target_index, faiss.IndexFlat):
                # IndexFlat: Stores vectors directly in memory without compression
                # This is the simplest case for merging as vectors can be directly copied
                if isinstance(src_index, faiss.IndexFlat):
                    try:
                        # Direct flat-to-flat merging: extract all vectors and add them
                        # This is the most efficient path for flat indices
                        vectors = extract_vectors(src_index, 0, src_index.ntotal)
                        # Add extracted vectors directly to the target index
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
                    # Source is different type: use general extraction approach
                    # This handles cases like IVF->Flat, HNSW->Flat, etc.
                    vectors = extract_vectors(src_index, 0, src_index.ntotal)
                    target_index.add(vectors)
                    merged_stats[src_id] = {
                        "vectors_added": src_index.ntotal,
                        "type": "extract_add"
                    }

            elif isinstance(target_index, faiss.IndexIVF):
                # IndexIVF: Uses inverted file structure with clustering
                # Merge strategy depends on cluster compatibility
                if isinstance(src_index, faiss.IndexIVF):
                    # Both indices use IVF structure - check cluster compatibility
                    if target_index.nlist != src_index.nlist:
                        # Different cluster counts: cannot use direct IVF merging
                        # Fall back to vector extraction and re-addition
                        vectors = extract_vectors(src_index, 0, src_index.ntotal)
                        target_index.add(vectors)
                        merged_stats[src_id] = {
                            "vectors_added": src_index.ntotal,
                            "type": "extract_add"
                        }
                    else:
                        # Compatible cluster counts: try FAISS native merge_from
                        # This is more efficient as it preserves the clustering structure
                        try:
                            target_index.merge_from(src_index, 0)
                            merged_stats[src_id] = {
                                "vectors_added": src_index.ntotal,
                                "type": "merge_from"
                            }
                        except AttributeError:
                            # merge_from not available: fall back to extract and add
                            vectors = extract_vectors(src_index, 0, src_index.ntotal)
                            target_index.add(vectors)
                            merged_stats[src_id] = {
                                "vectors_added": src_index.ntotal,
                                "type": "extract_add"
                            }
                else:
                    # Source is not IVF: use general extraction approach
                    # Vectors will be re-clustered according to target's quantizer
                    vectors = extract_vectors(src_index, 0, src_index.ntotal)
                    target_index.add(vectors)
                    merged_stats[src_id] = {
                        "vectors_added": src_index.ntotal,
                        "type": "extract_add"
                    }

            elif isinstance(target_index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                # IndexIDMap: Wrapper that adds ID mapping to any base index
                # Special handling required to preserve vector-to-ID associations
                if isinstance(src_index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                    # Both indices have ID mapping: preserve ID associations during merge
                    try:
                        # Extract both vectors and their associated IDs
                        # This is more complex than simple vector extraction
                        ids = []
                        vectors = []

                        # Iterate through all vectors in the source IDMap
                        for i in range(src_index.ntotal):
                            # Get the actual ID for this vector position
                            idx = src_index.id_map.at(i)
                            ids.append(idx)
                            # Reconstruct the vector using its ID
                            vectors.append(src_index.reconstruct(idx))

                        if ids and vectors:
                            # Convert to numpy arrays for efficient batch operations
                            ids_np = np.array(ids, dtype=np.int64)
                            vectors_np = np.array(vectors, dtype=np.float32)
                            # Add vectors with their original IDs preserved
                            target_index.add_with_ids(vectors_np, ids_np)
                            merged_stats[src_id] = {
                                "vectors_added": len(ids),
                                "type": "add_with_ids"
                            }
                        else:
                            logger.warning(f"No IDs or vectors found in source index {src_id}")
                    except Exception as e:
                        logger.error(f"Error extracting IDs and vectors: {e}")
                        # Fall back to standard extraction (IDs will be lost)
                        vectors = extract_vectors(src_index, 0, src_index.ntotal)
                        target_index.add(vectors)
                        merged_stats[src_id] = {
                            "vectors_added": src_index.ntotal,
                            "type": "extract_add"
                        }
                else:
                    # Source has no ID mapping: just add vectors (auto-assign IDs)
                    vectors = extract_vectors(src_index, 0, src_index.ntotal)
                    target_index.add(vectors)
                    merged_stats[src_id] = {
                        "vectors_added": src_index.ntotal,
                        "type": "extract_add"
                    }

            else:
                # Generic fallback for other index types (HNSW, PQ, etc.)
                # Use universal vector extraction and addition approach
                vectors = extract_vectors(src_index, 0, src_index.ntotal)
                target_index.add(vectors)
                merged_stats[src_id] = {
                    "vectors_added": src_index.ntotal,
                    "type": "extract_add"
                }

            # Verify that vectors were actually added by checking count difference
            # This helps detect silent failures in the merge process
            src_ntotal_after = target_index.ntotal
            vectors_added = src_ntotal_after - src_ntotal_before
            merged_stats[src_id]["actual_added"] = vectors_added

            # Log warning if expected vs actual counts don't match
            # This can indicate issues with the merge process
            if vectors_added != merged_stats[src_id]["vectors_added"]:
                logger.warning(
                    f"Expected to add {merged_stats[src_id]['vectors_added']} "
                    f"vectors, but actually added {vectors_added}"
                )

        # Prepare comprehensive success response with detailed statistics
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
        # Handle any unexpected errors during the merge process
        return {"success": False, "error": f"Error merging indices: {str(e)}"}


def extract_vectors(
    index: Any,
    start_idx: int = 0,
    num_vectors: Optional[int] = None
) -> np.ndarray:
    """
    Extract vectors from an index, with optimized handling for different index types.

    This function provides a unified interface for extracting vectors from any FAISS
    index type. It automatically detects the index characteristics and uses the most
    efficient extraction method available. The function handles both binary and float
    indices, with special optimizations for bulk extraction when supported.

    The extraction process preserves the original vector format and precision,
    making it suitable for index merging, backup operations, and data migration.

    Args:
        index: FAISS index to extract vectors from. Can be any FAISS index type
               including IndexFlat, IndexIVF, IndexHNSW, IndexPQ, etc.
        start_idx: Starting index for extraction (0-based). Must be less than
                  the total number of vectors in the index.
        num_vectors: Number of vectors to extract starting from start_idx.
                    If None, extracts all remaining vectors from start_idx.

    Returns:
        numpy.ndarray: Extracted vectors in their original format:
                      - For float indices: float32 array of shape (n_vectors, dimension)
                      - For binary indices: uint8 array of shape (n_vectors, bytes_per_vector)

    Performance Notes:
        - Uses reconstruct_n() when available for efficient bulk extraction
        - Falls back to individual reconstruction for unsupported index types
        - Binary indices require individual vector reconstruction
        - Memory usage scales with the number of vectors being extracted

    Error Handling:
        - Invalid indices are skipped with warnings logged
        - Reconstruction failures are logged but don't stop the extraction
        - Returns partial results if some vectors cannot be reconstructed
    """
    if num_vectors is None:
        num_vectors = index.ntotal - start_idx

    # Return empty array if no vectors to extract
    if num_vectors <= 0:
        return np.array([], dtype=np.float32)

    # Detect if this is a binary index by checking against known binary types
    # Binary indices store vectors as packed bits rather than float values
    is_binary = any(isinstance(index, binary_class) for binary_class in BINARY_INDEX_CLASSES)

    # Handle binary indices with specialized extraction logic
    if is_binary:
        # Binary indices are more complex to extract from because:
        # 1. Vectors are stored as packed bits (8 bits per byte)
        # 2. Individual reconstruction is often the only available method
        # 3. The output format is uint8 bytes rather than float32 values

        # Calculate how many bytes are needed per vector
        dimension_bytes = index.d // 8
        # Pre-allocate array for all binary vectors
        binary_vectors = np.zeros((num_vectors, dimension_bytes), dtype=np.uint8)

        # Extract each binary vector individually
        # Most binary indices don't support bulk reconstruction methods
        for i in range(num_vectors):
            idx = start_idx + i
            # Ensure we don't exceed the index bounds
            if idx < index.ntotal:
                try:
                    # Reconstruct binary vector directly into pre-allocated array
                    # This modifies binary_vectors[i] in-place
                    index.reconstruct(idx, binary_vectors[i])
                except Exception as e:
                    # Some binary indices might not support reconstruction
                    # Log the error but continue with remaining vectors
                    logger.warning(f"Failed to reconstruct binary vector at index {idx}: {e}")

        return binary_vectors

    # Handle standard float indices with optimized extraction strategies
    # Float indices typically offer more extraction options than binary indices

    # Try to use bulk reconstruction method first (most efficient)
    # reconstruct_n() can extract multiple vectors in a single call
    if hasattr(index, "reconstruct_n"):
        try:
            # Bulk extraction: much faster than individual reconstruction
            return index.reconstruct_n(start_idx, num_vectors)
        except Exception as e:
            # Log the failure but don't give up - fall back to individual method
            logger.warning(f"reconstruct_n failed, falling back to individual reconstruction: {e}")

    # Fall back to individual vector reconstruction
    # This is slower but works with all index types that support reconstruction
    dimension = index.d
    # Pre-allocate array for all float vectors
    vectors = np.zeros((num_vectors, dimension), dtype=np.float32)

    # Extract each vector individually using reconstruct() method
    for i in range(num_vectors):
        idx = start_idx + i
        # Ensure we don't exceed the index bounds
        if idx < index.ntotal:
            try:
                # Reconstruct individual vector and store in pre-allocated array
                vectors[i] = index.reconstruct(idx)
            except Exception as e:
                # Log reconstruction failures but continue with remaining vectors
                # This allows partial extraction even if some vectors are corrupted
                logger.warning(f"Failed to reconstruct vector at index {idx}: {e}")

    return vectors


def compute_clustering(
    server: Optional[Any] = None,
    vectors: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
    metric_type: int = faiss.METRIC_L2,
    niter: int = DEFAULT_CLUSTERING_ITERATIONS
) -> Dict[str, Any]:
    """
    Compute k-means clustering on a set of vectors.

    This function performs k-means clustering using FAISS's optimized implementation.
    It's particularly useful for creating quantizers for IVF indices, analyzing data
    distribution, and understanding vector patterns in high-dimensional spaces.

    The clustering process uses Lloyd's algorithm (standard k-means) with configurable
    parameters for convergence criteria and distance metrics. The function supports
    both L2 (Euclidean) and inner product distance metrics.

    Args:
        server: FaissIndex server instance (optional, may be passed by action handler).
               Not used in the current implementation but kept for API compatibility.
        vectors: Input vectors as numpy array of shape (n_vectors, dimension).
                Must be float32 dtype for optimal performance with FAISS.
        n_clusters: Number of clusters to compute. Must be <= number of input vectors.
                   Typical values range from sqrt(n_vectors) to n_vectors/10.
        metric_type: Distance metric to use for clustering:
                    - faiss.METRIC_L2: Euclidean distance (default)
                    - faiss.METRIC_INNER_PRODUCT: Inner product distance
        niter: Number of iterations for k-means convergence. More iterations
              improve quality but increase computation time. Default: 25.

    Returns:
        dict: Response containing clustering results with the following structure:
            - success (bool): Whether clustering completed successfully
            - centroids (list): Cluster centers as list of float lists
            - assignments (list): Vector-to-cluster assignments (0-based indices)
            - cluster_sizes (list): Number of vectors assigned to each cluster
            - n_clusters (int): Number of clusters created
            - dimension (int): Vector dimension
            - obj (float): Final objective value (sum of squared distances)
            - error (str, optional): Error description if success is False

    Algorithm Details:
        1. Initialize centroids randomly or using k-means++ method
        2. Assign each vector to nearest centroid based on chosen metric
        3. Update centroids as mean of assigned vectors
        4. Repeat assignment and update steps until convergence or max iterations
        5. Return final centroids, assignments, and quality metrics

    Performance Considerations:
        - Computation time scales with O(n_vectors * n_clusters * dimension * niter)
        - Memory usage scales with O(n_vectors * dimension + n_clusters * dimension)
        - Large datasets may benefit from sampling before clustering
        - FAISS uses optimized BLAS operations for better performance

    Quality Guidelines:
        - Ensure n_clusters << n_vectors for meaningful results
        - Use more iterations (niter) for better convergence
        - Consider the curse of dimensionality for high-dimensional data
        - Evaluate clustering quality using the returned objective value
    """
    try:
        # Validate that vectors were provided for clustering
        # This is the primary input and is required for any clustering operation
        if vectors is None:
            return {"success": False, "error": "No vectors provided for clustering"}

        # Validate that the number of clusters was specified
        if n_clusters is None:
            return {"success": False, "error": "Number of clusters not specified"}

        # Convert input to numpy array if needed and ensure proper data type
        # FAISS requires float32 for optimal performance and compatibility
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        # Ensure we have float32 vectors for FAISS compatibility
        vectors = vectors.astype(np.float32)

        # Extract and validate vector dimensions
        n_vectors, dimension = vectors.shape

        # Check if we have enough vectors for the requested number of clusters
        # This is a fundamental requirement for meaningful clustering
        if n_vectors < n_clusters:
            return {
                "success": False,
                "error": (
                    f"Number of vectors ({n_vectors}) must be >= "
                    f"number of clusters ({n_clusters})"
                )
            }

        # Handle metric type conversion from string to FAISS constants
        # This provides flexibility in the API while using FAISS internal constants
        if isinstance(metric_type, str):
            if metric_type.upper() == "L2":
                metric_type = faiss.METRIC_L2
            elif metric_type.upper() in ["IP", "INNER_PRODUCT"]:
                metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                return {"success": False, "error": f"Unknown metric type: {metric_type}"}

        # Create and configure the k-means clustering object
        # FAISS Kmeans provides optimized clustering with various configuration options
        kmeans = faiss.Kmeans(
            dimension,          # Vector dimension
            n_clusters,         # Number of clusters to create
            niter=niter,        # Maximum iterations for convergence
            verbose=False,      # Disable progress output for cleaner logs
            gpu=False           # Use CPU implementation for broader compatibility
        )

        # Perform the clustering operation
        # This is the main computational step and may take significant time
        kmeans.train(vectors)

        # Compute cluster assignments for all input vectors
        # This determines which cluster each vector belongs to
        _, assignments = kmeans.index.search(vectors, 1)
        assignments = assignments.flatten()  # Convert to 1D array

        # Calculate cluster sizes for quality assessment
        # This helps identify empty clusters or highly imbalanced assignments
        cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
        for c in assignments:
            cluster_sizes[c] += 1

        # Return comprehensive clustering results
        return {
            "success": True,
            "centroids": kmeans.centroids.tolist(),    # Convert to list for JSON serialization
            "assignments": assignments.tolist(),       # Vector-to-cluster mapping
            "cluster_sizes": cluster_sizes.tolist(),   # Cluster size distribution
            "n_clusters": n_clusters,                  # Requested number of clusters
            "dimension": dimension,                    # Vector dimension
            "obj": float(kmeans.obj[-1])              # Final objective value (clustering quality)
        }

    except Exception as e:
        # Handle any unexpected errors during clustering
        logger.exception(f"Error in compute_clustering: {e}")
        return {"success": False, "error": f"Error computing clusters: {str(e)}"}


def recluster_index(
    server: Any,
    index_id: str,
    n_clusters: Optional[int] = None,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO
) -> Dict[str, Any]:
    """
    Re-cluster an index to optimize its structure.

    This function is particularly useful for optimizing IVF indices after they have
    changed significantly through vector additions or when the original clustering
    is no longer optimal. The process involves sampling vectors from the current
    index, computing new optimal clusters, and rebuilding the index structure.

    Re-clustering can dramatically improve search performance when:
    - The index has grown significantly since creation
    - Vector distribution has changed due to new data patterns
    - The original cluster count was suboptimal
    - Search performance has degraded over time

    The function preserves all existing vectors while optimizing the underlying
    clustering structure for better search performance and accuracy.

    Args:
        server: FaissIndex server instance containing the index to recluster
        index_id: ID of the index to recluster. Must be an IVF index type.
        n_clusters: Number of clusters for the new structure. If None, uses the
                   current cluster count. Consider increasing for larger indices.
        sample_ratio: Ratio of vectors to sample for clustering (0.0-1.0).
                     Higher ratios provide better cluster quality but increase
                     computation time. Default: 0.5 (50% of vectors).

    Returns:
        dict: Response indicating success or failure with the following structure:
            - success (bool): Whether re-clustering completed successfully
            - message (str): Human-readable description of the result
            - previous_nlist (int): Original number of clusters
            - new_nlist (int): New number of clusters after re-clustering
            - ntotal (int): Total number of vectors preserved
            - error (str, optional): Error description if success is False

    Performance Impact:
        - Search quality typically improves with better clustering
        - Re-clustering time scales with index size and sample ratio
        - Memory usage temporarily doubles during the rebuild process
        - New index inherits original parameters (quantization, etc.)

    Supported Index Types:
        - IndexIVFFlat: Basic IVF with flat quantization
        - IndexIVFPQ: IVF with product quantization
        - IndexIVFScalarQuantizer: IVF with scalar quantization

    Algorithm Steps:
        1. Sample vectors from the existing index
        2. Compute optimal clusters using k-means
        3. Create new index with same parameters but new clusters
        4. Transfer all vectors to the new index
        5. Replace the original index with the optimized version
    """
    # Validate that the specified index exists
    if index_id not in server.indexes:
        return {"success": False, "error": f"Index {index_id} not found"}

    index = server.indexes[index_id]

    # Verify that this is an IVF index type that supports re-clustering
    # Only IVF indices have a clustering structure that can be optimized
    if not isinstance(index, faiss.IndexIVF):
        return {
            "success": False,
            "error": (
                f"Reclustering only supported for IVF indices, "
                f"not {type(index).__name__}"
            )
        }

    # Check that the index contains vectors to sample from
    ntotal = index.ntotal
    if ntotal == 0:
        return {"success": False, "error": "Index is empty, cannot recluster"}

    # Determine the number of clusters for the new structure
    # Default to current cluster count if not specified
    if n_clusters is None:
        n_clusters = index.nlist

    # Calculate sample size for clustering
    # Use at least 10 vectors per cluster for meaningful statistics
    # But don't exceed the total number of available vectors
    sample_size = max(n_clusters * MIN_TRAINING_VECTORS_FACTOR, int(ntotal * sample_ratio))
    sample_size = min(sample_size, ntotal)

    try:
        # Sample vectors randomly from the index for clustering analysis
        # Random sampling ensures representative cluster centers
        indices = np.random.choice(ntotal, sample_size, replace=False)
        vectors = np.zeros((sample_size, index.d), dtype=np.float32)

        # Extract the sampled vectors for clustering
        for i, idx in enumerate(indices):
            vectors[i] = index.reconstruct(idx)

        # Compute new optimal clusters using the sampled vectors
        # This determines the new quantizer structure
        clustering_result = compute_clustering(
            vectors=vectors,
            n_clusters=n_clusters,
            metric_type=index.metric_type
        )
        if not clustering_result["success"]:
            return clustering_result

        # Create a new index with the same parameters but optimized clusters
        new_index = None

        if isinstance(index, faiss.IndexIVFFlat):
            # Recreate IVFFlat with new clustering but same parameters
            quantizer = faiss.IndexFlat(index.d, index.metric_type)
            quantizer.add(clustering_result["centroids"])
            new_index = faiss.IndexIVFFlat(quantizer, index.d, n_clusters, index.metric_type)

        elif isinstance(index, faiss.IndexIVFPQ):
            # Recreate IVFPQ preserving product quantization parameters
            quantizer = faiss.IndexFlat(index.d, index.metric_type)
            quantizer.add(clustering_result["centroids"])
            m = index.pq.M          # Number of sub-quantizers
            nbits = index.pq.nbits  # Bits per sub-quantizer
            new_index = faiss.IndexIVFPQ(
                quantizer, index.d, n_clusters, m, nbits, index.metric_type
            )

        elif isinstance(index, faiss.IndexIVFScalarQuantizer):
            # Recreate IVFSQ preserving SQ parameters
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


def hybrid_search(
    server: Any,
    index_id: str,
    query_vectors: List[List[float]],
    vector_weight: float = 0.5,
    metadata_filter: Optional[Dict[str, Any]] = None,
    k: int = 10,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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

    if metadata_filter is not None:
        return {
            "success": False,
            "error": (
                "Metadata filtering is not yet implemented. "
                "Use hybrid_search without metadata_filter for pure "
                "vector similarity search."
            ),
        }

    if vector_weight != 1.0 and vector_weight != 0.5:
        logger.warning(
            "vector_weight is ignored without metadata filtering; "
            "performing pure vector search"
        )

    try:
        return server._search(index_id, query_vectors, k=k, params=params)

    except Exception as e:
        logger.exception(f"Error in hybrid_search: {e}")
        return {"success": False, "error": f"Error in hybrid search: {str(e)}"}


def batch_add_with_ids(
    server: Any,
    index_id: str,
    vectors: List[List[float]],
    ids: List[int],
    batch_size: int = DEFAULT_MERGE_BATCH_SIZE
) -> Dict[str, Any]:
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
            "error": (
                f"add_with_ids requires an IDMap index, but {index_id} is "
                f"{type(index).__name__}"
            )
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


def optimize_index(
    server: Any,
    index_id: str,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL
) -> Dict[str, Any]:
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

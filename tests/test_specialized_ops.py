#!/usr/bin/env python3
#
# FAISSx Client Specialized Operations Test
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
Tests for specialized operations in the FAISSx proxy server
"""

import logging
import os
import sys

import numpy as np

from faissx.client.client import FaissXClient

# Configure logging for test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("specialized_ops_test")

# Server settings from environment variable or default
SERVER_URL = os.environ.get("FAISSX_SERVER", "tcp://localhost:45678")
DIMENSION = 128
NUM_VECTORS = 1000


def generate_test_vectors(num_vectors: int, dimension: int) -> np.ndarray:
    """Generate random test vectors"""
    return np.random.random((num_vectors, dimension)).astype(np.float32)


def get_client() -> FaissXClient:
    """Create and configure a FaissXClient instance"""
    client = FaissXClient()
    client.configure(server=SERVER_URL)
    return client


def test_merge_indices():
    """Test merging multiple indices"""
    # Get client without custom timeout
    client = get_client()

    # Create source indices
    logger.info("Creating source indices for merge test")
    source_indices = []
    for i in range(2):  # Use fewer indices for faster testing
        index_id = f"merge_source_{i}"
        source_indices.append(index_id)

        # Create index
        client.create_index(index_id, DIMENSION, "L2")
        logger.info(f"Created source index {index_id}")

        # Add vectors
        vectors = generate_test_vectors(NUM_VECTORS, DIMENSION)
        result = client.add_vectors(index_id, vectors)
        assert result["success"], f"Failed to add vectors to {index_id}: {result.get('error')}"

        logger.info(f"Added {NUM_VECTORS} vectors to {index_id}")

    # Create target index
    target_index_id = "merge_target"
    client.create_index(target_index_id, DIMENSION, "L2")
    logger.info(f"Created target index {target_index_id}")

    # Merge indices
    logger.info(f"Merging {len(source_indices)} indices into {target_index_id}")
    result = client._send_request({
        "action": "merge_indices",
        "target_index_id": target_index_id,
        "source_index_ids": source_indices
    })

    assert result["success"], f"Failed to merge indices: {result.get('error')}"

    # Verify result
    assert "ntotal" in result, "No ntotal in merge result"
    expected_total = NUM_VECTORS * len(source_indices)
    assert result["ntotal"] == expected_total, \
        f"Expected {expected_total} vectors, got {result['ntotal']}"

    logger.info(f"Successfully merged indices: {result}")

    # Clean up
    for index_id in source_indices + [target_index_id]:
        client.delete_index(index_id)


def test_optimize_index():
    """Test index optimization"""
    client = get_client()

    # Create an IVF index
    index_id = "opt_ivf"
    try:
        # Create the index
        client.create_index(index_id, DIMENSION, "IVF")

        # Generate training vectors - need at least 100 for IVF
        train_vectors = generate_test_vectors(150, DIMENSION)

        # Train the index first before adding vectors
        result = client._send_request({
            "action": "train_index",
            "index_id": index_id,
            "training_vectors": train_vectors.tolist()
        })
        assert result["success"], f"Failed to train index: {result.get('error')}"
        logger.info(f"Successfully trained index {index_id}")

        # Add vectors to index
        vectors = generate_test_vectors(NUM_VECTORS, DIMENSION)
        result = client.add_vectors(index_id, vectors)
        assert result["success"], f"Failed to add vectors to index: {result.get('error')}"

        # Apply optimization
        logger.info("Optimizing index with optimization_level=2")
        result = client._send_request({
            "action": "optimize_index",
            "index_id": index_id,
            "optimization_level": 2
        })

        assert result["success"], f"Optimization failed: {result.get('error')}"
        assert "optimizations" in result, "No optimizations field in response"

        # Verify optimization
        logger.info(f"Applied optimizations: {result['optimizations']}")

    finally:
        # Clean up
        try:
            client.delete_index(index_id)
        except Exception as e:
            logger.error(f"Failed to delete index {index_id}: {str(e)}")


def test_clustering():
    """Test vector clustering"""
    client = get_client()

    # Generate vectors for clustering
    n_clusters = 10
    vectors = generate_test_vectors(500, DIMENSION)

    # Call compute_clustering
    logger.info(f"Computing {n_clusters} clusters on {len(vectors)} vectors")
    # Using _send_request directly since there's no call method
    result = client._send_request({
        "action": "compute_clustering",
        "vectors": vectors.tolist(),
        "n_clusters": n_clusters,
        "metric_type": "L2",
        "niter": 20
    })
    assert result["success"], f"Failed to compute clusters: {result.get('error')}"

    # Verify result
    assert "centroids" in result, "No centroids in result"
    assert "assignments" in result, "No assignments in result"
    assert "cluster_sizes" in result, "No cluster sizes in result"

    # Check if all clusters were assigned
    cluster_sizes = result["cluster_sizes"]
    logger.info(f"Cluster sizes: {cluster_sizes}")
    assert len(cluster_sizes) == n_clusters, (
        f"Expected {n_clusters} clusters, got {len(cluster_sizes)}"
    )
    total_assigned = sum(cluster_sizes)
    assert total_assigned == len(vectors), "Total assigned vectors doesn't match input count"


def test_batch_add_with_ids():
    """Test batch adding vectors with IDs"""
    client = get_client()

    # Create an IDMap index for testing
    index_id = "idmap_test"
    base_index_id = f"{index_id}_base"

    try:
        # Create IDMap index
        client.create_index(index_id, DIMENSION, "IDMap:L2")

        # Generate test vectors and IDs
        vectors = generate_test_vectors(100, DIMENSION)
        ids = list(range(1000, 1000 + len(vectors)))  # Use IDs starting from 1000

        logger.info(f"Testing batch add with {len(vectors)} vectors and IDs")

        # Call batch_add_with_ids
        result = client._send_request({
            "action": "batch_add_with_ids",
            "index_id": index_id,
            "vectors": vectors.tolist(),
            "ids": ids,
            "batch_size": 25  # Use smaller batches for testing
        })

        assert result["success"], f"Batch add failed: {result.get('error')}"
        assert "count" in result, "No count in result"
        assert result["count"] == len(vectors), (
            f"Expected {len(vectors)} vectors added, got {result['count']}"
        )

        logger.info(f"Successfully added {result['count']} vectors with IDs")

    finally:
        # Clean up
        try:
            client.delete_index(index_id)
        except Exception as e:
            logger.error(f"Failed to delete index {index_id}: {str(e)}")
        try:
            client.delete_index(base_index_id)
        except Exception as e:
            logger.error(f"Failed to delete base index {base_index_id}: {str(e)}")


def test_hybrid_search():
    """Test hybrid search functionality"""
    client = get_client()

    # Create index for testing
    index_id = "hybrid_test"

    try:
        # Create index
        client.create_index(index_id, DIMENSION, "L2")

        # Add some vectors
        vectors = generate_test_vectors(100, DIMENSION)
        result = client.add_vectors(index_id, vectors)
        assert result["success"], f"Failed to add vectors: {result.get('error')}"

        # Generate query vectors
        query_vectors = generate_test_vectors(5, DIMENSION)

        logger.info("Testing hybrid search")

        # Call hybrid_search
        result = client._send_request({
            "action": "hybrid_search",
            "index_id": index_id,
            "query_vectors": query_vectors.tolist(),
            "vector_weight": 0.7,
            "metadata_filter": None,  # No metadata filtering for this simple test
            "k": 10
        })

        assert result["success"], f"Hybrid search failed: {result.get('error')}"
        assert "results" in result, "No results in hybrid search response"

        results = result["results"]
        assert len(results) == len(query_vectors), (
            f"Expected results for {len(query_vectors)} queries, got {len(results)}"
        )

        logger.info(f"Hybrid search returned results for {len(results)} queries")

    finally:
        # Clean up
        try:
            client.delete_index(index_id)
        except Exception as e:
            logger.error(f"Failed to delete index {index_id}: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting specialized operations tests")

    tests = [
        test_merge_indices,
        test_optimize_index,
        test_clustering,
        test_batch_add_with_ids,
        test_hybrid_search
    ]

    failures = 0
    for test_func in tests:
        test_name = test_func.__name__
        logger.info(f"Running test: {test_name}")

        try:
            test_func()
            logger.info(f"{test_name}: PASSED")
        except Exception as e:
            logger.exception(f"{test_name}: ERROR - {str(e)}")
            failures += 1

    if failures == 0:
        logger.info("All specialized operations tests PASSED")
    else:
        logger.error(f"{failures} tests FAILED")
        sys.exit(1)

#!/usr/bin/env python3

"""
Test script for checking HNSW index in both local and remote modes.
"""

import logging
import sys

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("HNSW-TEST")


def test_local_mode():
    """Test HNSW index in local mode."""
    logger.info("=== Testing HNSW Index in Local Mode ===")

    from faissx.client.indices.hnsw_flat import IndexHNSWFlat

    # Create a test index
    dimension = 128
    M = 16  # HNSW connections per node
    index = IndexHNSWFlat(dimension, M)

    logger.info(f"Created index: {index.name}")

    # Generate some test vectors
    num_vectors = 100
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)

    # Add vectors
    logger.info(f"Adding {num_vectors} vectors...")
    index.add(vectors)

    logger.info(f"Index contains {index.ntotal} vectors")

    # Perform a search
    num_queries = 5
    k = 10
    query_vectors = np.random.random((num_queries, dimension)).astype(np.float32)

    logger.info(f"Searching for {num_queries} queries, k={k}...")
    distances, indices = index.search(query_vectors, k)

    logger.info(
        f"Search returned results with shape: distances {distances.shape}, "
        f"indices {indices.shape}"
    )

    # Check if indices are within expected range
    max_idx = indices.max()
    min_idx = indices[indices >= 0].min() if (indices >= 0).any() else -1

    logger.info(f"Index range: {min_idx} to {max_idx} (expected: 0 to {num_vectors-1})")
    assert max_idx < num_vectors, "Found index out of range"
    assert min_idx >= 0 or min_idx == -1, "Found negative index (other than -1)"

    # Test range search if available
    try:
        logger.info("Testing range search...")
        radius = 0.5
        lims, range_distances, range_indices = index.range_search(query_vectors[:1], radius)
        logger.info(f"Range search returned {len(range_distances)} results within radius {radius}")
    except Exception as e:
        logger.warning(f"Range search failed: {e}")

    # Reset the index
    logger.info("Resetting index...")
    index.reset()
    logger.info(f"Index contains {index.ntotal} vectors after reset")

    logger.info("Local mode test completed successfully!")
    return True


def test_remote_mode():
    """Test HNSW index in remote mode."""
    logger.info("=== Testing HNSW Index in Remote Mode ===")

    # Configure client first
    from faissx.client import client

    try:
        client.configure(server="tcp://localhost:45678", timeout=5.0)
        logger.info("Connected to server")
    except Exception as e:
        logger.error(f"Could not connect to server: {e}")
        logger.warning("Skipping remote mode test")
        return False

    from faissx.client.indices.hnsw_flat import IndexHNSWFlat

    # Create a test index
    dimension = 128
    M = 16  # HNSW connections per node
    index = IndexHNSWFlat(dimension, M)

    logger.info(f"Created remote index: {index.name} (ID: {index.index_id})")

    # Generate some test vectors
    num_vectors = 100
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)

    # Add vectors
    logger.info(f"Adding {num_vectors} vectors to remote index...")
    index.add(vectors)

    logger.info(f"Remote index contains {index.ntotal} vectors")

    # Perform a search
    num_queries = 5
    k = 10
    query_vectors = np.random.random((num_queries, dimension)).astype(np.float32)

    logger.info(f"Searching remote index for {num_queries} queries, k={k}...")
    distances, indices = index.search(query_vectors, k)

    logger.info(
        f"Search returned results with shape: distances {distances.shape}, "
        f"indices {indices.shape}"
    )

    # Check if indices are within expected range
    max_idx = indices.max()
    min_idx = indices[indices >= 0].min() if (indices >= 0).any() else -1

    logger.info(f"Index range: {min_idx} to {max_idx} (expected: 0 to {num_vectors-1})")

    # Test range search if available
    try:
        logger.info("Testing range search...")
        radius = 0.5
        lims, range_distances, range_indices = index.range_search(query_vectors[:1], radius)
        logger.info(f"Range search returned {len(range_distances)} results within radius {radius}")
    except Exception as e:
        logger.warning(f"Range search failed: {e}")

    # Reset the index
    logger.info("Resetting remote index...")
    index.reset()
    logger.info(f"Remote index contains {index.ntotal} vectors after reset")

    logger.info("Remote mode test completed successfully!")
    return True


def main():
    """Run the tests."""
    results = []

    # Test local mode
    local_success = test_local_mode()
    results.append(("Local Mode", local_success))

    # Test remote mode
    remote_success = test_remote_mode()
    results.append(("Remote Mode", remote_success))

    # Print summary
    logger.info("\n=== Test Results ===")
    for mode, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{mode}: {status}")

    all_passed = all(success for _, success in results)
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

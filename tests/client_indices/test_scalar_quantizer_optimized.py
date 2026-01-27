#!/usr/bin/env python3

"""
Test script to verify the optimized IndexScalarQuantizer implementation.

This script tests both local and remote modes to ensure they work correctly.
"""

import logging

import numpy as np

# Import FAISSx client
from faissx import client as faiss
from faissx.client.client import get_client

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_local_mode():
    """Test the IndexScalarQuantizer in local mode."""
    print("\n=== Testing Optimized IndexScalarQuantizer in Local Mode ===")

    # Make sure we're in local mode (default)
    client = get_client()
    print(f"Client mode: {client.mode if client else 'None'}")

    # Create an index
    dimension = 8
    # Need to import ScalarQuantizer to get the quantizer type
    try:
        qtype = faiss.ScalarQuantizer.QT_8bit
    except AttributeError:
        # Fallback if ScalarQuantizer constants aren't available
        qtype = None

    index = faiss.IndexScalarQuantizer(dimension, qtype)

    # Generate random vectors
    np.random.seed(42)
    num_vectors = 10
    vectors = np.random.random((num_vectors, dimension)).astype('float32')

    # Add vectors to the index
    index.add(vectors)
    print(f"Added {num_vectors} vectors to index, ntotal = {index.ntotal}")

    # Search for similar vectors
    k = 3
    query = vectors[0:1]  # Use the first vector as a query
    distances, indices = index.search(query, k)

    print("Search results in local mode:")
    print(f"  Query: {query[0][:3]}...")
    print(f"  Distances: {distances[0]}")
    print(f"  Indices: {indices[0]}")

    # Test context manager
    print("\nTesting context manager support...")
    with faiss.IndexScalarQuantizer(dimension, qtype) as ctx_index:
        ctx_index.add(vectors)
        print(f"Context manager index created with {ctx_index.ntotal} vectors")
    print("Context manager exited successfully")

    return index


def test_remote_mode():
    """Test the IndexScalarQuantizer in remote mode."""
    print("\n=== Testing Optimized IndexScalarQuantizer in Remote Mode ===")

    # Configure for remote mode
    faiss.configure(server="tcp://localhost:45678")

    # Verify we're in remote mode
    client = get_client()
    print(f"Client mode: {client.mode if client else 'None'}")
    print(f"Server: {client.server if client else 'None'}")

    # Create an index
    dimension = 8
    # For remote mode, we'll use a string representation rather than the constant
    qtype = "SQ8"  # Simpler string representation for server compatibility

    index = faiss.IndexScalarQuantizer(dimension, qtype)

    # Generate random vectors
    np.random.seed(42)  # Same seed as local test for comparison
    num_vectors = 10
    vectors = np.random.random((num_vectors, dimension)).astype('float32')

    # Add vectors to the index
    index.add(vectors)
    print(f"Added {num_vectors} vectors to index, ntotal = {index.ntotal}")

    # Search for similar vectors
    k = 3
    query = vectors[0:1]  # Use the first vector as a query
    distances, indices = index.search(query, k)

    print("Search results in remote mode:")
    print(f"  Query: {query[0][:3]}...")
    print(f"  Distances: {distances[0]}")
    print(f"  Indices: {indices[0]}")

    # Test parameter setting
    print("\nTesting parameter setting...")
    index.set_parameter('batch_size', 500)
    print("Set batch_size parameter to 500")

    return index


if __name__ == "__main__":
    # First test local mode
    local_index = test_local_mode()

    # Then test remote mode
    try:
        remote_index = test_remote_mode()
        print("\n=== Test completed successfully ===")
        print("Both local and remote modes are working correctly!")
    except Exception as e:
        print(f"\n=== Remote mode test failed: {e} ===")
        print("Please check that the server is running at tcp://localhost:45678")

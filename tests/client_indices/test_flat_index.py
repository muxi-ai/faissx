#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify that IndexFlatL2 works correctly.

This script tests IndexFlatL2 in both local and remote modes.
"""

import numpy as np
import logging
from time import sleep

# Import FAISSx client
from faissx import client as faiss
from faissx.client.client import get_client

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_local_mode():
    """Test IndexFlatL2 in local mode."""
    print("\n=== Testing IndexFlatL2 in Local Mode ===")

    # Ensure we're in local mode
    client = get_client()
    print(f"Client mode: {client.mode if client else 'None'}")

    # Create an index
    dimension = 8
    index = faiss.IndexFlatL2(dimension)

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
    print(f"  Query shape: {query.shape}")
    print(f"  Distances: {distances[0]}")
    print(f"  Indices: {indices[0]}")

    # The first result should be the query vector itself with distance near 0
    success = (indices[0][0] == 0 and distances[0][0] < 0.001)
    print(f"Local mode test {'PASSED' if success else 'FAILED'}")
    return success


def test_remote_mode():
    """Test IndexFlatL2 in remote mode."""
    print("\n=== Testing IndexFlatL2 in Remote Mode ===")

    # Configure for remote mode
    faiss.configure(server="tcp://localhost:45678")
    sleep(1)  # Give a moment for connection to establish

    # Verify we're in remote mode
    client = get_client()
    print(f"Client mode: {client.mode if client else 'None'}")
    if client:
        print(f"Server: {client.server}")

    # Create an index
    dimension = 8
    index = faiss.IndexFlatL2(dimension)

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
    print(f"  Query shape: {query.shape}")
    print(f"  Distances: {distances[0]}")
    print(f"  Indices: {indices[0]}")

    # The first result should be the query vector itself with distance near 0
    success = (indices[0][0] == 0 and distances[0][0] < 0.001)
    print(f"Remote mode test {'PASSED' if success else 'FAILED'}")
    return success


if __name__ == "__main__":
    print("\n============================================================")
    print("TESTING IndexFlatL2")
    print("============================================================")

    # First test local mode
    local_success = test_local_mode()

    # Then test remote mode
    try:
        remote_success = test_remote_mode()
        print("\nOverall test result:")
        if local_success and remote_success:
            print("✅ SUCCESS: Both local and remote modes are working correctly!")
        else:
            print("❌ FAILURE: One or both modes failed.")
            if not local_success:
                print("   - Local mode failed")
            if not remote_success:
                print("   - Remote mode failed")
    except Exception as e:
        print(f"\n❌ FAILURE: Remote mode test failed with error: {e}")
        print("Please check that the server is running at tcp://localhost:45678")

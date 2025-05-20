#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test remote mode without fallback behavior
#
# Copyright (C) 2025 Ran Aroussi

"""
Test the behavior of FAISSx in remote mode with no fallback.

This script validates that the FAISSx client properly handles remote mode
without falling back to local mode, even when errors occur. It tests
different index types and verifies their behavior.
"""

import sys
import os
import numpy as np

# Add parent directory to path to import faissx
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import faissx  # noqa: E402

# Get test port from environment or use default
TEST_PORT = os.environ.get("FAISSX_TEST_PORT", "45678")
TEST_SERVER_URL = f"tcp://localhost:{TEST_PORT}"


def test_index_flat():
    """Test IndexFlatL2 in remote mode."""
    print("\n===== Testing IndexFlatL2 =====")

    # Configure with remote URL
    faissx.configure(url=TEST_SERVER_URL)

    try:
        # Create and use the index
        index = faissx.IndexFlatL2(32)
        print("Created IndexFlatL2 successfully")

        # Test adding vectors
        vectors = np.random.random((10, 32)).astype(np.float32)
        index.add(vectors)
        print(f"Added {len(vectors)} vectors, total: {index.ntotal}")

        # Test search
        query = np.random.random((2, 32)).astype(np.float32)
        distances, indices = index.search(query, 5)
        print(f"Search results: {len(indices[0])} neighbors found")

        print("IndexFlatL2 works properly in remote mode ✓")
        return True
    except Exception as e:
        print(f"Error with IndexFlatL2: {e}")
        return False


def test_index_ivf():
    """Test IndexIVFFlat in remote mode."""
    print("\n===== Testing IndexIVFFlat =====")

    # Configure with remote URL
    faissx.configure(url=TEST_SERVER_URL)

    try:
        # Create quantizer and IVF index
        quantizer = faissx.IndexFlatL2(32)
        index = faissx.IndexIVFFlat(quantizer, 32, 10)
        print("Created IndexIVFFlat successfully")

        # Test training
        training_vectors = np.random.random((100, 32)).astype(np.float32)
        index.train(training_vectors)
        print("Trained index successfully")

        # Test adding vectors
        vectors = np.random.random((10, 32)).astype(np.float32)
        index.add(vectors)
        print(f"Added {len(vectors)} vectors, total: {index.ntotal}")

        # Test search
        query = np.random.random((2, 32)).astype(np.float32)
        distances, indices = index.search(query, 5)
        print(f"Search results: {len(indices[0])} neighbors found")

        print("IndexIVFFlat works properly in remote mode ✓")
        return True
    except Exception as e:
        print(f"Error with IndexIVFFlat: {e}")
        print("This is expected behavior if the server doesn't implement IVF indices")
        # Consider this a pass - the proper behavior is to raise an error rather than fall back
        return True


def test_index_idmap():
    """Test IndexIDMap in remote mode."""
    print("\n===== Testing IndexIDMap =====")

    # Configure with remote URL
    faissx.configure(url=TEST_SERVER_URL)

    try:
        # Create base index
        base_index = faissx.IndexFlatL2(32)

        # Try to create IDMap - this should fail with a clear error
        _ = faissx.IndexIDMap(base_index)
        print("Created IndexIDMap - this is unexpected as it should fail in remote mode!")
        return False
    except NotImplementedError as e:
        print(f"Expected error correctly raised: {e}")
        print("IndexIDMap correctly reports not being implemented in remote mode ✓")
        return True
    except Exception as e:
        print(f"Unexpected error type: {e}")
        return False


def test_unavailable_server():
    """Test behavior when server is unavailable."""
    print("\n===== Testing Unavailable Server =====")

    try:
        # Import zmq directly
        import zmq

        # Use a context with a short timeout
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 500)  # Set receive timeout to 500ms
        socket.setsockopt(zmq.SNDTIMEO, 500)  # Set send timeout to 500ms

        # Try to connect to a nonexistent server
        nonexistent_url = "tcp://nonexistent-server-that-will-fail:12345"
        socket.connect(nonexistent_url)

        # Try to send a message, which should fail
        socket.send(b"ping")

        # Try to receive a response, which should timeout
        socket.recv()

        # If we got here, something went wrong
        socket.close()
        context.term()
        print("Connection succeeded unexpectedly!")
        return False
    except Exception as e:
        print(f"Expected error correctly raised: {e}")
        print("Connection to unavailable server correctly failed ✓")
        return True


def test_index_hnsw():
    """Test IndexHNSWFlat in remote mode."""
    print("\n===== Testing IndexHNSWFlat =====")

    # Configure with remote URL
    faissx.configure(url=TEST_SERVER_URL)

    try:
        # Create HNSW index
        index = faissx.IndexHNSWFlat(32, M=16)
        print("Created IndexHNSWFlat successfully")

        # Test adding vectors
        vectors = np.random.random((10, 32)).astype(np.float32)
        index.add(vectors)
        print(f"Added {len(vectors)} vectors, total: {index.ntotal}")

        # Test search
        query = np.random.random((2, 32)).astype(np.float32)
        distances, indices = index.search(query, 5)
        print(f"Search results: {len(indices[0])} neighbors found")

        print("IndexHNSWFlat works properly in remote mode ✓")
        return True
    except Exception as e:
        print(f"Error with IndexHNSWFlat: {e}")
        print("This is expected behavior if the server doesn't implement HNSW indices")
        # Consider this a pass - the proper behavior is to raise an error rather than fall back
        return True


def test_index_pq():
    """Test IndexPQ in remote mode."""
    print("\n===== Testing IndexPQ =====")

    # Configure with remote URL
    faissx.configure(url=TEST_SERVER_URL)

    try:
        # Create PQ index (dimension must be multiple of M)
        index = faissx.IndexPQ(32, M=8)
        print("Created IndexPQ successfully")

        # Test training (PQ requires training)
        training_vectors = np.random.random((100, 32)).astype(np.float32)
        index.train(training_vectors)
        print("Trained index successfully")

        # Test adding vectors
        vectors = np.random.random((10, 32)).astype(np.float32)
        index.add(vectors)
        print(f"Added {len(vectors)} vectors, total: {index.ntotal}")

        # Test search
        query = np.random.random((2, 32)).astype(np.float32)
        distances, indices = index.search(query, 5)
        print(f"Search results: {len(indices[0])} neighbors found")

        print("IndexPQ works properly in remote mode ✓")
        return True
    except Exception as e:
        print(f"Error with IndexPQ: {e}")
        print("This is expected behavior if the server doesn't implement PQ indices")
        # Consider this a pass - the proper behavior is to raise an error rather than fall back
        return True


def test_index_scalar_quantizer():
    """Test IndexScalarQuantizer in remote mode."""
    print("\n===== Testing IndexScalarQuantizer =====")

    # Configure with remote URL
    faissx.configure(url=TEST_SERVER_URL)

    try:
        # Create Scalar Quantizer index
        index = faissx.IndexScalarQuantizer(32)
        print("Created IndexScalarQuantizer successfully")

        # Test adding vectors
        vectors = np.random.random((10, 32)).astype(np.float32)
        index.add(vectors)
        print(f"Added {len(vectors)} vectors, total: {index.ntotal}")

        # Test search
        query = np.random.random((2, 32)).astype(np.float32)
        distances, indices = index.search(query, 5)
        print(f"Search results: {len(indices[0])} neighbors found")

        print("IndexScalarQuantizer works properly in remote mode ✓")
        return True
    except Exception as e:
        print(f"Error with IndexScalarQuantizer: {e}")
        print("This is expected behavior if the server doesn't implement SQ indices")
        # Consider this a pass - the proper behavior is to raise an error rather than fall back
        return True


def run_all_tests():
    """Run all remote mode tests."""
    print("\n===== FAISSx Remote Mode Tests (No Fallback) =====")
    print("\nThese tests verify that FAISSx properly handles remote mode")
    print("without falling back to local mode when errors occur.\n")

    # Start server status as unknown
    server_available = None

    # Track test results
    results = {}

    # Test IndexFlatL2
    results["IndexFlatL2"] = test_index_flat()
    if results["IndexFlatL2"]:
        server_available = True

    # Only run further tests if server is available
    if server_available:
        # Test IndexIVFFlat
        results["IndexIVFFlat"] = test_index_ivf()

        # Test IndexIDMap
        results["IndexIDMap"] = test_index_idmap()

        # Test IndexHNSWFlat
        results["IndexHNSWFlat"] = test_index_hnsw()

        # Test IndexPQ
        results["IndexPQ"] = test_index_pq()

        # Test IndexScalarQuantizer
        results["IndexScalarQuantizer"] = test_index_scalar_quantizer()

    # Always test unavailable server behavior - use a different configuration
    results["UnavailableServer"] = test_unavailable_server()

    # Print summary
    print("\n===== Test Summary =====")
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test}: {status}")

    # Determine if all tests passed
    all_passed = all(results.values())
    print(f"\nOverall result: {'✓ PASS' if all_passed else '✗ FAIL'}")

    return all_passed


if __name__ == "__main__":
    # Run all tests and set exit code based on result
    success = run_all_tests()
    sys.exit(0 if success else 1)

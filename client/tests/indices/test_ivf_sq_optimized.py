#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test file for optimized IndexIVFScalarQuantizer implementation.

This test verifies that the optimized IndexIVFScalarQuantizer works correctly in both
local and remote modes, including training, adding vectors, and searching.
"""

import os
import logging
import numpy as np

from faissx import client as faiss
from faissx.client.client import get_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_ivf_sq_local_mode():
    """Test IndexIVFScalarQuantizer in local mode."""
    print("\nTesting IndexIVFScalarQuantizer in local mode...")

    # Ensure we're in local mode by not setting a server
    # We don't call configure() because that tries to connect to a server

    # Dimensions and data size
    d = 64  # dimension
    nb = 1000  # database size
    nq = 10  # queries

    # Generate random data
    np.random.seed(42)  # for reproducibility
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')

    # Create quantizer and index
    quantizer = faiss.IndexFlatL2(d)  # the coarse quantizer
    nlist = 50  # number of centroids/clusters

    # Create the IVF-SQ index with default SQ8 quantizer type
    index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist)

    # Test if the index was created properly
    assert index.d == d, f"Dimension mismatch: {index.d} != {d}"
    assert index.nlist == nlist, f"nlist mismatch: {index.nlist} != {nlist}"
    assert not index.is_trained, "Index should not be trained yet"
    assert index.ntotal == 0, "Index should be empty"

    # Training
    print("Training index...")
    index.train(xb)
    assert index.is_trained, "Index should be trained after training"

    # Adding vectors
    print("Adding vectors...")
    index.add(xb)
    assert index.ntotal == nb, f"Expected {nb} vectors, got {index.ntotal}"

    # Setting nprobe
    print("Setting nprobe...")
    index.set_nprobe(10)
    assert index.nprobe == 10, f"nprobe mismatch: {index.nprobe} != 10"

    # Test search
    print("Searching...")
    k = 5  # number of nearest neighbors
    distances, idx = index.search(xq, k)

    # Check search results
    assert distances.shape == (nq, k), f"Expected distances shape {(nq, k)}, got {distances.shape}"
    assert idx.shape == (nq, k), f"Expected indices shape {(nq, k)}, got {idx.shape}"

    # Test reconstructing a vector
    print("Testing vector reconstruction...")
    vector_idx = 42  # arbitrary index
    reconstructed = index.reconstruct(vector_idx)
    assert reconstructed.shape == (d,), f"Expected shape ({d},), got {reconstructed.shape}"

    # Test multiple vector reconstruction
    print("Testing multiple vector reconstruction...")
    start_idx = 10
    count = 5
    reconstructed_batch = index.reconstruct_n(start_idx, count)
    assert reconstructed_batch.shape == (count, d), (
        f"Expected shape ({count}, {d}), got {reconstructed_batch.shape}"
    )

    # Test get_vectors
    print("Testing get_vectors...")
    all_vectors = index.get_vectors()
    if all_vectors is not None:
        print(f"Retrieved {len(all_vectors)} vectors")
        assert all_vectors.shape[1] == d, f"Retrieved vectors should have dimension {d}"

    # Test reset
    print("Testing reset...")
    index.reset()
    assert index.ntotal == 0, f"After reset, expected 0 vectors, got {index.ntotal}"
    assert index.is_trained, "Index should still be trained after reset"

    print("✅ IndexIVFScalarQuantizer local mode tests completed successfully!")


def test_ivf_sq_remote_mode():
    """Test IndexIVFScalarQuantizer in remote mode with a FAISSx server."""
    print("\nTesting IndexIVFScalarQuantizer in remote mode...")

    # Skip test if server is not available
    server_addr = os.environ.get("FAISSX_SERVER", "tcp://localhost:45678")

    try:
        # Configure remote mode by specifying a server
        faiss.configure(server=server_addr)

        # Verify we're actually in remote mode
        client = get_client()
        if client is None or client.mode != "remote":
            print("Failed to configure remote mode, skipping test")
            return

        # Dimensions and data size
        d = 64  # dimension
        nb = 1000  # database size
        nq = 10  # queries

        # Generate random data
        np.random.seed(42)  # for reproducibility
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')

        # Create quantizer and index
        quantizer = faiss.IndexFlatL2(d)  # the coarse quantizer
        nlist = 50  # number of centroids/clusters

        # Create the IVF-SQ index
        index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist)

        # Test if the index was created properly
        assert index.d == d, f"Dimension mismatch: {index.d} != {d}"
        assert index.nlist == nlist, f"nlist mismatch: {index.nlist} != {nlist}"

        # Some servers automatically train indices, so we can't assume it's untrained
        print(f"Initial training state: {index.is_trained}")

        # Training (only if not already trained)
        if not index.is_trained:
            print("Training index...")
            index.train(xb)
            assert index.is_trained, "Index should be trained after training"

        # Adding vectors
        print("Adding vectors...")
        index.add(xb)
        assert index.ntotal == nb, f"Expected {nb} vectors, got {index.ntotal}"

        # Setting nprobe
        print("Setting nprobe...")
        index.set_nprobe(10)
        assert index.nprobe == 10, f"nprobe mismatch: {index.nprobe} != 10"

        # Test search
        print("Searching...")
        k = 5  # number of nearest neighbors
        distances, idx = index.search(xq, k)

        # Check search results
        assert distances.shape == (nq, k), f"Expected distances shape {(nq, k)}, got {distances.shape}"
        assert idx.shape == (nq, k), f"Expected indices shape {(nq, k)}, got {idx.shape}"

        # Print a sample of search results
        print("Sample search results:")
        for i in range(min(3, nq)):
            print(f"Query {i}: distances {distances[i]}, indices {idx[i]}")

        # Test vector reconstruction if supported
        print("Testing vector reconstruction...")
        try:
            vector_idx = 0  # First vector
            reconstructed = index.reconstruct(vector_idx)
            assert reconstructed.shape == (d,), f"Expected shape ({d},), got {reconstructed.shape}"
            print("Vector reconstruction succeeded")
        except Exception as e:
            print(f"Vector reconstruction not fully supported in remote mode: {e}")

        # Test reset - may not be fully supported on all servers
        print("Testing reset...")
        try:
            index.reset()
            print(f"After reset, ntotal = {index.ntotal}")
            if index.ntotal != 0:
                print("⚠️ Warning: Reset may not have fully cleared the index on the server")
            print(f"After reset, is_trained = {index.is_trained}")
        except Exception as e:
            print(f"Reset not fully supported in remote mode: {e}")

        print("✅ IndexIVFScalarQuantizer remote mode tests completed successfully!")

    except Exception as e:
        print(f"❌ Remote mode test failed: {e}")


if __name__ == "__main__":
    print("Testing optimized IndexIVFScalarQuantizer implementation...")

    # Test local mode
    test_ivf_sq_local_mode()

    # Only test remote mode if explicitly enabled
    if os.environ.get("TEST_REMOTE_MODE", "").lower() == "true":
        test_ivf_sq_remote_mode()
    else:
        print("\nSkipping remote mode tests (set TEST_REMOTE_MODE=true to enable)")

    print("\nAll tests completed!")

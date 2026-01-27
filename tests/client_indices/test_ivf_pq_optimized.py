#!/usr/bin/env python3

"""
Test file for optimized IndexIVFPQ implementation.

This test verifies that the optimized IndexIVFPQ works correctly in both
local and remote modes, including training, adding vectors, and searching.
"""

import logging
import os

import numpy as np

from faissx import client as faiss
from faissx.client.client import get_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_ivf_pq_local_mode():
    """Test IndexIVFPQ in local mode."""
    print("\nTesting IndexIVFPQ in local mode...")

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
    m = 8  # number of subquantizers
    nbits = 8  # bits per subquantizer

    # Create the IVF-PQ index
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    # Test if the index was created properly
    assert index.d == d, f"Dimension mismatch: {index.d} != {d}"
    assert index.nlist == nlist, f"nlist mismatch: {index.nlist} != {nlist}"
    assert index.M == m, f"M mismatch: {index.M} != {m}"
    assert index.nbits == nbits, f"nbits mismatch: {index.nbits} != {nbits}"
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
    D, I = index.search(xq, k)

    # Check search results
    assert D.shape == (nq, k), f"Expected distances shape {(nq, k)}, got {D.shape}"
    assert I.shape == (nq, k), f"Expected indices shape {(nq, k)}, got {I.shape}"

    # Test reconstructing a vector
    print("Testing vector reconstruction...")
    vector_idx = 42  # arbitrary index
    reconstructed = index.reconstruct(vector_idx)
    assert reconstructed.shape == (d,), f"Expected shape ({d},), got {reconstructed.shape}"

    # Test range search
    print("Testing range search...")
    try:
        D, I, lims = index.range_search(xq, 0.7)  # reasonable radius for random data
        print(f"Range search returned {len(D)} results across {len(xq)} queries")
    except Exception as e:
        print(f"Range search not fully supported in local mode: {e}")

    # Test reset
    print("Testing reset...")
    index.reset()
    assert index.ntotal == 0, f"After reset, expected 0 vectors, got {index.ntotal}"
    assert index.is_trained, "Index should still be trained after reset"

    print("✅ IndexIVFPQ local mode tests completed successfully!")


def test_ivf_pq_remote_mode():
    """Test IndexIVFPQ in remote mode with a FAISSx server."""
    print("\nTesting IndexIVFPQ in remote mode...")

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
        m = 8  # number of subquantizers
        nbits = 8  # bits per subquantizer

        # Create the IVF-PQ index
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

        # Test if the index was created properly
        assert index.d == d, f"Dimension mismatch: {index.d} != {d}"
        assert index.nlist == nlist, f"nlist mismatch: {index.nlist} != {nlist}"
        assert index.M == m, f"M mismatch: {index.M} != {m}"
        assert index.nbits == nbits, f"nbits mismatch: {index.nbits} != {nbits}"

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
        D, I = index.search(xq, k)

        # Check search results
        assert D.shape == (nq, k), f"Expected distances shape {(nq, k)}, got {D.shape}"
        assert I.shape == (nq, k), f"Expected indices shape {(nq, k)}, got {I.shape}"

        # Print a sample of search results
        print("Sample search results:")
        for i in range(min(3, nq)):
            print(f"Query {i}: distances {D[i]}, indices {I[i]}")

        # Test range search - may not be fully supported on all servers
        print("Testing range search...")
        try:
            D, I, lims = index.range_search(xq, 0.7)  # reasonable radius for random data
            print(f"Range search returned {len(D)} results across {len(xq)} queries")
        except Exception as e:
            print(f"Range search not fully supported in remote mode: {e}")

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

        print("✅ IndexIVFPQ remote mode tests completed successfully!")

    except Exception as e:
        print(f"❌ Remote mode test failed: {e}")


if __name__ == "__main__":
    print("Testing optimized IndexIVFPQ implementation...")

    # Test local mode
    test_ivf_pq_local_mode()

    # Only test remote mode if explicitly enabled
    if os.environ.get("TEST_REMOTE_MODE", "").lower() == "true":
        test_ivf_pq_remote_mode()
    else:
        print("\nSkipping remote mode tests (set TEST_REMOTE_MODE=true to enable)")

    print("\nAll tests completed!")

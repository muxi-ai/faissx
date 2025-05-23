#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test for Optimized FAISSx Implementations

This test verifies the optimized implementations of:
1. IndexPQ
2. IndexIVFScalarQuantizer
3. modification module (merge_indices, split_index)

It tests both local and remote modes.
"""

import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Try to import FAISSx
try:
    from faissx.client.client import get_client, FaissXClient
    from faissx.client.indices.pq import IndexPQ
    from faissx.client.indices.ivf_scalar_quantizer import IndexIVFScalarQuantizer
    from faissx.client.indices.flat import IndexFlatL2
    from faissx.client.indices.modification import merge_indices, split_index
    import faissx
except ImportError as e:
    logger.error(f"Failed to import FAISSx: {e}")
    sys.exit(1)

# Set to True to test remote mode
TEST_REMOTE = True
# Dimension for all tests
DIM = 8
# Number of vectors for basic operations
NUM_VECTORS = 100
# Batch size for tests
BATCH_SIZE = 10

def test_mode_info():
    """Print information about the current mode (local or remote)"""
    client = get_client()
    logger.info(f"FAISSx Version: {faissx.__version__}")

    if client is not None:
        logger.info(f"Mode: Remote (server: {client.server})")
    else:
        logger.info("Mode: Local")

def setup_remote_client():
    """Set up remote client if testing in remote mode"""
    if TEST_REMOTE:
        try:
            # Create client properly
            client = FaissXClient()
            client.configure(
                server="tcp://localhost:45678",
                api_key="",
                tenant_id="test-tenant"
            )
            # Set client as global
            faissx.client.client._client = client
            logger.info(f"Remote client created: {client}")
            return client
        except Exception as e:
            logger.error(f"Failed to create remote client: {e}")
            sys.exit(1)
    return None

def test_pq_implementation():
    """Test the optimized IndexPQ implementation"""
    logger.info("\n=== Testing IndexPQ Implementation ===")

    # Create PQ index
    m = 4  # Number of subquantizers (must be a divisor of dimension)
    nbits = 8  # Bits per subquantizer

    try:
        logger.info(f"Creating IndexPQ({DIM}, {m}, {nbits})...")
        index = IndexPQ(DIM, m, nbits)
        logger.info(f"Index created: {type(index)}")

        # Create random training vectors
        logger.info("Training index...")
        train_vectors = np.random.random((1000, DIM)).astype('float32')
        index.train(train_vectors)
        logger.info(f"Training complete, is_trained: {index.is_trained}")

        # Add vectors
        logger.info(f"Adding {NUM_VECTORS} vectors...")
        vectors = np.random.random((NUM_VECTORS, DIM)).astype('float32')
        index.add(vectors)
        logger.info(f"Added vectors, ntotal: {index.ntotal}")

        # Verify vector caching
        if hasattr(index, "_cached_vectors") and index._cached_vectors is not None:
            logger.info("Vector caching working correctly")

        # Test reconstruction
        logger.info("Testing reconstruction...")
        idx = NUM_VECTORS // 2
        reconstructed = index.reconstruct(idx)
        logger.info(f"Reconstructed vector {idx}, shape: {reconstructed.shape}")

        # Test batch reconstruction
        logger.info("Testing batch reconstruction...")
        batch = index.reconstruct_n(0, min(BATCH_SIZE, NUM_VECTORS))
        logger.info(f"Reconstructed {BATCH_SIZE} vectors, shape: {batch.shape}")

        # Test search
        logger.info("Testing search...")
        query = np.random.random((1, DIM)).astype('float32')
        distances, indices = index.search(query, k=5)
        logger.info(f"Search results: top 5 indices: {indices[0]}")

        # Test get_vectors
        logger.info("Testing get_vectors...")
        all_vectors = index.get_vectors()
        if all_vectors is not None:
            logger.info(f"get_vectors returned array of shape: {all_vectors.shape}")

        # Test reset
        logger.info("Testing reset...")
        index.reset()
        logger.info(f"After reset, ntotal: {index.ntotal}")

        return True
    except Exception as e:
        logger.error(f"IndexPQ test failed: {e}")
        return False

def test_ivf_sq_implementation():
    """Test the optimized IndexIVFScalarQuantizer implementation"""
    logger.info("\n=== Testing IndexIVFScalarQuantizer Implementation ===")

    try:
        # Create quantizer
        logger.info(f"Creating quantizer IndexFlatL2({DIM})...")
        quantizer = IndexFlatL2(DIM)

        # Create IVF-SQ index
        nlist = 4  # Number of inverted lists/centroids
        logger.info(f"Creating IndexIVFScalarQuantizer with nlist={nlist}...")
        index = IndexIVFScalarQuantizer(quantizer, DIM, nlist)
        logger.info(f"Index created: {type(index)}")

        # Create random training vectors
        logger.info("Training index...")
        train_vectors = np.random.random((1000, DIM)).astype('float32')
        index.train(train_vectors)
        logger.info(f"Training complete, is_trained: {index.is_trained}")

        # Add vectors
        logger.info(f"Adding {NUM_VECTORS} vectors...")
        vectors = np.random.random((NUM_VECTORS, DIM)).astype('float32')
        index.add(vectors)
        logger.info(f"Added vectors, ntotal: {index.ntotal}")

        # Verify vector caching
        if hasattr(index, "_cached_vectors") and index._cached_vectors is not None:
            logger.info("Vector caching working correctly")

        # Test reconstruction
        logger.info("Testing reconstruction...")
        idx = NUM_VECTORS // 2
        reconstructed = index.reconstruct(idx)
        logger.info(f"Reconstructed vector {idx}, shape: {reconstructed.shape}")

        # Test batch reconstruction
        logger.info("Testing batch reconstruction...")
        batch = index.reconstruct_n(0, min(BATCH_SIZE, NUM_VECTORS))
        logger.info(f"Reconstructed {BATCH_SIZE} vectors, shape: {batch.shape}")

        # Test search
        logger.info("Testing search...")
        query = np.random.random((1, DIM)).astype('float32')
        distances, indices = index.search(query, k=5)
        logger.info(f"Search results: top 5 indices: {indices[0]}")

        # Test get_vectors
        logger.info("Testing get_vectors...")
        all_vectors = index.get_vectors()
        if all_vectors is not None:
            logger.info(f"get_vectors returned array of shape: {all_vectors.shape}")

        # Test nprobe setter
        logger.info("Testing nprobe setter...")
        index.nprobe = 2
        logger.info(f"nprobe set to: {index.nprobe}")

        # Test reset
        logger.info("Testing reset...")
        index.reset()
        logger.info(f"After reset, ntotal: {index.ntotal}")

        return True
    except Exception as e:
        logger.error(f"IndexIVFScalarQuantizer test failed: {e}")
        return False

def test_modification_module():
    """Test the optimized modification module (merge_indices, split_index)"""
    logger.info("\n=== Testing Modification Module ===")

    try:
        # Create indices to merge
        logger.info("Creating indices for merging test...")
        index1 = IndexFlatL2(DIM)
        index2 = IndexFlatL2(DIM)

        # Add different vectors to each index
        vecs1 = np.random.random((NUM_VECTORS // 2, DIM)).astype('float32')
        vecs2 = np.random.random((NUM_VECTORS // 2, DIM)).astype('float32')

        index1.add(vecs1)
        index2.add(vecs2)

        logger.info(f"Created two indices with {index1.ntotal} and {index2.ntotal} vectors")

        # Test merge_indices
        logger.info("Testing merge_indices...")
        merged = merge_indices([index1, index2], batch_size=BATCH_SIZE)
        logger.info(f"Merged index has {merged.ntotal} vectors")

        # Test search on merged index
        query = np.random.random((1, DIM)).astype('float32')
        distances, indices = merged.search(query, k=5)
        logger.info(f"Search on merged index results: {indices[0]}")

        # Test split_index
        logger.info("Testing split_index...")
        num_parts = 3
        split_results = split_index(merged, num_parts=num_parts, batch_size=BATCH_SIZE)

        logger.info(f"Split into {len(split_results)} parts")
        for i, part in enumerate(split_results):
            logger.info(f"Part {i} has {part.ntotal} vectors")

            # Test search on each part
            d, idx = part.search(query, k=2)
            logger.info(f"Search on part {i} results: {idx[0]}")

        return True
    except Exception as e:
        logger.error(f"Modification module test failed: {e}")
        return False

def run_all_tests():
    """Run all optimization tests"""
    logger.info("Starting optimization tests...")

    # Setup client if testing remote mode
    setup_remote_client()
    test_mode_info()

    # Run tests
    results = {
        "IndexPQ": test_pq_implementation(),
        "IndexIVFScalarQuantizer": test_ivf_sq_implementation(),
        "Modification Module": test_modification_module()
    }

    # Print summary
    logger.info("\n=== Test Results Summary ===")
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\nAll optimization tests PASSED!")
    else:
        logger.error("\nSome optimization tests FAILED!")

if __name__ == "__main__":
    run_all_tests()

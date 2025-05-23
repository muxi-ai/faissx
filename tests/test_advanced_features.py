#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Client Advanced Features Test
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
Tests for advanced FAISS features like reconstruction and index merging
"""

import os
import numpy as np
import logging
import uuid
from faissx.client.client import FaissXClient


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Set up dimensions and test data constants
DIM = 8
NUM_TRAIN = 1000
NUM_TEST = 100
K = 5  # Number of nearest neighbors to search for


def generate_test_data(dim=DIM, num=100, seed=42):
    """Generate random test data"""
    np.random.seed(seed)
    return np.random.random((num, dim)).astype(np.float32)


def test_reconstruction(client):
    """Test vector reconstruction functionality"""
    logger.info("\n=== Testing Vector Reconstruction ===")

    # Generate random data
    test_data = generate_test_data(dim=DIM, num=NUM_TEST, seed=123)

    # Create index name with a unique identifier - use Flat index which supports reconstruction
    index_name = f"flat-test-{uuid.uuid4().hex[:8]}"

    # Create the index
    logger.info(f"Creating Flat index: {index_name}")
    client.create_index(index_name, DIM, 'L2')

    # Add vectors
    logger.info(f"Adding {NUM_TEST} vectors to index")
    result = client.add_vectors(index_name, test_data)
    ntotal = result.get('total', 0)
    logger.info(f"Added {ntotal} vectors to index")

    # Test vector reconstruction - try reconstructing vector at index 50
    logger.info("Testing vector reconstruction...")
    try:
        # Single vector reconstruction
        reconstruct_result = client.reconstruct(index_name, 50)
        reconstructed_vector = reconstruct_result.get('vector', [])

        if len(reconstructed_vector) > 0:
            # Compare with original vector
            original_vector = test_data[50]
            is_similar = np.allclose(original_vector, reconstructed_vector, rtol=1e-5)
            logger.info(f"Reconstruction successful, vector matches original: {is_similar}")

            # Test multiple vector reconstruction
            logger.info("Testing multiple vector reconstruction...")
            reconstruct_n_result = client.reconstruct_n(index_name, 20, 5)
            reconstructed_vectors = reconstruct_n_result.get('vectors', [])
            logger.info(f"Reconstructed {len(reconstructed_vectors)} vectors")

            # Success if both reconstructions worked
            success = is_similar and len(reconstructed_vectors) == 5
        else:
            logger.error("Failed to reconstruct vector - empty result")
            success = False
    except Exception as e:
        logger.error(f"Reconstruction test failed: {e}")
        success = False

    logger.info(f"Reconstruction test {'PASSED' if success else 'FAILED'}")
    return success


def test_search_and_reconstruct(client):
    """Test search with reconstruction"""
    logger.info("\n=== Testing Search and Reconstruct ===")

    # Generate random data
    test_data = generate_test_data(dim=DIM, num=NUM_TEST, seed=123)
    xq = generate_test_data(dim=DIM, num=5, seed=456)

    # Create index name with a unique identifier
    index_name = f"flat-search-{uuid.uuid4().hex[:8]}"

    # Create the index
    logger.info(f"Creating index: {index_name}")
    client.create_index(index_name, DIM, 'L2')

    # Add vectors
    logger.info(f"Adding {NUM_TEST} vectors to index")
    result = client.add_vectors(index_name, test_data)
    ntotal = result.get('total', 0)
    logger.info(f"Added {ntotal} vectors to index")

    # Test search and reconstruct
    logger.info(f"Testing search and reconstruct with {len(xq)} queries")
    try:
        search_result = client.search_and_reconstruct(index_name, xq, K)
        results = search_result.get('results', [])

        # Check if the response includes vectors along with indices and distances
        success = len(results) == len(xq)

        if success and len(results) > 0:
            # Check if vectors were included in the results
            first_result = results[0]
            has_vectors = 'vectors' in first_result
            has_indices = 'indices' in first_result
            has_distances = 'distances' in first_result

            success = has_vectors and has_indices and has_distances

            logger.info(
                f"Results include vectors: {has_vectors}, "
                f"indices: {has_indices}, distances: {has_distances}"
            )
        else:
            logger.error("No results returned from search_and_reconstruct")
    except Exception as e:
        logger.error(f"Search and reconstruct test failed: {e}")
        success = False

    logger.info(f"Search and reconstruct test {'PASSED' if success else 'FAILED'}")
    return success


def test_merge_indices(client):
    """Test merging multiple indices"""
    logger.info("\n=== Testing Index Merging ===")

    # Generate different sets of random data
    data1 = generate_test_data(dim=DIM, num=50, seed=123)
    data2 = generate_test_data(dim=DIM, num=50, seed=456)

    # Create index names with unique identifiers
    index1_name = f"merge-source1-{uuid.uuid4().hex[:8]}"
    index2_name = f"merge-source2-{uuid.uuid4().hex[:8]}"
    target_name = f"merge-target-{uuid.uuid4().hex[:8]}"

    # Create indices
    logger.info(f"Creating source indices: {index1_name}, {index2_name}")
    client.create_index(index1_name, DIM, 'L2')
    client.create_index(index2_name, DIM, 'L2')
    client.create_index(target_name, DIM, 'L2')

    # Add vectors to source indices
    logger.info("Adding vectors to source indices")
    client.add_vectors(index1_name, data1)
    client.add_vectors(index2_name, data2)

    # Perform merge
    logger.info(f"Merging indices into {target_name}")
    try:
        merge_result = client.merge_indices(target_name, [index1_name, index2_name])
        success = merge_result.get('success', False)

        if success:
            # Verify merged index has the combined vector count
            ntotal = merge_result.get('ntotal', 0)
            expected_total = len(data1) + len(data2)

            logger.info(f"Merged index has {ntotal} vectors, expected {expected_total}")
            success = ntotal == expected_total
        else:
            logger.error(f"Merge failed: {merge_result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Merge test failed: {e}")
        success = False

    logger.info(f"Index merging test {'PASSED' if success else 'FAILED'}")
    return success


def run_all_tests():
    """Run all advanced feature tests"""
    logger.info("Starting advanced feature tests...")

    # Create client instance
    client = FaissXClient()

    # Configure client to use server if environment variable is set
    server_addr = os.environ.get('FAISSX_SERVER')
    if server_addr:
        logger.info(f"Running in remote mode with server: {server_addr}")
        client.configure(server=server_addr)
    else:
        logger.info("Running in local mode")

    # Run tests
    reconstruction_success = test_reconstruction(client)
    search_reconstruct_success = test_search_and_reconstruct(client)
    merge_success = test_merge_indices(client)

    # Print summary
    logger.info("\n=== Test Results ===")
    logger.info(f"Reconstruction: {'PASSED' if reconstruction_success else 'FAILED'}")
    logger.info(f"Search & Reconstruct: {'PASSED' if search_reconstruct_success else 'FAILED'}")
    logger.info(f"Index Merging: {'PASSED' if merge_success else 'FAILED'}")

    # Overall result
    all_passed = reconstruction_success and search_reconstruct_success and merge_success

    if all_passed:
        logger.info("\nAll advanced feature tests PASSED!")
    else:
        logger.error("\nSome advanced feature tests FAILED!")

    return all_passed


if __name__ == "__main__":
    run_all_tests()

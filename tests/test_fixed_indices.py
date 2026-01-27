#!/usr/bin/env python3
#
# FAISSx Client Specialized Index Test
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
Tests for the fixed specialized index types (PQ and IVF_SQ)
"""

import logging
import os
import time
import uuid

import numpy as np

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


def test_pq_index(client):
    """Test PQ index creation and basic operations"""
    logger.info("\n=== Testing PQ Index ===")

    # Generate random data
    train_data = generate_test_data(dim=DIM, num=NUM_TRAIN)
    test_data = generate_test_data(dim=DIM, num=NUM_TEST, seed=123)
    xq = generate_test_data(dim=DIM, num=5, seed=456)

    # Create index name with a unique identifier
    index_name = f"pq-test-{uuid.uuid4().hex[:8]}"

    # Create the index
    logger.info(f"Creating PQ index: {index_name}")
    start_time = time.time()

    # Create remote index directly
    client.create_index(index_name, DIM, 'PQ4x8')

    # Train the index
    logger.info(f"Training PQ index with {NUM_TRAIN} vectors")
    client.train_index(index_name, train_data)
    logger.info(f"Training completed in {time.time() - start_time:.2f}s")

    # Add vectors
    logger.info(f"Adding {NUM_TEST} vectors to index")
    result = client.add_vectors(index_name, test_data)
    ntotal = result.get('count', 0)
    logger.info(f"Added {ntotal} vectors to index")

    # Search
    logger.info(f"Searching for {len(xq)} queries with k={K}")
    search_result = client.search(index_name, xq, K)
    logger.info(f"Search result status: {search_result.get('success', False)}")

    # Extract results list
    results = search_result.get('results', [])
    if len(results) > 0:
        # Each result has 'distances' and 'indices'
        first_result = results[0]
        logger.info(f"First result - indices: {first_result.get('indices')}")
        logger.info(f"First result - distances: {first_result.get('distances')}")

    # Verify results
    success = (
        search_result.get('success', False) and
        len(results) == len(xq)
    )

    logger.info(f"PQ index test {'PASSED' if success else 'FAILED'}")
    return success


def test_ivf_sq_index(client):
    """Test IVF_SQ index creation and basic operations"""
    logger.info("\n=== Testing IVF_SQ Index ===")

    # Generate random data
    train_data = generate_test_data(dim=DIM, num=NUM_TRAIN)
    test_data = generate_test_data(dim=DIM, num=NUM_TEST, seed=123)
    xq = generate_test_data(dim=DIM, num=5, seed=456)

    # Create index name with a unique identifier
    index_name = f"ivf-sq-test-{uuid.uuid4().hex[:8]}"

    # Create the index
    logger.info(f"Creating IVF_SQ index: {index_name}")
    start_time = time.time()

    # Create remote index directly
    client.create_index(index_name, DIM, 'IVF4_SQ0')

    # Train the index
    logger.info(f"Training IVF_SQ index with {NUM_TRAIN} vectors")
    client.train_index(index_name, train_data)
    logger.info(f"Training completed in {time.time() - start_time:.2f}s")

    # Add vectors
    logger.info(f"Adding {NUM_TEST} vectors to index")
    result = client.add_vectors(index_name, test_data)
    ntotal = result.get('count', 0)
    logger.info(f"Added {ntotal} vectors to index")

    # Search with parameters in the search call directly
    logger.info(f"Searching for {len(xq)} queries with k={K}")
    search_result = client.search(index_name, xq, K, {"nprobe": 2})
    logger.info(f"Search result status: {search_result.get('success', False)}")

    # Extract results list
    results = search_result.get('results', [])
    if len(results) > 0:
        # Each result has 'distances' and 'indices'
        first_result = results[0]
        logger.info(f"First result - indices: {first_result.get('indices')}")
        logger.info(f"First result - distances: {first_result.get('distances')}")

    # Verify results
    success = (
        search_result.get('success', False) and
        len(results) == len(xq)
    )

    logger.info(f"IVF_SQ index test {'PASSED' if success else 'FAILED'}")
    return success


def run_all_tests():
    """Run all index tests"""
    logger.info("Starting specialized index tests...")

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
    pq_success = test_pq_index(client)
    ivf_sq_success = test_ivf_sq_index(client)

    # Print summary
    logger.info("\n=== Test Results ===")
    logger.info(f"PQ Index: {'PASSED' if pq_success else 'FAILED'}")
    logger.info(f"IVF_SQ Index: {'PASSED' if ivf_sq_success else 'FAILED'}")

    if pq_success and ivf_sq_success:
        logger.info("\nAll specialized index tests PASSED!")
    else:
        logger.error("\nSome specialized index tests FAILED!")

    return pq_success and ivf_sq_success


if __name__ == "__main__":
    run_all_tests()

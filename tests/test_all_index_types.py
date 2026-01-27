#!/usr/bin/env python3
#
# FAISSx Client Index Types Test
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
Tests for all available FAISS index types, ensuring they work properly
"""

import logging
import os
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
NUM_QUERY = 5
K = 5  # Number of nearest neighbors to search for

# All index types to test
INDEX_TYPES = [
    # Basic types
    "L2",         # Flat L2 index
    "IP",         # Flat IP index
    "BINARY_FLAT",  # Binary flat index

    # Quantization types
    "PQ4x2",      # Product Quantization with 4 subquantizers, 2 bits each
    "PQ4",        # Product Quantization with 4 subquantizers

    # IVF types
    "IVF16",       # IVF index with 16 clusters
    "IVF16_IP",    # IVF index with IP distance
    "IVF4_SQ8",    # IVF index with 4 clusters and 8-bit scalar quantization

    # Transformation types
    "OPQ4_8,L2",   # OPQ transformation + L2 index
    "PCA4,L2",     # PCA transformation + L2 index
    "NORM,L2",     # L2 normalization + L2 index

    # HNSW types
    "HNSW32",      # HNSW index with 32 neighbors per node
    "HNSW16_IP",   # HNSW index with IP distance

    # ID mapping types
    "IDMap:L2",    # IDMap with L2 flat index
    "IDMap2:L2",   # IDMap2 with L2 flat index
]


def generate_test_data(dim=DIM, num=100, seed=42):
    """Generate random test data"""
    np.random.seed(seed)
    return np.random.random((num, dim)).astype(np.float32)


def test_index_type(client, index_type):
    """Test a specific index type for basic functionality"""
    logger.info(f"\n=== Testing Index Type: {index_type} ===")

    # Generate test data
    train_data = generate_test_data(dim=DIM, num=NUM_TRAIN, seed=42)
    test_data = generate_test_data(dim=DIM, num=NUM_TEST, seed=123)
    query_data = generate_test_data(dim=DIM, num=NUM_QUERY, seed=456)

    # Create unique index name
    index_name = f"{index_type.replace(':', '-')}-{uuid.uuid4().hex[:8]}"

    # Create the index
    logger.info(f"Creating index: {index_name} (type: {index_type})")
    try:
        client.create_index(index_name, DIM, index_type)
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        return False

    # Train the index if needed
    needs_training = (
        "IVF" in index_type or
        "PQ" in index_type or
        "OPQ" in index_type
    )

    if needs_training:
        logger.info(f"Training index with {NUM_TRAIN} vectors")
        try:
            train_result = client.train_index(index_name, train_data)
            if not train_result.get('success', False):
                logger.error(f"Training failed: {train_result.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"Exception during training: {e}")
            return False

    # Add vectors
    logger.info(f"Adding {NUM_TEST} vectors")
    try:
        add_result = client.add_vectors(index_name, test_data)
        if not add_result.get('success', False):
            logger.error(f"Adding vectors failed: {add_result.get('error', 'Unknown error')}")
            return False

        # Check ntotal
        ntotal = add_result.get('total', 0)
        logger.info(f"Index now contains {ntotal} vectors")

        # Binary indices sometimes have different vector counting
        if "BINARY" not in index_type and ntotal != NUM_TEST:
            logger.warning(f"Vector count mismatch: expected {NUM_TEST}, got {ntotal}")
    except Exception as e:
        logger.error(f"Exception adding vectors: {e}")
        return False

    # Perform search
    logger.info(f"Searching with {NUM_QUERY} query vectors")
    try:
        search_result = client.search(index_name, query_data, K)
        if not search_result.get('success', False):
            logger.error(f"Search failed: {search_result.get('error', 'Unknown error')}")
            return False

        # Check if we got results
        results = search_result.get('results', [])
        if len(results) != NUM_QUERY:
            logger.error(f"Search returned {len(results)} results, expected {NUM_QUERY}")
            return False

        # Check if each result has the right number of entries
        first_result = results[0]
        if 'distances' not in first_result or 'indices' not in first_result:
            logger.error("Search results missing distances or indices")
            return False

        actual_k = len(first_result['distances'])
        logger.info(f"Search returned {actual_k} neighbors per query")

        # Some indices might return fewer results than requested
        if not (actual_k > 0 and actual_k <= K):
            logger.error(f"Got {actual_k} neighbors, expected between 1 and {K}")
            return False
    except Exception as e:
        logger.error(f"Exception during search: {e}")
        return False

    # Try to get index stats
    try:
        stats = client.get_index_stats(index_name)
        if stats.get('success', False):
            index_type_reported = stats.get('index_type', 'Unknown')
            logger.info(f"Index type from stats: {index_type_reported}")
    except Exception:
        # Stats are optional, don't fail the test if they don't work
        pass

    # Report success
    logger.info(f"Index {index_type} test PASSED")
    return True


def run_tests():
    """Run tests for all specified index types"""
    logger.info("Starting index type compatibility tests...")

    # Create client instance
    client = FaissXClient()

    # Configure client to use server if environment variable is set
    server_addr = os.environ.get('FAISSX_SERVER')
    if server_addr:
        logger.info(f"Running in remote mode with server: {server_addr}")
        client.configure(server=server_addr)
    else:
        logger.info("Running in local mode")

    # Test each index type
    results = {}

    for index_type in INDEX_TYPES:
        success = test_index_type(client, index_type)
        results[index_type] = success

    # Print summary
    logger.info("\n=== Test Results Summary ===")
    passed = 0
    failed = 0

    for index_type, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{index_type.ljust(15)}: {status}")
        if success:
            passed += 1
        else:
            failed += 1

    logger.info(f"\nOverall: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    run_tests()

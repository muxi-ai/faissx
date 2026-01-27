#!/usr/bin/env python3
#
# FAISSx Client ntotal test
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
A simple test to verify the ntotal fix
"""

import logging
import os
import uuid

import numpy as np

from faissx.client.client import FaissXClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_ntotal():
    """Test that ntotal is properly returned in add_vectors response"""
    logger.info("Testing ntotal in add_vectors response...")

    # Create client instance
    client = FaissXClient()

    # Configure client to use server if environment variable is set
    server_addr = os.environ.get('FAISSX_SERVER')
    if server_addr:
        logger.info(f"Running in remote mode with server: {server_addr}")
        client.configure(server=server_addr)
        client.connect()
    else:
        logger.info("Running in local mode")

    # Generate test data
    dimension = 8
    num_vectors = 10
    np.random.seed(42)
    test_data = np.random.random((num_vectors, dimension)).astype(np.float32)

    # Create index with unique name
    index_name = f"ntotal-test-{uuid.uuid4().hex[:8]}"
    logger.info(f"Creating index: {index_name}")
    client.create_index(index_name, dimension, "L2")

    try:
        # Add vectors in two batches
        logger.info(f"Adding first batch of {num_vectors} vectors")
        result1 = client.add_vectors(index_name, test_data)
        assert result1["success"], f"Failed to add vectors: {result1.get('error')}"

        # Check if ntotal and total are in the response
        ntotal1 = result1.get('ntotal', None)
        total1 = result1.get('total', None)
        count1 = result1.get('count', None)

        logger.info(f"First add result - ntotal: {ntotal1}, total: {total1}, count: {count1}")

        # Add second batch
        logger.info(f"Adding second batch of {num_vectors} vectors")
        result2 = client.add_vectors(index_name, test_data)
        assert result2["success"], f"Failed to add vectors: {result2.get('error')}"

        # Check if ntotal and total are in the response and have increased
        ntotal2 = result2.get('ntotal', None)
        total2 = result2.get('total', None)
        count2 = result2.get('count', None)

        logger.info(f"Second add result - ntotal: {ntotal2}, total: {total2}, count: {count2}")

        # Verify that the counts increased correctly
        assert ntotal1 is not None, "ntotal missing from first response"
        assert total1 is not None, "total missing from first response"
        assert count1 is not None, "count missing from first response"
        assert ntotal2 is not None, "ntotal missing from second response"
        assert total2 is not None, "total missing from second response"
        assert count2 is not None, "count missing from second response"

        assert ntotal1 == num_vectors, f"Expected ntotal1={num_vectors}, got {ntotal1}"
        assert total1 == num_vectors, f"Expected total1={num_vectors}, got {total1}"
        assert count1 == num_vectors, f"Expected count1={num_vectors}, got {count1}"
        assert ntotal2 == 2 * num_vectors, f"Expected ntotal2={2*num_vectors}, got {ntotal2}"
        assert total2 == 2 * num_vectors, f"Expected total2={2*num_vectors}, got {total2}"
        assert count2 == num_vectors, f"Expected count2={num_vectors}, got {count2}"

        logger.info("PASSED: ntotal correctly returned and incremented")

    finally:
        # Clean up
        try:
            client.delete_index(index_name)
        except Exception as e:
            logger.error(f"Failed to clean up index {index_name}: {e}")

        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    test_ntotal()

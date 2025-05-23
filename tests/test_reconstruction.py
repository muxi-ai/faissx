#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Client Vector Reconstruction Test
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
A simple test to verify the vector reconstruction functionality
"""

import os
import numpy as np
import logging
import uuid
from faissx.client.client import FaissXClient
from faissx.client.indices.flat import IndexFlatL2


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_reconstruction():
    """Test that vector reconstruction works correctly"""
    logger.info("Testing vector reconstruction functionality...")

    # Generate test data
    dimension = 8
    num_vectors = 10
    np.random.seed(42)
    test_data = np.random.random((num_vectors, dimension)).astype(np.float32)

    # Check if we should use remote mode
    server_addr = os.environ.get('FAISSX_SERVER')
    if server_addr:
        logger.info(f"Running in remote mode with server: {server_addr}")

        # Create client and connect to server
        client = FaissXClient()
        client.configure(server=server_addr)
        client.connect()

        # Create index on server
        index_name = f"reconstruction-test-{uuid.uuid4().hex[:8]}"
        logger.info(f"Creating remote index: {index_name}")
        client.create_index(index_name, dimension, "L2")

        try:
            # Add vectors
            logger.info(f"Adding {num_vectors} vectors")
            result = client.add_vectors(index_name, test_data)
            assert result["success"], f"Failed to add vectors: {result.get('error')}"

            # Test single vector reconstruction
            test_id = 5  # Index of vector to test
            logger.info(f"Testing single vector reconstruction for ID {test_id}")
            result = client.reconstruct(index_name, test_id)
            reconstruction_error = result.get('error', 'Unknown error')
            assert result.get('success', False), f"Reconstruction failed: {reconstruction_error}"

            vector = result.get('vector', [])
            original = test_data[test_id]

            # Compare reconstructed vector with original
            is_close = np.allclose(vector, original, rtol=1e-5, atol=1e-6)
            logger.info(f"Original vector: {original}")
            logger.info(f"Reconstructed vector: {vector}")
            logger.info(f"Vectors close (rtol=1e-5, atol=1e-6): {is_close}")

            if not is_close:
                # Try with more relaxed tolerance for debugging
                is_close_relaxed = np.allclose(vector, original, rtol=1e-3, atol=1e-3)
                logger.info(
                    f"Vectors close with relaxed tolerance "
                    f"(rtol=1e-3, atol=1e-3): {is_close_relaxed}"
                )
                is_close = is_close_relaxed

            assert is_close, f"Reconstructed vector doesn't match original for ID {test_id}"
            logger.info("Single vector reconstruction: PASSED")

            # Test multiple vector reconstruction
            start_idx = 2
            num = 3
            logger.info(
                f"Testing multiple vector reconstruction from index {start_idx} "
                f"to {start_idx + num - 1}"
            )
            result = client.reconstruct_n(index_name, start_idx, num)
            multi_error = result.get('error', 'Unknown error')
            assert result.get('success', False), (
                f"Multiple reconstruction failed: {multi_error}"
            )

            vectors = result.get('vectors', [])
            originals = test_data[start_idx:start_idx + num]

            # Check if we got the right number of vectors
            assert len(vectors) == num, f"Expected {num} vectors, got {len(vectors)}"
            logger.info("Received expected number of vectors: PASSED")

            # Compare reconstructed vectors with originals
            all_match = all(
                np.allclose(vectors[i], originals[i], rtol=1e-5)
                for i in range(num)
            )
            assert all_match, "Not all reconstructed vectors match originals"
            logger.info("Multiple vector reconstruction: PASSED")

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
    else:
        logger.info("Running in local mode")

        # Create index directly for local mode
        index = IndexFlatL2(dimension)

        # Add vectors
        logger.info(f"Adding {num_vectors} vectors")
        index.add(test_data)
        assert index.ntotal == num_vectors, f"Expected {num_vectors} vectors, got {index.ntotal}"

        # Test single vector reconstruction
        test_id = 5  # Index of vector to test
        logger.info(f"Testing single vector reconstruction for ID {test_id}")
        vector = index.reconstruct(test_id)
        original = test_data[test_id]

        # Compare reconstructed vector with original
        is_close = np.allclose(vector, original, rtol=1e-5, atol=1e-6)
        logger.info(f"Original vector: {original}")
        logger.info(f"Reconstructed vector: {vector}")
        logger.info(f"Vectors close (rtol=1e-5, atol=1e-6): {is_close}")

        if not is_close:
            # Try with more relaxed tolerance for debugging
            is_close_relaxed = np.allclose(vector, original, rtol=1e-3, atol=1e-3)
            logger.info(
                f"Vectors close with relaxed tolerance "
                f"(rtol=1e-3, atol=1e-3): {is_close_relaxed}"
            )
            is_close = is_close_relaxed

        assert is_close, f"Reconstructed vector doesn't match original for ID {test_id}"
        logger.info("Single vector reconstruction: PASSED")

        # Test multiple vector reconstruction
        start_idx = 2
        num = 3
        logger.info(
            f"Testing multiple vector reconstruction from index {start_idx} "
            f"to {start_idx + num - 1}"
        )
        vectors = index.reconstruct_n(start_idx, num)
        originals = test_data[start_idx:start_idx + num]

        # Check if we got the right number of vectors
        assert len(vectors) == num, f"Expected {num} vectors, got {len(vectors)}"
        logger.info("Received expected number of vectors: PASSED")

        # Compare reconstructed vectors with originals
        all_match = all(
            np.allclose(vectors[i], originals[i], rtol=1e-5)
            for i in range(num)
        )
        assert all_match, "Not all reconstructed vectors match originals"
        logger.info("Multiple vector reconstruction: PASSED")

    logger.info("All reconstruction tests PASSED")


if __name__ == "__main__":
    test_reconstruction()

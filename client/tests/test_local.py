#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
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
FAISSx Local Mode Test

This module tests that FAISSx uses local FAISS implementation by default
when no configure() method is called. This should make it a true drop-in
replacement for FAISS without requiring any additional configuration.
"""

import os
import sys
import unittest
import numpy as np
import logging

# Ensure we're using the client from our project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)


class TestLocalMode(unittest.TestCase):
    """Test that FAISSx works as a drop-in replacement for FAISS when configure() is not called"""

    def setUp(self):
        """Clear any existing environment variables that might affect the test"""
        # Save original environment variables
        self.original_env = {}
        env_vars = [
            'FAISSX_SERVER',
            'FAISSX_API_KEY',
            'FAISSX_TENANT_ID'
        ]
        for key in env_vars:
            self.original_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]

        # Import after environment changes to ensure clean state
        self.imported = False

    def tearDown(self):
        """Restore original environment variables"""
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def test_local_without_configure(self):
        """Test if client uses local FAISS when configure() is not called"""
        # Avoid importing at the top level to ensure environment variables take effect
        from faissx import client as faiss
        self.imported = True

        # We don't call configure()

        # Create index, add vectors, and perform search
        try:
            # Create a small index
            dimension = 10
            index = faiss.IndexFlatL2(dimension)

            # Create some vectors
            num_vectors = 5
            vectors = np.random.random((num_vectors, dimension)).astype('float32')

            # Add vectors to the index
            index.add(vectors)

            # Check that vectors were added
            self.assertEqual(index.ntotal, num_vectors)

            # Search for a vector
            query = np.random.random((1, dimension)).astype('float32')
            distances, indices = index.search(query, k=3)

            # Verify search results structure
            self.assertEqual(distances.shape, (1, 3))
            self.assertEqual(indices.shape, (1, 3))

            # Log success
            logging.info("Successfully used local FAISS without configure()")
            logging.info(f"Found {len(indices[0])} results for query")

            # Verify this is a real FAISS implementation
            self.verify_local_implementation(index)

        except Exception as e:
            logging.error(f"Error during local mode test: {str(e)}")
            self.fail(f"Local FAISS usage failed: {str(e)}")

    def verify_local_implementation(self, index):
        """Verify we're using the local FAISS implementation"""
        # Get type information to check implementation
        index_type = type(index)
        implementation_details = str(index_type)
        logging.info(f"Index implementation: {implementation_details}")

        # Local FAISS has these properties/methods
        self.assertTrue(hasattr(index, 'reset') and callable(index.reset))
        self.assertTrue(hasattr(index, 'ntotal'))
        self.assertTrue(hasattr(index, 'd'))
        self.assertTrue(hasattr(index, 'is_trained'))


if __name__ == "__main__":
    unittest.main()

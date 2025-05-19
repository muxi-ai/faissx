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
FAISSx Basic Client Tests

This module contains unit tests for the FAISSx client library,
testing both the low-level client API and the high-level index interface.
It verifies correct communication with the FAISSx server for operations
such as creating indices, adding vectors, and performing similarity searches.
"""

import os
import sys
import time
import unittest
import numpy as np
import uuid

# Add parent directory to path to import faissx client
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from faissx import client as faiss_client
from faissx.client import FaissXClient


class TestFAISSxClient(unittest.TestCase):
    """Test the FAISSx ZeroMQ client implementation"""

    def setUp(self):
        """Set up client connection to the server"""
        server = os.environ.get("FAISSX_SERVER", "tcp://localhost:45678")
        api_key = os.environ.get("FAISSX_API_KEY", "test-key-1")
        tenant_id = os.environ.get("FAISSX_TENANT_ID", "tenant-1")
        self.client = FaissXClient(server=server, api_key=api_key, tenant_id=tenant_id)

    def tearDown(self):
        """Clean up client connection"""
        if hasattr(self, 'client'):
            self.client.close()

    def test_ping(self):
        """Test basic ping to the server"""
        response = self.client._send_request({"action": "ping"})
        self.assertTrue(response.get("success"))
        self.assertEqual(response.get("message"), "pong")

    def test_create_index(self):
        """Test creating an index"""
        test_name = f"test-index-{uuid.uuid4().hex[:8]}"
        index_id = self.client.create_index(test_name, 128)
        self.assertEqual(index_id, test_name)

    def test_add_vectors_and_search(self):
        """Test adding vectors and searching"""
        # Create an index
        test_name = f"test-search-{uuid.uuid4().hex[:8]}"
        index_id = self.client.create_index(test_name, 128)

        # Create 10 random vectors
        vectors = np.random.random((10, 128)).astype(np.float32)

        # Add vectors to the index
        result = self.client.add_vectors(index_id, vectors)
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("count"), 10)

        # Search for vectors
        query = np.random.random((1, 128)).astype(np.float32)
        search_result = self.client.search(index_id, query, k=5)

        # Verify search results structure
        self.assertTrue(search_result.get("success"))
        results = search_result.get("results", [])
        self.assertEqual(len(results), 1)  # 1 query vector

        # First result should have distances and indices
        self.assertIn("distances", results[0])
        self.assertIn("indices", results[0])

        # Should have returned k results
        self.assertEqual(len(results[0]["distances"]), 5)
        self.assertEqual(len(results[0]["indices"]), 5)

    def test_get_index_stats(self):
        """Test getting index stats"""
        # Create an index
        test_name = f"test-stats-{uuid.uuid4().hex[:8]}"
        index_id = self.client.create_index(test_name, 64)

        # Add vectors
        vectors = np.random.random((5, 64)).astype(np.float32)
        self.client.add_vectors(index_id, vectors)

        # Get index stats
        stats = self.client.get_index_stats(index_id)

        # Verify stats
        self.assertTrue(stats.get("success"))
        stats_data = stats.get("stats", {})
        self.assertEqual(stats_data.get("index_id"), index_id)
        self.assertEqual(stats_data.get("dimension"), 64)
        self.assertEqual(stats_data.get("vector_count"), 5)


class TestFAISSxIndex(unittest.TestCase):
    """Test the FAISSx index implementation"""

    def setUp(self):
        """Set up client configuration"""
        server = os.environ.get("FAISSX_SERVER", "tcp://localhost:45678")
        api_key = os.environ.get("FAISSX_API_KEY", "test-key-1")
        tenant_id = os.environ.get("FAISSX_TENANT_ID", "tenant-1")
        faiss_client.configure(server=server, api_key=api_key, tenant_id=tenant_id)

    def test_index_flat_l2(self):
        """Test IndexFlatL2 implementation"""
        # Create an index
        index = faiss_client.IndexFlatL2(128)

        # Add vectors
        vectors = np.random.random((10, 128)).astype(np.float32)
        index.add(vectors)

        # Verify ntotal is updated
        self.assertEqual(index.ntotal, 10)

        # Search for vectors
        query = np.random.random((2, 128)).astype(np.float32)
        distances, indices = index.search(query, k=3)

        # Verify search results shape
        self.assertEqual(distances.shape, (2, 3))
        self.assertEqual(indices.shape, (2, 3))

    def test_reset(self):
        """Test index reset functionality"""
        # Create an index
        index = faiss_client.IndexFlatL2(64)

        # Add vectors
        vectors = np.random.random((5, 64)).astype(np.float32)
        index.add(vectors)

        # Verify vectors were added
        self.assertEqual(index.ntotal, 5)

        # Reset the index
        index.reset()

        # Verify ntotal is reset
        self.assertEqual(index.ntotal, 0)


if __name__ == "__main__":
    unittest.main()

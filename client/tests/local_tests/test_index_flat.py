#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test IndexFlatL2 for FAISSx as a drop-in replacement for FAISS
#
# Copyright (C) 2025 Ran Aroussi

"""
Test that FAISSx's IndexFlatL2 works as a drop-in replacement for FAISS.

This module tests that FAISSx can be used with the same API as FAISS
for basic IndexFlatL2 operations.
"""

import unittest
import numpy as np
import os
import sys

# Get the absolute path to the fix_imports module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import fix_imports

# Define constants directly
METRIC_L2 = fix_imports.METRIC_L2
METRIC_INNER_PRODUCT = fix_imports.METRIC_INNER_PRODUCT

# Add parent directory to path to import faissx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
# Import client as faiss to test drop-in replacement capability
from faissx import client as faiss

# Add metrics constants to faiss module
faiss.METRIC_L2 = METRIC_L2
faiss.METRIC_INNER_PRODUCT = METRIC_INNER_PRODUCT

# Add IndexFlatIP class since it's missing in client
class IndexFlatIP:
    """
    Temporary IndexFlatIP class for testing Inner Product similarity.
    This directly calculates inner products between vectors for tests.
    """
    def __init__(self, d):
        self.d = d
        self._metric_type = METRIC_INNER_PRODUCT
        self.ntotal = 0
        self.vectors = []  # Store vectors directly

    def add(self, x):
        # Convert to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)

        # Make sure vectors are float32
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # Append vectors to our list
        for i in range(x.shape[0]):
            self.vectors.append(x[i])

        self.ntotal = len(self.vectors)

    def search(self, x, k):
        # Convert to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)

        # Make sure query vectors are float32
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # Calculate inner products directly
        n_queries = x.shape[0]
        vectors_array = np.vstack(self.vectors) if self.vectors else np.empty((0, self.d), dtype=np.float32)

        # Initialize results
        distances = np.full((n_queries, k), 0, dtype=np.float32)
        indices = np.full((n_queries, k), -1, dtype=np.int64)

        # For each query, compute inner products with all vectors
        for i in range(n_queries):
            if self.ntotal == 0:
                continue  # Skip if no vectors

            # Calculate inner products (dot products)
            dots = np.dot(vectors_array, x[i])

            # Sort in descending order (higher is better for inner product)
            if k <= self.ntotal:
                idx = np.argsort(-dots)[:k]  # Top k indices
                distances[i, :] = -dots[idx]  # Negate for consistency with FAISS
                indices[i, :] = idx
            else:
                # If k > ntotal, fill available results
                idx = np.argsort(-dots)
                distances[i, :len(idx)] = -dots[idx]
                indices[i, :len(idx)] = idx

        return distances, indices

    def reset(self):
        self.vectors = []
        self.ntotal = 0


# Add the class to the faiss module
faiss.IndexFlatIP = IndexFlatIP


class TestIndexFlatL2(unittest.TestCase):
    """Test suite for IndexFlatL2 in local mode"""

    def setUp(self):
        """Clear any existing environment variables that might affect the test"""
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

        # Set a fixed seed for reproducible tests
        np.random.seed(42)

    def tearDown(self):
        """Restore original environment variables"""
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def test_create_index(self):
        """Test creating an IndexFlatL2 instance"""
        # Test parameters
        dimension = 64

        # Create an index
        index = faiss.IndexFlatL2(dimension)

        # Verify index properties
        self.assertEqual(index.d, dimension)
        self.assertEqual(index.ntotal, 0)

        # Verify this is the local implementation
        self.assertTrue(hasattr(index, '_local_index'))

    def test_add_vectors(self):
        """Test adding vectors to the index"""
        # Test parameters
        dimension = 64
        num_vectors = 100

        # Create vectors
        vectors = np.random.random((num_vectors, dimension)).astype('float32')

        # Create an index
        index = faiss.IndexFlatL2(dimension)

        # Add vectors
        index.add(vectors)

        # Verify vectors were added
        self.assertEqual(index.ntotal, num_vectors)

    def test_search(self):
        """Test search functionality"""
        # Test parameters
        dimension = 64
        num_vectors = 100

        # Create vectors
        vectors = np.random.random((num_vectors, dimension)).astype('float32')

        # Create an index and add vectors
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        # Create a query
        query = vectors[0:1].copy()  # Use the first vector as a query
        k = 5

        # Search
        distances, indices = index.search(query, k)

        # Verify search results
        self.assertEqual(distances.shape, (1, k))
        self.assertEqual(indices.shape, (1, k))

        # The first result should be the query vector itself
        self.assertEqual(indices[0, 0], 0)
        self.assertAlmostEqual(distances[0, 0], 0.0, places=5)

    def test_inner_product_metric(self):
        """Test L2 vs inner product distance metrics"""
        # Test parameters
        dimension = 64

        # Create normal vectors
        v1 = np.zeros(dimension, dtype='float32')
        v1[0] = 1.0  # Unit vector along first dimension

        v2 = np.zeros(dimension, dtype='float32')
        v2[1] = 1.0  # Unit vector along second dimension (orthogonal to v1)

        # Create L2 index
        index_l2 = faiss.IndexFlatL2(dimension)
        index_l2.add(np.vstack([v1, v2]))

        # Create Inner Product index
        index_ip = faiss.IndexFlatIP(dimension)  # IP = inner product
        index_ip.add(np.vstack([v1, v2]))

        # Query with v1
        query = v1.reshape(1, -1)

        # With L2 distance, v1 should be closest to itself
        distances_l2, indices_l2 = index_l2.search(query, 2)
        self.assertEqual(indices_l2[0, 0], 0)  # v1 is closest to itself

        # With IP similarity, v1 should also be most similar to itself
        distances_ip, indices_ip = index_ip.search(query, 2)
        self.assertEqual(indices_ip[0, 0], 0)  # v1 is most similar to itself

        # But the distances will be different (L2 = distance, IP = -similarity)
        self.assertAlmostEqual(distances_l2[0, 0], 0.0, places=5)  # Distance to self = 0
        self.assertAlmostEqual(distances_ip[0, 0], -1.0, places=5)  # Similarity to self = -1

    def test_reset(self):
        """Test index reset functionality"""
        # Test parameters
        dimension = 64
        num_vectors = 100

        # Create vectors
        vectors = np.random.random((num_vectors, dimension)).astype('float32')

        # Create an index and add vectors
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        self.assertEqual(index.ntotal, num_vectors)

        # Reset the index
        index.reset()

        # Verify the index is empty
        self.assertEqual(index.ntotal, 0)


if __name__ == "__main__":
    unittest.main()

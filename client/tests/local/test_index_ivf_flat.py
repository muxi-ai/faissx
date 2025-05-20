#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test IndexIVFFlat for FAISSx as a drop-in replacement for FAISS
#
# Copyright (C) 2025 Ran Aroussi

"""
Test that FAISSx's IndexIVFFlat works as a drop-in replacement for FAISS.

This module tests that FAISSx can be used with the same API as FAISS
for IndexIVFFlat operations including training and searching with nprobe.
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
from faissx import client as faiss

# Add metrics constants to faiss module
faiss.METRIC_L2 = METRIC_L2
faiss.METRIC_INNER_PRODUCT = METRIC_INNER_PRODUCT


class TestIndexIVFFlat(unittest.TestCase):
    """Test suite for IndexIVFFlat in local mode"""

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

        # Create standard test data
        self.dimension = 64
        self.nlist = 10

        # Generate training data
        self.training_size = 1000
        self.training_vectors = np.random.random((self.training_size, self.dimension)).astype('float32')

        # Generate vectors to add
        self.num_vectors = 500
        self.vectors = np.random.random((self.num_vectors, self.dimension)).astype('float32')

        # Generate query vectors
        self.num_queries = 10
        self.queries = np.random.random((self.num_queries, self.dimension)).astype('float32')

    def tearDown(self):
        """Restore original environment variables"""
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def _create_trained_index(self):
        """Helper to create and train an index with the test data"""
        # Create a quantizer
        quantizer = faiss.IndexFlatL2(self.dimension)

        # Create an IVF index
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)

        # Train the index
        index.train(self.training_vectors)

        return index

    def test_create_index(self):
        """Test creating an IndexIVFFlat instance"""
        # Create a quantizer
        quantizer = faiss.IndexFlatL2(self.dimension)

        # Create an IVF index
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)

        self.assertEqual(index.d, self.dimension)
        self.assertEqual(index.ntotal, 0)
        self.assertFalse(index.is_trained)
        self.assertEqual(index.nlist, self.nlist)

        # Verify this is a real FAISS implementation
        self.assertTrue(hasattr(index, '_local_index'))

    def test_train_index(self):
        """Test training an IndexIVFFlat"""
        index = self._create_trained_index()

        # Verify the index is trained
        self.assertTrue(index.is_trained)

    def test_add_vectors(self):
        """Test adding vectors to a trained index"""
        index = self._create_trained_index()

        # Add vectors
        index.add(self.vectors)

        # Verify vectors were added
        self.assertEqual(index.ntotal, self.num_vectors)

    def test_search(self):
        """Test search functionality"""
        index = self._create_trained_index()
        index.add(self.vectors)

        # Search with default nprobe
        k = 5
        distances, indices = index.search(self.queries, k)

        # Verify search results shape
        self.assertEqual(distances.shape, (self.num_queries, k))
        self.assertEqual(indices.shape, (self.num_queries, k))

        # Verify all returned indices are valid
        self.assertTrue(np.all(indices >= -1))  # -1 means not found
        self.assertTrue(np.all(indices < self.num_vectors))

    def test_nprobe_parameter(self):
        """Test setting nprobe parameter affects search results"""
        index = self._create_trained_index()
        index.add(self.vectors)

        # Set a small nprobe (more approximate, faster)
        index.nprobe = 1
        k = 10
        distances_small_nprobe, _ = index.search(self.queries, k)

        # Set a larger nprobe (more accurate, slower)
        index.nprobe = self.nlist  # Search all clusters
        distances_large_nprobe, _ = index.search(self.queries, k)

        # With larger nprobe, distances should be smaller or equal
        # (better or same matches)
        better_or_same = (distances_large_nprobe <= distances_small_nprobe).sum()

        # Allow some tolerance, but most results should be better or same
        self.assertGreaterEqual(better_or_same, self.num_queries * k * 0.8)

    def test_reset(self):
        """Test reset functionality"""
        index = self._create_trained_index()
        index.add(self.vectors)

        self.assertEqual(index.ntotal, self.num_vectors)

        # Reset index
        index.reset()

        # Verify ntotal is reset but index remains trained
        self.assertEqual(index.ntotal, 0)
        self.assertTrue(index.is_trained)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        index = self._create_trained_index()

        # Test adding vectors before training (should work since we trained already)
        vectors = np.random.random((5, self.dimension)).astype('float32')
        index.add(vectors)

        # Test adding vectors with wrong dimension
        wrong_dim_vectors = np.random.random((5, self.dimension+1)).astype('float32')
        with self.assertRaises(ValueError):
            index.add(wrong_dim_vectors)

        # Test searching with wrong dimension
        wrong_dim_query = np.random.random((1, self.dimension+1)).astype('float32')
        with self.assertRaises(ValueError):
            index.search(wrong_dim_query, 5)


if __name__ == "__main__":
    unittest.main()

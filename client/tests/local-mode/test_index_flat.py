#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for IndexFlatL2 in local mode (no configure call).
This tests that FAISSx truly works as a drop-in replacement for FAISS.
"""

import unittest
import numpy as np

# Import FAISSx client as faiss - this should use local mode by default
from faissx import client as faiss


class TestIndexFlatL2LocalMode(unittest.TestCase):
    """Test IndexFlatL2 in local mode"""

    def setUp(self):
        """Set up test vectors and dimensions"""
        self.d = 64  # dimensions
        self.nb = 1000  # database size
        self.nq = 10  # nb of queries

        # Generate random test data
        np.random.seed(42)  # for reproducibility
        self.xb = np.random.random((self.nb, self.d)).astype('float32')
        self.xq = np.random.random((self.nq, self.d)).astype('float32')

    def test_create_index(self):
        """Test creating an IndexFlatL2"""
        index = faiss.IndexFlatL2(self.d)
        self.assertEqual(index.d, self.d)
        self.assertEqual(index.ntotal, 0)

    def test_add_vectors(self):
        """Test adding vectors to the index"""
        index = faiss.IndexFlatL2(self.d)
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

    def test_search(self):
        """Test searching for nearest vectors"""
        index = faiss.IndexFlatL2(self.d)
        index.add(self.xb)

        k = 5  # return 5 nearest neighbors
        distances, indices = index.search(self.xq, k)

        # Verify shapes
        self.assertEqual(distances.shape, (self.nq, k))
        self.assertEqual(indices.shape, (self.nq, k))

        # Verify distances are non-negative and sorted (ascending)
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.all(distances[:, 1:] >= distances[:, :-1]))

        # Verify indices are within bounds
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < self.nb))

    def test_reset(self):
        """Test resetting the index"""
        index = faiss.IndexFlatL2(self.d)
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        index.reset()
        self.assertEqual(index.ntotal, 0)

        # Verify search after reset returns empty results
        k = 5
        distances, indices = index.search(self.xq, k)
        # After reset with no vectors, indices should be -1 (no results)
        self.assertTrue(np.all(indices == -1))


if __name__ == "__main__":
    unittest.main()

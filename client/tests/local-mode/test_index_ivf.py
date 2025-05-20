#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for IVF indices (IndexIVFFlat, IndexIVFPQ) in local mode.
This tests that FAISSx truly works as a drop-in replacement for FAISS.
"""

import unittest
import numpy as np

# Import FAISSx client as faiss - this should use local mode by default
from faissx import client as faiss


class TestIndexIVFLocalMode(unittest.TestCase):
    """Test IVF indices in local mode"""

    def setUp(self):
        """Set up test vectors and dimensions"""
        self.d = 64  # dimensions
        self.nb = 5000  # database size
        self.nq = 10  # nb of queries
        self.nlist = 100  # number of clusters for IVF

        # Generate random test data
        np.random.seed(42)  # for reproducibility
        self.xb = np.random.random((self.nb, self.d)).astype('float32')
        self.xq = np.random.random((self.nq, self.d)).astype('float32')

        # Create a quantizer for IVF indices
        self.quantizer = faiss.IndexFlatL2(self.d)

    def test_ivf_flat(self):
        """Test IndexIVFFlat functionality"""
        # Create index
        index = faiss.IndexIVFFlat(self.quantizer, self.d, self.nlist)

        # Check properties before training
        self.assertEqual(index.d, self.d)
        self.assertEqual(index.nlist, self.nlist)
        self.assertEqual(index.ntotal, 0)

        # Train the index
        index.train(self.xb)
        self.assertTrue(index.is_trained)

        # Add vectors
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        # Set nprobe parameter
        index.nprobe = 10
        self.assertEqual(index.nprobe, 10)

        # Search
        k = 5
        distances, indices = index.search(self.xq, k)

        # Verify shapes
        self.assertEqual(distances.shape, (self.nq, k))
        self.assertEqual(indices.shape, (self.nq, k))

        # Verify distances are non-negative and sorted
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.all(distances[:, 1:] >= distances[:, :-1]))

    def test_ivf_pq(self):
        """Test IndexIVFPQ functionality"""
        # Product Quantization params - ensure m divides d evenly
        # Since d = 64, valid values for m are 1, 2, 4, 8, 16, 32, 64
        m = 4  # number of subquantizers (must be a divisor of d)
        bits = 8  # bits per code (usually 8)

        # Create index
        index = faiss.IndexIVFPQ(self.quantizer, self.d, self.nlist, m, bits)

        # Check properties
        self.assertEqual(index.d, self.d)
        self.assertEqual(index.nlist, self.nlist)
        self.assertEqual(index.pq.M, m)
        self.assertEqual(index.ntotal, 0)

        # Train the index
        index.train(self.xb)
        self.assertTrue(index.is_trained)

        # Add vectors
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        # Set parameters
        index.nprobe = 10
        self.assertEqual(index.nprobe, 10)

        # Search
        k = 5
        distances, indices = index.search(self.xq, k)

        # Verify shapes
        self.assertEqual(distances.shape, (self.nq, k))
        self.assertEqual(indices.shape, (self.nq, k))

        # Verify distances are non-negative and sorted
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.all(distances[:, 1:] >= distances[:, :-1]))

    def test_reset(self):
        """Test resetting of IVF indices"""
        # Create and train index
        index = faiss.IndexIVFFlat(self.quantizer, self.d, self.nlist)
        index.train(self.xb)
        index.add(self.xb)

        # Reset
        index.reset()
        self.assertEqual(index.ntotal, 0)
        self.assertTrue(index.is_trained)  # Training should be preserved


if __name__ == "__main__":
    unittest.main()

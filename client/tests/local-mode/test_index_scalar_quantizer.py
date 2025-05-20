#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for Scalar Quantizer indices in local mode.
This tests that FAISSx truly works as a drop-in replacement for FAISS.
"""

import unittest
import numpy as np

# Import FAISSx client as faiss - this should use local mode by default
from faissx import client as faiss


class TestScalarQuantizerLocalMode(unittest.TestCase):
    """Test Scalar Quantizer indices in local mode"""

    def setUp(self):
        """Set up test vectors and dimensions"""
        self.d = 64  # dimensions
        self.nb = 1000  # database size
        self.nq = 10  # nb of queries

        # Generate random test data
        np.random.seed(42)  # for reproducibility
        self.xb = np.random.random((self.nb, self.d)).astype('float32')
        self.xq = np.random.random((self.nq, self.d)).astype('float32')

    def test_index_scalar_quantizer(self):
        """Test IndexScalarQuantizer functionality"""
        # Create index with scalar quantization
        index = faiss.IndexScalarQuantizer(self.d, faiss.ScalarQuantizer.QT_8bit)

        # Check properties
        self.assertEqual(index.d, self.d)
        self.assertEqual(index.ntotal, 0)

        # Add vectors
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        # Search
        k = 5
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

    def test_ivf_scalar_quantizer(self):
        """Test IndexIVFScalarQuantizer functionality"""
        # Parameters
        nlist = 50  # number of clusters

        # Create quantizer for IVF
        quantizer = faiss.IndexFlatL2(self.d)

        # Create IVF index with scalar quantization
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, self.d, nlist, faiss.ScalarQuantizer.QT_8bit
        )

        # Check properties
        self.assertEqual(index.d, self.d)
        self.assertEqual(index.nlist, nlist)
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

        # Verify indices are within bounds
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < self.nb))


if __name__ == "__main__":
    unittest.main()

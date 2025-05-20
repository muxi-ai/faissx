#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for HNSW indices (IndexHNSWFlat) in local mode.
This tests that FAISSx truly works as a drop-in replacement for FAISS.
"""

import unittest
import numpy as np

# Import FAISSx client as faiss - this should use local mode by default
from faissx import client as faiss


class TestIndexHNSWLocalMode(unittest.TestCase):
    """Test HNSW indices in local mode"""

    def setUp(self):
        """Set up test vectors and dimensions"""
        self.d = 32  # dimensions
        self.nb = 1000  # database size (smaller for HNSW as it's memory-intensive)
        self.nq = 10  # nb of queries

        # Generate random test data
        np.random.seed(42)  # for reproducibility
        self.xb = np.random.random((self.nb, self.d)).astype('float32')
        self.xq = np.random.random((self.nq, self.d)).astype('float32')

    def test_hnsw_flat(self):
        """Test IndexHNSWFlat functionality"""
        # HNSW parameters
        M = 16  # number of neighbors for each node (default: 32)
        efConstruction = 40  # construction time/accuracy trade-off (default: 40)

        # Create index
        index = faiss.IndexHNSWFlat(self.d, M)

        # Check properties
        self.assertEqual(index.d, self.d)
        self.assertEqual(index.hnsw.M, M)
        self.assertEqual(index.ntotal, 0)

        # Set efConstruction parameter
        index.hnsw.efConstruction = efConstruction
        self.assertEqual(index.hnsw.efConstruction, efConstruction)

        # Add vectors
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        # Set search parameters
        efSearch = 64  # accuracy/speed trade-off for search
        index.hnsw.efSearch = efSearch
        self.assertEqual(index.hnsw.efSearch, efSearch)

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

    def test_hnsw_incremental(self):
        """Test incremental additions to HNSW index"""
        index = faiss.IndexHNSWFlat(self.d, 16)

        # Add vectors in batches
        batch_size = 200
        for i in range(0, self.nb, batch_size):
            end = min(i + batch_size, self.nb)
            batch = self.xb[i:end]
            index.add(batch)

        # Verify total count
        self.assertEqual(index.ntotal, self.nb)

        # Search
        k = 5
        distances, indices = index.search(self.xq, k)

        # Verify shapes
        self.assertEqual(distances.shape, (self.nq, k))
        self.assertEqual(indices.shape, (self.nq, k))


if __name__ == "__main__":
    unittest.main()

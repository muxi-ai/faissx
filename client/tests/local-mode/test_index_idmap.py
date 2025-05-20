#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for IDMap indices (IndexIDMap, IndexIDMap2) in local mode.
This tests that FAISSx truly works as a drop-in replacement for FAISS.
"""

import unittest
import numpy as np

# Import FAISSx client as faiss - this should use local mode by default
from faissx import client as faiss


class TestIndexIDMapLocalMode(unittest.TestCase):
    """Test IDMap indices in local mode"""

    def setUp(self):
        """Set up test vectors and dimensions"""
        self.d = 64  # dimensions
        self.nb = 1000  # database size
        self.nq = 10  # nb of queries

        # Generate random test data
        np.random.seed(42)  # for reproducibility
        self.xb = np.random.random((self.nb, self.d)).astype('float32')
        self.xq = np.random.random((self.nq, self.d)).astype('float32')

        # Generate custom IDs
        self.ids = np.arange(1000, 1000 + self.nb).astype('int64')

        # Create a base index for the IDMap
        self.base_index = faiss.IndexFlatL2(self.d)

    def test_index_id_map(self):
        """Test IndexIDMap functionality"""
        # Create IndexIDMap
        index = faiss.IndexIDMap(self.base_index)

        # Check properties
        self.assertEqual(index.d, self.d)
        self.assertEqual(index.ntotal, 0)

        # Add vectors with IDs
        index.add_with_ids(self.xb, self.ids)
        self.assertEqual(index.ntotal, self.nb)

        # Search
        k = 5
        distances, indices = index.search(self.xq, k)

        # Verify shapes
        self.assertEqual(distances.shape, (self.nq, k))
        self.assertEqual(indices.shape, (self.nq, k))

        # Verify indices are our custom IDs (should be >= 1000)
        valid_results = indices != -1
        self.assertTrue(np.all(indices[valid_results] >= 1000))

        # Test remove_ids
        # Remove first 10 IDs
        remove_ids = self.ids[:10]
        index.remove_ids(remove_ids)

        # Verify count reduced
        self.assertEqual(index.ntotal, self.nb - 10)

        # Check that removed IDs are not returned in search
        distances, indices = index.search(self.xb[:1], k)
        valid_results = indices != -1
        for id in remove_ids:
            self.assertTrue(id not in indices[valid_results])

    def test_index_id_map2(self):
        """Test IndexIDMap2 functionality (supports updates)"""
        # Create IndexIDMap2
        index = faiss.IndexIDMap2(self.base_index)

        # Check properties
        self.assertEqual(index.d, self.d)
        self.assertEqual(index.ntotal, 0)

        # Add vectors with IDs
        index.add_with_ids(self.xb, self.ids)
        self.assertEqual(index.ntotal, self.nb)

        # Update some vectors (replace first 10 with modified versions)
        update_vectors = self.xb[:10].copy()
        update_vectors += 0.1  # slightly modify the vectors
        update_ids = self.ids[:10]

        index.add_with_ids(update_vectors, update_ids)

        # Total should still be the same (we updated, not added)
        self.assertEqual(index.ntotal, self.nb)

        # Test reconstruction
        for i, id in enumerate(update_ids):
            reconstructed = index.reconstruct(int(id))
            # Verify the updated vector was stored (should be close to our modified version)
            self.assertTrue(np.allclose(reconstructed, update_vectors[i], atol=1e-5))

    def test_reconstruct(self):
        """Test reconstruction functionality"""
        index = faiss.IndexIDMap(self.base_index)
        index.add_with_ids(self.xb, self.ids)

        # Test reconstruct
        for i in range(10):
            id = int(self.ids[i])
            reconstructed = index.reconstruct(id)
            # Verify reconstructed vector matches original
            self.assertTrue(np.allclose(reconstructed, self.xb[i], atol=1e-5))

        # Test reconstruct_n
        n = 5
        reconstructed = index.reconstruct_n(0, n)
        # Verify reconstructed vectors match originals
        for i in range(n):
            self.assertTrue(np.allclose(reconstructed[i], self.xb[i], atol=1e-5))


if __name__ == "__main__":
    unittest.main()

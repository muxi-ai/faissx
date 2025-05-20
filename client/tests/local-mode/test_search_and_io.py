#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for range search and IO operations in local mode.
This tests that FAISSx truly works as a drop-in replacement for FAISS.
"""

import unittest
import numpy as np
import tempfile
import os

# Import FAISSx client as faiss - this should use local mode by default
from faissx import client as faiss


class TestSearchAndIOLocalMode(unittest.TestCase):
    """Test special search operations and I/O functionality in local mode"""

    def setUp(self):
        """Set up test vectors and dimensions"""
        self.d = 32  # dimensions
        self.nb = 1000  # database size
        self.nq = 10  # nb of queries

        # Generate random test data
        np.random.seed(42)  # for reproducibility
        self.xb = np.random.random((self.nb, self.d)).astype('float32')
        self.xq = np.random.random((self.nq, self.d)).astype('float32')

        # Create a temporary directory for index files
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        for file in os.listdir(self.temp_dir):
            os.unlink(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_range_search(self):
        """Test range search functionality"""
        # Create index
        index = faiss.IndexFlatL2(self.d)
        index.add(self.xb)

        # Perform range search
        radius = 1.0  # search radius
        lims, distances, indices = index.range_search(self.xq, radius)

        # Verify shapes and properties
        self.assertEqual(len(lims), self.nq + 1)  # n+1 limits for n queries
        self.assertTrue(np.all(lims[1:] >= lims[:-1]))  # limits are non-decreasing
        self.assertEqual(len(distances), lims[-1])  # distances contains all distances
        self.assertEqual(len(indices), lims[-1])  # indices contains all indices

        # Verify distances are within radius
        self.assertTrue(np.all(distances <= radius))

        # Verify indices are valid
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < self.nb))

    def test_io_flat(self):
        """Test I/O operations with IndexFlatL2"""
        # Create and populate index
        index = faiss.IndexFlatL2(self.d)
        index.add(self.xb)

        # Define file path
        filepath = os.path.join(self.temp_dir, "flat_index.bin")

        # Write index to file
        faiss.write_index(index, filepath)

        # Verify file was created
        self.assertTrue(os.path.exists(filepath))

        # Read index from file
        index2 = faiss.read_index(filepath)

        # Verify properties
        self.assertEqual(index2.d, self.d)
        self.assertEqual(index2.ntotal, self.nb)

        # Verify search results are the same
        k = 5
        D1, I1 = index.search(self.xq, k)
        D2, I2 = index2.search(self.xq, k)

        # Results should match exactly for exact indices like FlatL2
        self.assertTrue(np.array_equal(D1, D2))
        self.assertTrue(np.array_equal(I1, I2))

    def test_io_ivf(self):
        """Test I/O operations with IVF indices"""
        # Create and train an IVF index
        quantizer = faiss.IndexFlatL2(self.d)
        index = faiss.IndexIVFFlat(quantizer, self.d, 50)

        index.train(self.xb)
        index.add(self.xb)

        # Define file path
        filepath = os.path.join(self.temp_dir, "ivf_index.bin")

        # Write index to file
        faiss.write_index(index, filepath)

        # Read index from file
        index2 = faiss.read_index(filepath)

        # Verify properties
        self.assertEqual(index2.d, self.d)
        self.assertEqual(index2.ntotal, self.nb)
        self.assertEqual(index2.nlist, 50)
        self.assertTrue(index2.is_trained)

    def test_factory(self):
        """Test index_factory functionality"""
        # Create various indices using the factory pattern
        factory_strings = [
            "Flat",
            "IVF100,Flat",
            # Skip complex types that aren't implemented yet
            # "PCAR32,IVF100,SQ8",
            "HNSW32",
            "IVF100,PQ8"  # Use IVF100,PQ8 instead of OPQ16_64,IVF100,PQ8
        ]

        for factory_str in factory_strings:
            # Create index using factory
            index = faiss.index_factory(self.d, factory_str)

            # Verify dimension
            self.assertEqual(index.d, self.d)

            # If index requires training, train it
            if not index.is_trained:
                index.train(self.xb)

            # Add vectors
            index.add(self.xb)

            # Verify count
            self.assertEqual(index.ntotal, self.nb)

            # Verify search works
            k = 5
            distances, indices = index.search(self.xq, k)

            # Verify shapes
            self.assertEqual(distances.shape, (self.nq, k))
            self.assertEqual(indices.shape, (self.nq, k))


if __name__ == "__main__":
    unittest.main()

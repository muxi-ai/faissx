#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test that IndexIVFFlat never falls back to local mode when remote mode is configured.

This script verifies that the IndexIVFFlat implementation properly operates in remote
mode with no fallbacks to local mode, raising appropriate errors for unsupported operations
rather than silently falling back.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from force_remote import force_remote_mode

# Add parent directory to path to import faissx
parent_dir = os.path.abspath(os.path.join(os.path.dirname(current_dir), "../../"))
sys.path.insert(0, parent_dir)

# Import after setting up path
from faissx import client as faiss


class TestIVFNoFallback(unittest.TestCase):
    """Test that IndexIVFFlat never falls back to local mode."""

    @classmethod
    def setUpClass(cls):
        """Apply forced remote mode for all tests."""
        # Force remote mode with no fallbacks
        force_remote_mode(server_url="tcp://localhost:45679")

        # Set fixed seed for reproducible results
        np.random.seed(42)

    def setUp(self):
        """Prepare test data."""
        self.dimension = 32
        self.nlist = 4
        self.num_vectors = 50
        self.vectors = np.random.random((self.num_vectors, self.dimension)).astype('float32')
        self.query = np.random.random((1, self.dimension)).astype('float32')
        self.training_vectors = np.random.random((100, self.dimension)).astype('float32')

    def test_01_create_index(self):
        """Test index creation with forced remote mode."""
        # Should work in remote mode with no fallback
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)

        # Verify index has remote setup
        self.assertFalse(hasattr(index, "_local_index") and index._local_index is not None)
        self.assertTrue(hasattr(index, "client"))
        self.assertTrue(hasattr(index, "index_id"))

    def test_02_train_index(self):
        """Test index training in remote mode."""
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)

        # Training should work in remote mode
        index.train(self.training_vectors)
        self.assertTrue(index.is_trained)

    def test_03_add_vectors(self):
        """Test adding vectors in remote mode."""
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        index.train(self.training_vectors)

        # Adding vectors should work in remote mode
        index.add(self.vectors)
        self.assertEqual(index.ntotal, self.num_vectors)

    def test_04_search(self):
        """Test search with nprobe parameter in remote mode."""
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        index.train(self.training_vectors)
        index.add(self.vectors)

        # Set different nprobe values
        index.nprobe = 1
        distances1, indices1 = index.search(self.query, k=5)

        index.nprobe = self.nlist
        distances2, indices2 = index.search(self.query, k=5)

        # Results should be different with different nprobe values
        # At least one result should differ with different nprobe settings
        self.assertFalse(np.allclose(distances1, distances2))

    def test_05_error_handling(self):
        """Test proper error handling for invalid operations."""
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)

        # Adding before training should raise error
        with self.assertRaises(RuntimeError):
            index.add(self.vectors)

        # Invalid dimension should raise ValueError, not fallback
        wrong_dim = np.random.random((5, self.dimension+1)).astype('float32')
        with self.assertRaises(ValueError):
            index.add(wrong_dim)

    def test_06_reset(self):
        """Test reset operation in remote mode."""
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        index.train(self.training_vectors)
        index.add(self.vectors)

        # Reset should work in remote mode
        self.assertEqual(index.ntotal, self.num_vectors)
        index.reset()
        self.assertEqual(index.ntotal, 0)
        self.assertTrue(index.is_trained)


if __name__ == "__main__":
    unittest.main()

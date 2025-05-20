#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test IndexIDMap remote mode functionality.

These tests verify that IndexIDMap and IndexIDMap2 work correctly in remote mode,
including vector addition with IDs, search, vector removal, and reconstruction.
"""

import unittest
import numpy as np
import os
import subprocess
import time
import sys
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after modifying the system path
from faissx import client as faiss


class TestIndexIDMapRemote(unittest.TestCase):
    """Test suite for IndexIDMap remote functionality."""

    server_process = None

    @classmethod
    def setUpClass(cls):
        """Start a test server on a different port to avoid conflicts."""
        # Use a different port to avoid conflicts with running servers
        test_port = 45679

        # Define environment variables for the server
        env = os.environ.copy()
        env["FAISSX_PORT"] = str(test_port)

        # Start server as a subprocess
        cls.server_process = subprocess.Popen(
            [sys.executable, "-m", "faissx.server.cli", "run", "--port", str(test_port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )

        # Wait for server to start
        time.sleep(1)

        # Configure client to use our test server
        faiss.configure(server=f"tcp://localhost:{test_port}")

    @classmethod
    def tearDownClass(cls):
        """Shut down the test server."""
        if cls.server_process:
            if os.name != 'nt':
                # On Unix, we can kill the whole process group
                os.killpg(os.getpgid(cls.server_process.pid), signal.SIGTERM)
            else:
                # On Windows
                cls.server_process.terminate()
            cls.server_process.wait()

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a dimension and test vectors
        self.dimension = 32
        self.num_vectors = 50

        # Create random test data
        self.vectors = np.random.random((self.num_vectors, self.dimension)).astype('float32')
        self.ids = np.array([i * 2 + 100 for i in range(self.num_vectors)], dtype=np.int64)

        # Create a query vector
        self.query = np.random.random((1, self.dimension)).astype('float32')

    def test_01_create_index(self):
        """Test creating an IndexIDMap in remote mode."""
        # Create base index
        base_index = faiss.IndexFlatL2(self.dimension)

        # Create IDMap wrapper
        idmap = faiss.IndexIDMap(base_index)

        # Verify basic properties
        self.assertEqual(idmap.d, self.dimension)
        self.assertEqual(idmap.ntotal, 0)
        self.assertTrue(hasattr(idmap, 'index_id'))

    def test_02_add_with_ids(self):
        """Test adding vectors with IDs in remote mode."""
        # Create indices
        base_index = faiss.IndexFlatL2(self.dimension)
        idmap = faiss.IndexIDMap(base_index)

        # Add vectors with IDs
        idmap.add_with_ids(self.vectors, self.ids)

        # Verify count
        self.assertEqual(idmap.ntotal, self.num_vectors)

    def test_03_search(self):
        """Test search functionality with IndexIDMap in remote mode."""
        # Create and populate index
        base_index = faiss.IndexFlatL2(self.dimension)
        idmap = faiss.IndexIDMap(base_index)
        idmap.add_with_ids(self.vectors, self.ids)

        # Search for vectors
        k = 5
        distances, indices = idmap.search(self.query, k)

        # Verify result shapes
        self.assertEqual(distances.shape, (1, k))
        self.assertEqual(indices.shape, (1, k))

        # Verify returned IDs are in our original ID list
        for idx in indices[0]:
            if idx != -1:  # Skip "not found" indicators
                self.assertIn(idx, self.ids)

    def test_04_remove_ids(self):
        """Test removing vectors by ID in remote mode."""
        # Create and populate index
        base_index = faiss.IndexFlatL2(self.dimension)
        idmap = faiss.IndexIDMap(base_index)
        idmap.add_with_ids(self.vectors, self.ids)

        # Initial count
        initial_count = idmap.ntotal

        # Remove first 10 IDs
        ids_to_remove = self.ids[:10]
        idmap.remove_ids(ids_to_remove)

        # Verify count was reduced
        self.assertEqual(idmap.ntotal, initial_count - 10)

        # Search and verify removed IDs are not found
        _, indices = idmap.search(self.vectors[:1], 50)
        for id_val in ids_to_remove:
            self.assertNotIn(id_val, indices[0])

    def test_05_reconstruct(self):
        """Test vector reconstruction in remote mode."""
        # Create and populate index
        base_index = faiss.IndexFlatL2(self.dimension)
        idmap = faiss.IndexIDMap(base_index)
        idmap.add_with_ids(self.vectors, self.ids)

        # Try to reconstruct a vector
        id_to_reconstruct = self.ids[5]
        original_vector = self.vectors[5]

        try:
            reconstructed = idmap.reconstruct(id_to_reconstruct)

            # Verify the vector shape
            self.assertEqual(reconstructed.shape, (self.dimension,))

            # Verify the vector content is close to original
            # Due to possible quantization/serialization effects, use approximate match
            np.testing.assert_allclose(
                reconstructed, original_vector, rtol=1e-5, atol=1e-5
            )
        except Exception as e:
            # If reconstruction is not supported by the server, this is acceptable
            if "not supported" in str(e).lower():
                self.skipTest("Reconstruction not supported by server")
            else:
                raise

    def test_06_idmap2(self):
        """Test IndexIDMap2 functionality in remote mode."""
        # Create base index
        base_index = faiss.IndexFlatL2(self.dimension)

        # Create IDMap2 wrapper
        idmap2 = faiss.IndexIDMap2(base_index)

        # Add vectors
        idmap2.add_with_ids(self.vectors, self.ids)
        self.assertEqual(idmap2.ntotal, self.num_vectors)

        # Test vector replacement (if supported)
        try:
            # Create a new vector to replace an existing one
            id_to_update = self.ids[10]
            new_vector = np.random.random((1, self.dimension)).astype('float32')

            # Replace the vector
            idmap2.replace_vector(id_to_update, new_vector)

            # Reconstruct and verify (if supported)
            try:
                reconstructed = idmap2.reconstruct(id_to_update)
                np.testing.assert_allclose(
                    reconstructed, new_vector.reshape(-1), rtol=1e-5, atol=1e-5
                )
            except Exception as e:
                if "not supported" in str(e).lower():
                    self.skipTest("Reconstruction not supported for IDMap2")
                else:
                    raise
        except Exception as e:
            if "not supported" in str(e).lower():
                self.skipTest("Vector replacement not supported for IDMap2")
            else:
                raise


if __name__ == "__main__":
    unittest.main()

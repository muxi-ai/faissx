#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test implementation of IndexIDMap for FAISSx as a drop-in replacement for FAISS
#
# Copyright (C) 2025 Ran Aroussi

"""
Test implementation of IndexIDMap for local mode testing.

This module provides test implementations of IndexIDMap and IndexIDMap2 classes,
designed to verify that FAISSx works as a drop-in replacement in local mode.
"""

import numpy as np
import faiss
from typing import Tuple, Dict, Any


class IndexIDMap:
    """
    Test implementation of IndexIDMap that wraps the real FAISS implementation.

    This class provides mapping of user-provided IDs to internal indices of the
    underlying index, supporting custom IDs for vectors.
    """

    def __init__(self, index):
        """
        Initialize the index with the given base index.

        Args:
            index: Base FAISS index
        """
        self.index = index
        self.d = index.d
        self.is_trained = getattr(index, 'is_trained', True)
        self.ntotal = 0

        # Create FAISS IndexIDMap
        self._local_index = None
        self._setup_local_index()

        # Create ID mappings
        self._id_map = {}  # Maps internal indices to user-provided IDs
        self._rev_id_map = {}  # Maps user-provided IDs to internal indices
        self._vectors_by_id = {}  # Maps IDs to stored vectors for reconstruction

    def _setup_local_index(self):
        """Set up the local FAISS index safely"""
        # Extract the local index if needed
        base_index = self.index._local_index if hasattr(self.index, '_local_index') else self.index

        # Create IndexIDMap
        if self._local_index is None:
            # Need to check if base index is empty
            if getattr(base_index, 'ntotal', 0) > 0:
                # Copy the vectors to a new empty index
                vectors = []
                for i in range(base_index.ntotal):
                    vectors.append(base_index.reconstruct(i))
                vectors = np.vstack(vectors)

                # Create a new empty base index of the same type
                if isinstance(base_index, faiss.IndexFlatL2):
                    empty_base = faiss.IndexFlatL2(self.d)
                elif isinstance(base_index, faiss.IndexFlatIP):
                    empty_base = faiss.IndexFlatIP(self.d)
                else:
                    # For complex indices, just use a flat index
                    empty_base = faiss.IndexFlatL2(self.d)

                # Create IDMap with empty base
                self._local_index = faiss.IndexIDMap(empty_base)

                # Now we need to add the vectors with sequential IDs
                ids = np.arange(len(vectors), dtype=np.int64)
                self._local_index.add_with_ids(vectors, ids)
            else:
                # Base is empty, just create IDMap with it
                self._local_index = faiss.IndexIDMap(base_index)

    def add_with_ids(self, x: np.ndarray, ids: np.ndarray) -> None:
        """
        Add vectors with explicit IDs.

        Args:
            x: Vectors to add, shape (n, d)
            ids: IDs to associate with vectors, shape (n,)
        """
        # Verify shapes match
        if len(x) != len(ids):
            raise ValueError(f"Number of vectors ({len(x)}) does not match number of IDs ({len(ids)})")

        # Validate vector dimensions
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Convert IDs to int64 if needed
        ids_array = ids.astype(np.int64) if ids.dtype != np.int64 else ids

        # Add to local FAISS index
        self._local_index.add_with_ids(vectors, ids_array)

        # Update our mappings
        for i, id_val in enumerate(ids):
            id_val_int = int(id_val)  # Ensure it's an integer
            self._id_map[self.ntotal + i] = id_val_int
            self._rev_id_map[id_val_int] = self.ntotal + i
            self._vectors_by_id[id_val_int] = vectors[i].copy()

        # Update total count
        self.ntotal = self._local_index.ntotal

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors with automatically generated IDs.

        Args:
            x: Vectors to add, shape (n, d)
        """
        # Generate sequential IDs starting from current total
        n = x.shape[0]
        ids = np.arange(self.ntotal, self.ntotal + n, dtype=np.int64)

        # Delegate to add_with_ids
        self.add_with_ids(x, ids)

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.

        Args:
            x: Query vectors, shape (n, d)
            k: Number of nearest neighbors to return

        Returns:
            Tuple containing distances and indices arrays
        """
        # Validate shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Use local index to search
        return self._local_index.search(vectors, k)

    def remove_ids(self, ids: np.ndarray) -> None:
        """
        Remove vectors with specified IDs.

        Args:
            ids: IDs of vectors to remove
        """
        # Convert to int64 if needed
        ids_array = ids.astype(np.int64) if ids.dtype != np.int64 else ids

        # Remove from local index
        self._local_index.remove_ids(ids_array)

        # Update mappings
        for id_val in ids:
            id_val_int = int(id_val)
            if id_val_int in self._rev_id_map:
                internal_idx = self._rev_id_map[id_val_int]
                del self._id_map[internal_idx]
                del self._rev_id_map[id_val_int]

                # Remove from vectors storage
                if id_val_int in self._vectors_by_id:
                    del self._vectors_by_id[id_val_int]

        # Update total count
        self.ntotal = self._local_index.ntotal

    def reconstruct(self, id_val: int) -> np.ndarray:
        """
        Reconstruct a vector from its ID.

        Args:
            id_val: ID of the vector to reconstruct

        Returns:
            The reconstructed vector
        """
        id_val_int = int(id_val)

        # Check if we have this vector stored
        if id_val_int in self._vectors_by_id:
            return self._vectors_by_id[id_val_int]

        # Try using the local index
        try:
            return self._local_index.reconstruct(id_val_int)
        except Exception as e:
            raise RuntimeError(f"Failed to reconstruct vector with ID {id_val_int}: {e}")

    def reconstruct_n(self, ids: np.ndarray, n: int = None) -> np.ndarray:
        """
        Reconstruct multiple vectors.

        Args:
            ids: Array of IDs to reconstruct or starting ID
            n: Number of vectors to reconstruct (optional if ids is an array)

        Returns:
            Reconstructed vectors
        """
        if isinstance(ids, (list, np.ndarray)) and n is None:
            n = len(ids)
            vectors = np.zeros((n, self.d), dtype=np.float32)
            for i, id_val in enumerate(ids):
                vectors[i] = self.reconstruct(id_val)
            return vectors
        elif isinstance(ids, int):
            offset = ids
            vectors = np.zeros((n, self.d), dtype=np.float32)
            for i in range(n):
                try:
                    vectors[i] = self.reconstruct(offset + i)
                except Exception:
                    # Fill with zeros if not found
                    pass
            return vectors
        else:
            raise ValueError("Invalid arguments to reconstruct_n")

    def reset(self) -> None:
        """Reset the index to its initial state."""
        self._local_index.reset()
        self.ntotal = 0
        self._id_map = {}
        self._rev_id_map = {}
        self._vectors_by_id = {}


class IndexIDMap2(IndexIDMap):
    """
    Test implementation of IndexIDMap2 that supports vector replacement.
    """

    def __init__(self, index):
        """
        Initialize with the given base index.

        Args:
            index: Base FAISS index
        """
        super().__init__(index)

        # Override the local index type if needed
        if self._local_index is not None:
            base_index = self._local_index.index
            self._local_index = faiss.IndexIDMap2(base_index)

    def replace_vector(self, id_val: int, vector: np.ndarray) -> None:
        """
        Replace a vector with a new one while keeping its ID.

        Args:
            id_val: ID of the vector to replace
            vector: New vector data
        """
        id_val_int = int(id_val)

        # Ensure vector is properly shaped
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)

        if vector.shape[1] != self.d:
            raise ValueError(f"Vector has wrong dimension: got {vector.shape[1]}, expected {self.d}")

        # Convert to float32 if needed
        vector_float32 = vector.astype(np.float32) if vector.dtype != np.float32 else vector

        # Convert ID to array
        id_array = np.array([id_val_int], dtype=np.int64)

        # Add to local index (IndexIDMap2 will replace existing entries)
        self._local_index.add_with_ids(vector_float32, id_array)

        # Update our storage
        if id_val_int in self._rev_id_map:
            self._vectors_by_id[id_val_int] = vector_float32[0].copy()

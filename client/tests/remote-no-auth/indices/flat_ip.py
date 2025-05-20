#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test implementation of IndexFlatIP for FAISSx as a drop-in replacement for FAISS
#
# Copyright (C) 2025 Ran Aroussi

"""
Test implementation of IndexFlatIP for local mode testing.

This module provides a test implementation of the IndexFlatIP class for FAISS,
designed to verify that FAISSx works as a drop-in replacement in local mode.
"""

import numpy as np
import faiss
from typing import Tuple


class IndexFlatIP:
    """
    Test implementation of IndexFlatIP that wraps the real FAISS implementation.

    This class provides a flat index with inner product similarity metric.
    """

    def __init__(self, d: int):
        """
        Initialize the index with the given dimension.

        Args:
            d (int): Vector dimension
        """
        self.d = d  # Store dimension

        # Create the actual FAISS index using inner product metric
        self._local_index = faiss.IndexFlatIP(d)

        # Initialize state
        self.ntotal = 0

        # Set the index ID for tracking
        self.name = f"test-flat-ip-index-{id(self)}"
        self.index_id = self.name

        # Always trained (no training needed for flat index)
        self.is_trained = True

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Add vectors to the underlying index
        self._local_index.add(vectors)

        # Update total count
        self.ntotal = self._local_index.ntotal

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.

        Args:
            x (np.ndarray): Query vectors, shape (n, d)
            k (int): Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices arrays
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Perform search using the underlying index
        return self._local_index.search(vectors, k)

    def reconstruct(self, idx: int) -> np.ndarray:
        """
        Reconstruct a vector from its index.

        Args:
            idx (int): The index of the vector to reconstruct

        Returns:
            np.ndarray: The reconstructed vector
        """
        if idx < 0 or idx >= self.ntotal:
            raise RuntimeError(f"Index {idx} out of bounds [0, {self.ntotal})")

        return self._local_index.reconstruct(idx)

    def reconstruct_n(self, idx: int, n: int) -> np.ndarray:
        """
        Reconstruct multiple vectors starting from the given index.

        Args:
            idx (int): The starting index
            n (int): The number of vectors to reconstruct

        Returns:
            np.ndarray: The reconstructed vectors
        """
        if idx < 0 or idx + n > self.ntotal:
            raise RuntimeError(f"Range [{idx}, {idx+n}) out of bounds [0, {self.ntotal})")

        return self._local_index.reconstruct_n(idx, n)

    def reset(self) -> None:
        """Reset the index, removing all vectors."""
        self._local_index.reset()
        self.ntotal = 0

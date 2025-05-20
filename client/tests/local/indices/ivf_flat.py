#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test implementation of IndexIVFFlat for FAISSx as a drop-in replacement for FAISS
#
# Copyright (C) 2025 Ran Aroussi

"""
Test implementation of IndexIVFFlat for local mode testing.

This module provides a test implementation of the IndexIVFFlat class for FAISS,
designed to verify that FAISSx works as a drop-in replacement in local mode.
"""

import numpy as np
import faiss
from typing import Tuple


class IndexIVFFlat:
    """
    Test implementation of IndexIVFFlat that wraps the real FAISS implementation.

    This class provides inverted file indexing for efficient similarity search.
    It divides the vector space into partitions (clusters) for faster search,
    requiring a training step before use.
    """

    def __init__(self, quantizer, d, nlist, metric_type=faiss.METRIC_L2):
        """
        Initialize the inverted file index with specified parameters.

        Args:
            quantizer: Quantizer object that defines the centroids (usually IndexFlatL2)
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions
            metric_type: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        self.d = d
        self.nlist = nlist

        # Get the underlying FAISS index from the quantizer if it has one
        quantizer_index = quantizer._local_index if hasattr(quantizer, '_local_index') else quantizer

        # Create the actual FAISS index
        self._local_index = faiss.IndexIVFFlat(quantizer_index, d, nlist, metric_type)

        # Initialize state
        self.is_trained = self._local_index.is_trained
        self.ntotal = self._local_index.ntotal
        self._nprobe = 1  # Default number of probes

        # Set the index ID for tracking
        self.name = f"test-ivf-flat-index-{id(self)}"
        self.index_id = self.name

    @property
    def nprobe(self):
        """Get the current nprobe value"""
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value):
        """Set the nprobe value and update the local index"""
        self._nprobe = value
        self._local_index.nprobe = value

    def train(self, x: np.ndarray) -> None:
        """
        Train the index with the provided vectors.

        Args:
            x (np.ndarray): Training vectors, shape (n, d)
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Train the underlying index
        self._local_index.train(vectors)

        # Update trained state
        self.is_trained = self._local_index.is_trained

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Ensure index is trained
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

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

        # Ensure index is trained
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Set nprobe before searching
        self._local_index.nprobe = self._nprobe

        # Perform search using the underlying index
        return self._local_index.search(vectors, k)

    def reset(self) -> None:
        """Reset the index, removing all vectors but keeping training."""
        # Remember the trained state
        was_trained = self.is_trained

        # Reset the index
        self._local_index.reset()
        self.ntotal = 0

        # Restore the trained state
        self.is_trained = was_trained

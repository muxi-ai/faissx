# Copyright 2024 Muxi, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base class for binary indices in FAISSx.

This module provides the common base class for all binary indices which use
Hamming distance for similarity calculations on binary vectors.
"""

import uuid
import numpy as np
from typing import Dict, Optional, Tuple, Any

from faissx.client.indices.base import Index

# Lookup table for bit counting (popcount)
# This is used for efficient Hamming distance calculation
BIT_COUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class BinaryIndex(Index):
    """
    Base class for binary indices using Hamming distance.

    This class contains common functionality for binary vector indices,
    which store data as uint8 arrays and use Hamming distance for similarity.
    """

    def __init__(self, d: int):
        """
        Initialize the binary index.

        Args:
            d (int): Dimension of binary vectors in bytes.
                    (Each byte contains 8 bits, so the actual dimension in bits is 8*d)
        """
        super().__init__()  # Initialize base class

        self.d: int = d  # Dimension in bytes
        self.code_size: int = d  # Size of each code in bytes
        self.is_trained: bool = True  # Binary indices don't require training
        self.ntotal: int = 0  # Number of vectors in the index

        # Generate a unique index ID
        self.index_id = str(uuid.uuid4())

        # Storage for vectors (only used in local mode)
        self._vectors: Optional[np.ndarray] = None

        # Vector mapping for tracking indices
        self._vector_mapping: Dict[int, Dict[str, Any]] = {}

    def validate_binary_vectors(self, vectors: np.ndarray, operation: str = "add") -> None:
        """
        Validate that vectors are in the correct binary format.

        Args:
            vectors (np.ndarray): Binary vectors to validate
            operation (str): Name of the operation (for error messages)

        Raises:
            TypeError: If vectors are not uint8
            ValueError: If vector dimensions don't match the index
        """
        if vectors.dtype != np.uint8:
            raise TypeError(f"Binary indices require uint8 vectors, got {vectors.dtype}")

        if vectors.shape[1] != self.d:
            raise ValueError(f"Vector dimension mismatch for {operation}: expected {self.d}, "
                             f"got {vectors.shape[1]}")

    def compute_hamming_distances(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute Hamming distances between query and vectors.

        Args:
            query (np.ndarray): Query vector(s) of shape (n_queries, d)
            vectors (np.ndarray): Database vectors of shape (n_vectors, d)

        Returns:
            np.ndarray: Hamming distances of shape (n_queries, n_vectors)
        """
        n_queries = query.shape[0]
        n_vectors = vectors.shape[0]
        distances = np.zeros((n_queries, n_vectors), dtype=np.float32)

        for i in range(n_queries):
            # XOR query with all vectors - this gives us bytes where set bits represent differences
            xor_result = np.bitwise_xor(vectors, query[i])

            # Count bits in each byte using the lookup table
            # This is faster than using np.unpackbits for large datasets
            hamming_distances = np.zeros(n_vectors, dtype=np.float32)
            for j in range(n_vectors):
                # Sum the bit counts for each byte in the XOR result
                hamming_distances[j] = np.sum(BIT_COUNT_LUT[xor_result[j]])

            distances[i] = hamming_distances

        return distances

    def compute_hamming_distances_optimized(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute Hamming distances between query and vectors using optimized vectorized operations.

        This is a more efficient implementation for large datasets.

        Args:
            query (np.ndarray): Query vector(s) of shape (n_queries, d)
            vectors (np.ndarray): Database vectors of shape (n_vectors, d)

        Returns:
            np.ndarray: Hamming distances of shape (n_queries, n_vectors)
        """
        n_queries = query.shape[0]
        n_vectors = vectors.shape[0]
        distances = np.zeros((n_queries, n_vectors), dtype=np.float32)

        for i in range(n_queries):
            # XOR query with all vectors
            xor_result = np.bitwise_xor(vectors, query[i])

            # Reshape and apply the lookup table to count bits in each byte
            reshaped = BIT_COUNT_LUT[xor_result.reshape(-1)].reshape(n_vectors, -1)
            hamming_distances = np.sum(reshaped, axis=1)

            distances[i] = hamming_distances

        return distances

    def get_top_k(self, distances: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the top k nearest neighbors from distance matrix.

        Args:
            distances (np.ndarray): Distance matrix of shape (n_queries, n_vectors)
            k (int): Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: Top k distances and indices
                - top_distances: shape (n_queries, k)
                - top_indices: shape (n_queries, k)
        """
        n_queries, n_vectors = distances.shape
        k = min(k, n_vectors)  # Ensure k is not larger than available vectors

        top_distances = np.zeros((n_queries, k), dtype=np.float32)
        top_indices = np.zeros((n_queries, k), dtype=np.int64)

        for i in range(n_queries):
            if k < n_vectors:
                # Use argpartition for large datasets (more efficient than argsort)
                top_k_idx = np.argpartition(distances[i], k)[:k]
                # Sort the top k results
                sorted_idx = top_k_idx[np.argsort(distances[i][top_k_idx])]
                top_distances[i] = distances[i][sorted_idx]
                top_indices[i] = sorted_idx
            else:
                # If k >= n_vectors, just sort everything
                sorted_idx = np.argsort(distances[i])
                top_distances[i] = distances[i][sorted_idx[:k]]
                top_indices[i] = sorted_idx[:k]

        return top_distances, top_indices

    def reset(self) -> None:
        """
        Reset the index, removing all vectors.
        """
        self.ntotal = 0
        self._vectors = None
        self._vector_mapping = {}

    def __getstate__(self) -> Dict[str, Any]:
        """
        Get the state for serialization.

        Returns:
            Dict[str, Any]: The index state
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Set the state during deserialization.

        Args:
            state (Dict[str, Any]): The index state
        """
        self.__dict__.update(state)

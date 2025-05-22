#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/faissx
#
# Copyright (C) 2025 Ran Aroussi
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
Binary Flat Index implementation for FAISSx.

This module provides a binary vector index using Hamming distance,
which is compatible with the FAISS IndexBinaryFlat.
"""

import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from faissx.client.indices.binary_base import BinaryIndex
from faissx.client.client import get_client


class IndexBinaryFlat(BinaryIndex):
    """
    IndexBinaryFlat is a binary vector index that uses Hamming distance.

    This index stores binary vectors as uint8 arrays and performs exact search
    using Hamming distance (count of different bits).

    It is compatible with the FAISS IndexBinaryFlat implementation.
    """

    def __init__(self, d: int):
        """
        Initialize the binary flat index.

        Args:
            d (int): Dimension of binary vectors in bytes.
                    (Each byte contains 8 bits, so the actual dimension in bits is 8*d)
        """
        super().__init__(d)  # Initialize BinaryIndex base class

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

    def add(self, vectors: np.ndarray) -> None:
        """
        Add binary vectors to the index.

        Args:
            vectors (np.ndarray): Binary vectors to add, should be uint8 with shape
                                  (n, d) where d is the dimension in bytes
        """
        # Validate the input vectors
        self.validate_binary_vectors(vectors)

        # Determine starting index for new vectors
        start_idx = self.ntotal
        n = vectors.shape[0]

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: send to server
            # Placeholder for server implementation - to be completed
            client.add_vectors(self.index_id, vectors)

            # Update vector mapping for remote mode
            for i in range(n):
                self._vector_mapping[start_idx + i] = {"server_id": start_idx + i}
        else:
            # Local mode: store vectors locally
            if self._vectors is None:
                self._vectors = vectors.copy()
            else:
                self._vectors = np.vstack((self._vectors, vectors))

            # Update vector mapping for local mode
            for i in range(n):
                self._vector_mapping[start_idx + i] = {"local_id": start_idx + i}

        # Update total count
        self.ntotal += n

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors using Hamming distance.

        Args:
            query (np.ndarray): Binary query vectors, uint8 with shape (n, d)
            k (int): Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of nearest neighbors
                - distances: Hamming distances with shape (n, k)
                - indices: Vector indices with shape (n, k)
        """
        # Validate the query vectors
        self.validate_binary_vectors(query, operation="search")

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: send to server
            # Placeholder for server implementation - to be completed
            distances, indices = client.search(self.index_id, query, k)
            return distances, indices
        else:
            # Local mode: compute Hamming distances
            if self.ntotal == 0 or self._vectors is None:
                # Return empty results if no vectors
                n = query.shape[0]
                return np.zeros((n, k), dtype=np.float32), np.zeros((n, k), dtype=np.int64)

            # Compute Hamming distances using the optimized method
            distances = self.compute_hamming_distances_optimized(query, self._vectors)

            # Get top k results
            return self.get_top_k(distances, k)

    def reconstruct(self, idx: int) -> np.ndarray:
        """
        Reconstruct the binary vector at the given index.

        Args:
            idx (int): Index of the vector to reconstruct

        Returns:
            np.ndarray: The reconstructed binary vector
        """
        if idx < 0 or idx >= self.ntotal:
            raise IndexError(f"Index {idx} out of bounds for index with {self.ntotal} vectors")

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: request from server
            # Placeholder for server implementation - to be completed
            return client.reconstruct(self.index_id, idx)
        else:
            # Local mode: return stored vector
            if self._vectors is None:
                raise RuntimeError("No vectors stored in local mode")

            mapping = self._vector_mapping.get(idx)
            if mapping and "local_id" in mapping:
                local_id = mapping["local_id"]
                return self._vectors[local_id]
            else:
                raise IndexError(f"Vector mapping not found for index {idx}")

    def reconstruct_n(self, indices: List[int]) -> np.ndarray:
        """
        Reconstruct multiple binary vectors by their indices.

        Args:
            indices (List[int]): Indices of vectors to reconstruct

        Returns:
            np.ndarray: The reconstructed binary vectors with shape (len(indices), d)
        """
        vectors = np.zeros((len(indices), self.d), dtype=np.uint8)
        for i, idx in enumerate(indices):
            vectors[i] = self.reconstruct(idx)
        return vectors

    def reset(self) -> None:
        """
        Reset the index, removing all vectors.
        """
        self.ntotal = 0
        self._vectors = None
        self._vector_mapping = {}

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: reset on server
            # Placeholder for server implementation - to be completed
            client.reset_index(self.index_id)

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

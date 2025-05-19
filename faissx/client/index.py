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
FAISS index classes implementation for ZeroMQ-based FAISSx

This module provides a client-side implementation of FAISS index operations
that communicate with a remote FAISSx service via ZeroMQ. It implements
the same interface as FAISS but delegates all operations to the remote service.
"""

import uuid
import numpy as np
from typing import Dict, Tuple, Any

from .client import get_client


class IndexFlatL2:
    """
    Proxy implementation of FAISS IndexFlatL2

    This class mimics the behavior of FAISS IndexFlatL2 but uses the
    remote FAISSx service for all operations via ZeroMQ. It maintains
    a mapping between local and server-side indices to ensure consistent
    indexing across operations.

    Attributes:
        d (int): Vector dimension
        is_trained (bool): Always True for L2 index
        ntotal (int): Total number of vectors in the index
        name (str): Unique identifier for the index
        index_id (str): Server-side index identifier
        _vector_mapping (dict): Maps local indices to server indices
        _next_idx (int): Next available local index
    """

    def __init__(self, d: int):
        """
        Initialize the index with specified dimension.

        Args:
            d (int): Vector dimension for the index
        """
        # Store dimension and initialize basic attributes
        self.d = d
        self.is_trained = True  # L2 index doesn't require training
        self.ntotal = 0  # Track total vectors

        # Generate unique name for the index
        self.name = f"index-flat-l2-{uuid.uuid4().hex[:8]}"

        # Initialize connection to remote service and create index
        self.client = get_client()
        self.index_id = self.client.create_index(
            name=self.name,
            dimension=self.d,
            index_type="L2"
        )

        # Initialize local tracking of vectors
        self._vector_mapping = {}  # Maps local indices to server-side information
        self._next_idx = 0  # Counter for local indices

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        This method adds vectors to both the remote index and maintains
        local mapping between client and server indices.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d) where n is number of vectors
                           and d is the dimension

        Raises:
            ValueError: If vector shape doesn't match index dimension
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Add vectors to remote index
        result = self.client.add_vectors(self.index_id, vectors)

        # Update local tracking if addition was successful
        if result.get("success", False):
            added_count = result.get("count", 0)
            # Create mapping for each added vector
            for i in range(added_count):
                self._vector_mapping[self._next_idx] = {
                    "local_idx": self._next_idx,
                    "server_idx": self.ntotal + i
                }
                self._next_idx += 1

            self.ntotal += added_count

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.

        This method performs a k-nearest neighbor search using L2 distance
        and maps the server-side indices back to local indices.

        Args:
            x (np.ndarray): Query vectors, shape (n, d)
            k (int): Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Distances array of shape (n, k)
                - Indices array of shape (n, k)

        Raises:
            ValueError: If query vector shape doesn't match index dimension
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Perform search on remote index
        result = self.client.search(
            self.index_id,
            query_vectors=query_vectors,
            k=k
        )

        n = x.shape[0]  # Number of query vectors
        search_results = result.get("results", [])

        # Initialize output arrays with default values
        distances = np.full((n, k), float('inf'), dtype=np.float32)
        idx = np.full((n, k), -1, dtype=np.int64)

        # Process results for each query vector
        for i in range(min(n, len(search_results))):
            result_data = search_results[i]
            result_distances = result_data.get("distances", [])
            result_indices = result_data.get("indices", [])

            # Fill in results for this query vector
            for j in range(min(k, len(result_distances))):
                distances[i, j] = result_distances[j]

                # Map server index back to local index
                server_idx = result_indices[j]
                found = False
                for local_idx, info in self._vector_mapping.items():
                    if info["server_idx"] == server_idx:
                        idx[i, j] = local_idx
                        found = True
                        break

                # Keep -1 if mapping not found
                if not found:
                    idx[i, j] = -1

        return distances, idx

    def reset(self) -> None:
        """
        Reset the index to its initial state.

        This method creates a new remote index and resets all local tracking.
        If the current index exists, it creates a new one with a modified name.
        """
        # Check if current index exists and create new one
        try:
            stats = self.client.get_index_stats(self.index_id)
            if stats.get("success", False):
                # Create new index with modified name
                new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"
                self.index_id = self.client.create_index(
                    name=new_name,
                    dimension=self.d,
                    index_type="L2"
                )
                self.name = new_name
        except Exception:  # Catch specific exceptions if possible
            # Recreate with same name if error occurs
            self.index_id = self.client.create_index(
                name=self.name,
                dimension=self.d,
                index_type="L2"
            )

        # Reset all local state
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.

        Note: Currently no explicit cleanup is performed as the server API
        doesn't provide an index deletion operation.
        """
        # No explicit delete index operation in the server API
        pass

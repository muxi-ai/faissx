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
FAISSx Index Implementation Module

This module provides client-side implementations of FAISS index classes that
communicate with a remote FAISSx service via ZeroMQ. Key features include:

- Drop-in replacements for FAISS index types (currently IndexFlatL2)
- Identical API signatures to the original FAISS implementations
- Transparent remote execution of add, search, and other vector operations
- Local-to-server index mapping to maintain consistent vector references
- Automatic conversion of data types and array formats for ZeroMQ transport
- Support for all standard FAISS index operations with server delegation

Each index class matches the behavior of its FAISS counterpart while sending
the actual computational work to the FAISSx server.
"""

import uuid
import numpy as np
from typing import Tuple

from .client import get_client


class IndexFlatL2:
    """
    Proxy implementation of FAISS IndexFlatL2

    This class mimics the behavior of FAISS IndexFlatL2. It uses the local FAISS
    implementation by default, but can use the remote FAISSx service when explicitly
    configured via configure(). It maintains a mapping between local and server-side
    indices to ensure consistent indexing across operations when using remote mode.

    Attributes:
        d (int): Vector dimension
        is_trained (bool): Always True for L2 index
        ntotal (int): Total number of vectors in the index
        name (str): Unique identifier for the index
        index_id (str): Server-side index identifier (when in remote mode)
        _vector_mapping (dict): Maps local indices to server indices (remote mode only)
        _next_idx (int): Next available local index (remote mode only)
        _local_index: Local FAISS index (local mode only)
        _using_remote (bool): Whether we're using remote or local implementation
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

        # Check if we should use remote implementation
        # (this depends on if configure() has been called)
        try:
            # Import here to avoid circular imports
            import faissx

            # Check if API key or server URL are set - this indicates configure() was called
            configured = bool(faissx._API_KEY) or (
                faissx._API_URL != "tcp://localhost:45678"
            )

            # If configure was explicitly called, use remote mode
            if configured:
                self._using_remote = True
                self.client = get_client()
                self._local_index = None

                # Create index on server
                self.index_id = self.client.create_index(
                    name=self.name,
                    dimension=self.d,
                    index_type="L2"
                )

                # Initialize local tracking of vectors for remote mode
                self._vector_mapping = {}  # Maps local indices to server-side information
                self._next_idx = 0  # Counter for local indices
                return

        except Exception as e:
            import logging
            logging.warning(f"Error initializing remote mode: {e}, falling back to local mode")

        # Use local FAISS implementation by default
        self._using_remote = False
        self._local_index = None

        # Import local FAISS here to avoid module-level dependency
        try:
            import faiss
            self._local_index = faiss.IndexFlatL2(d)
            self.index_id = self.name  # Use name as ID for consistency
        except ImportError as e:
            raise ImportError(f"Failed to import FAISS for local mode: {e}")

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

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

        if not self._using_remote:
            # Use local FAISS implementation directly
            self._local_index.add(vectors)
            self.ntotal = self._local_index.ntotal
            return

        # Add vectors to remote index (remote mode)
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

        if not self._using_remote:
            # Use local FAISS implementation directly
            return self._local_index.search(query_vectors, k)

        # Perform search on remote index (remote mode)
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
        """
        if not self._using_remote:
            # Reset local FAISS index
            self._local_index.reset()
            self.ntotal = 0
            return

        # Remote mode reset
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
        """
        # Local index will be cleaned up by garbage collector
        pass

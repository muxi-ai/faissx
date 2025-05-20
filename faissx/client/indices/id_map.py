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
FAISSx IndexIDMap implementation.

This module provides a client-side implementation of the FAISS IndexIDMap class.
It serves as a wrapper around another index type that maps external IDs to vectors.
"""

from .base import np, Tuple, faiss, logging, get_client


class IndexIDMap:
    """
    Proxy implementation of FAISS IndexIDMap.

    This class wraps another index and adds support for mapping between user-provided IDs
    and the internal indices used by the wrapped index. This enables operations using
    custom identifiers instead of sequential indices.

    When adding vectors, the caller can provide explicit IDs. When searching, the results
    will contain these user-provided IDs instead of internal indices.

    Attributes:
        index: The wrapped index object
        is_trained (bool): Whether the underlying index is trained
        ntotal (int): Total number of vectors in the index
        d (int): Vector dimension (same as wrapped index)
        _id_map (dict): Maps internal indices to user-provided IDs
        _rev_id_map (dict): Maps user-provided IDs to internal indices
    """

    def __init__(self, index):
        """
        Initialize the IndexIDMap with the given underlying index.

        Args:
            index: The FAISS index to wrap (e.g., IndexFlatL2, IndexIVFFlat)
        """
        # Check if we should use remote implementation
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
                raise NotImplementedError("Remote mode for IndexIDMap not implemented yet")
        except Exception as e:
            logging.warning(
                f"Error initializing remote mode: {e}, falling back to local mode"
            )
            self._using_remote = False

        # Store the underlying index and its properties
        self.index = index
        self.is_trained = getattr(index, 'is_trained', True)
        self.ntotal = 0  # Start with no vectors
        self.d = index.d  # Vector dimension from underlying index

        # Get local index from wrapped index if available
        base_index = index._local_index if hasattr(index, '_local_index') else index

        # Create FAISS IndexIDMap
        try:
            self._local_index = faiss.IndexIDMap(base_index)
        except Exception as e:
            raise RuntimeError(f"Failed to create IndexIDMap: {e}")

        # Initialize bidirectional mapping dictionaries
        self._id_map = {}  # Maps internal indices to user IDs
        self._rev_id_map = {}  # Maps user IDs to internal indices

        # Store vectors for reconstruction
        self._vectors_by_id = {}  # Maps IDs to stored vectors

    def add_with_ids(self, x: np.ndarray, ids: np.ndarray) -> None:
        """
        Add vectors with explicit IDs.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)
            ids (np.ndarray): IDs to associate with vectors, shape (n,)

        Raises:
            ValueError: If shapes don't match or if duplicate IDs are provided
        """
        # Validate input shapes match
        if len(x) != len(ids):
            raise ValueError(
                f"Number of vectors ({len(x)}) does not match number of IDs ({len(ids)})"
            )

        # Validate vector dimensions
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Convert IDs to int64 if needed
        ids_array = ids.astype(np.int64) if ids.dtype != np.int64 else ids

        # Add to local FAISS index
        self._local_index.add_with_ids(vectors, ids_array)

        # Update our tracking and store vectors
        for i, id_val in enumerate(ids):
            id_val_int = int(id_val)  # Convert to int to use as key
            self._id_map[self.ntotal + i] = id_val_int
            self._rev_id_map[id_val_int] = self.ntotal + i
            self._vectors_by_id[id_val_int] = vectors[i].copy()

        # Update total count
        self.ntotal = self._local_index.ntotal

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors with automatically generated IDs.

        This will use the vector's position in the index as its ID.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)
        """
        # Generate sequential IDs starting from current total
        n = x.shape[0]
        ids = np.arange(self.ntotal, self.ntotal + n, dtype=np.int64)

        # Delegate to add_with_ids with generated IDs
        self.add_with_ids(x, ids)

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.

        Args:
            x (np.ndarray): Query vectors, shape (n, d)
            k (int): Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Distances array of shape (n, k)
                - IDs array of shape (n, k) containing user-provided IDs

        Raises:
            ValueError: If query vector shape doesn't match index dimension
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Use local FAISS index to search
        distances, indices = self._local_index.search(vectors, k)

        return distances, indices

    def remove_ids(self, ids: np.ndarray) -> None:
        """
        Remove vectors with the specified IDs.

        Args:
            ids (np.ndarray): IDs of vectors to remove

        Raises:
            ValueError: If any ID is not found
        """
        # Convert to int64 if needed
        ids_array = ids.astype(np.int64) if ids.dtype != np.int64 else ids

        # Remove from local FAISS index
        self._local_index.remove_ids(ids_array)

        # Update our ID mappings
        for id_val in ids:
            if id_val in self._rev_id_map:
                internal_idx = self._rev_id_map[id_val]
                del self._id_map[internal_idx]
                del self._rev_id_map[id_val]

            # Remove from vector storage
            if int(id_val) in self._vectors_by_id:
                del self._vectors_by_id[int(id_val)]

        # Update total count
        self.ntotal = self._local_index.ntotal

    def range_search(
            self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for all vectors within the specified radius.

        Args:
            x (np.ndarray): Query vectors, shape (n, d)
            radius (float): Maximum distance threshold

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - lims: array of shape (n+1) giving the boundaries of results for each query
                - distances: array of shape (sum_of_results) containing all distances
                - ids: array of shape (sum_of_results) containing user-provided IDs

        Raises:
            ValueError: If query vector shape doesn't match index dimension
            RuntimeError: If range search isn't supported by the underlying index
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Use local FAISS index for range search
        if hasattr(self._local_index, 'range_search'):
            return self._local_index.range_search(vectors, radius)
        else:
            raise RuntimeError("Underlying index doesn't support range search")

    def reconstruct(self, id_val: int) -> np.ndarray:
        """
        Reconstruct a vector from its ID.

        Args:
            id_val: The ID of the vector to reconstruct

        Returns:
            The reconstructed vector

        Raises:
            ValueError: If the ID is not found
            RuntimeError: If the underlying index doesn't support reconstruction
        """
        # Convert ID to int64 to ensure compatibility
        id_val = int(id_val)  # Make sure it's a Python int

        # Check if we have this vector stored
        if id_val in self._vectors_by_id:
            return self._vectors_by_id[id_val]

        # Try using the FAISS index
        try:
            return self._local_index.reconstruct(id_val)
        except Exception as e:
            raise ValueError(f"ID {id_val} not found in index: {e}")

    def reconstruct_n(self, ids, n=None) -> np.ndarray:
        """
        Reconstruct multiple vectors from their IDs.
        This function handles both calling formats:
        - reconstruct_n(ids, n) - where ids is a list/array of IDs and n is number to reconstruct
        - reconstruct_n(offset, n) - where offset is the starting index and n is how many to reconstruct

        Args:
            ids: Array of IDs to reconstruct or starting index
            n: Number of vectors to reconstruct (optional if ids is an array)

        Returns:
            Array of reconstructed vectors
        """
        # Check if this is the form reconstruct_n(ids) - array of IDs
        if isinstance(ids, (list, np.ndarray)) and n is None:
            n = len(ids)

            # Reconstruct vectors one by one
            vectors = np.zeros((n, self.d), dtype=np.float32)
            for i, id_val in enumerate(ids):
                vectors[i] = self.reconstruct(id_val)
            return vectors

        # This is the form reconstruct_n(offset, n) - starting index and count
        elif isinstance(ids, int):
            try:
                # Try using the underlying method first
                return self._local_index.reconstruct_n(ids, n)
            except Exception:
                # Fall back to individual reconstructions
                vectors = np.zeros((n, self.d), dtype=np.float32)
                for i in range(n):
                    try:
                        vectors[i] = self.reconstruct(ids + i)
                    except ValueError:
                        # If ID not found, fill with zeros
                        vectors[i].fill(0)
                return vectors

        # Invalid argument format
        else:
            raise ValueError(
                "Invalid arguments to reconstruct_n. Expected (ids, n) or (offset, n)"
            )

    def train(self, x: np.ndarray) -> None:
        """
        Train the underlying index if it requires training.

        Args:
            x (np.ndarray): Training vectors, shape (n, d)
        """
        if hasattr(self.index, 'train'):
            self.index.train(x)
            self.is_trained = getattr(self.index, 'is_trained', True)

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        self._local_index.reset()
        self.ntotal = 0
        self._id_map = {}
        self._rev_id_map = {}
        self._vectors_by_id = {}
        self.is_trained = getattr(self.index, 'is_trained', True)


class IndexIDMap2(IndexIDMap):
    """
    Proxy implementation of FAISS IndexIDMap2.

    IndexIDMap2 is an extension of IndexIDMap that allows replacing
    vector content while keeping the same IDs. This is useful when
    vectors need to be updated or when the indexed vectors are based on
    objects (like images or documents) that change over time.

    This class inherits most functionality from IndexIDMap but adds
    methods to replace vectors without changing their IDs.
    """

    def __init__(self, index):
        """
        Initialize the IndexIDMap2 with the given underlying index.

        Args:
            index: The FAISS index to wrap (e.g., IndexFlatL2, IndexIVFFlat)
        """
        # Initialize base attributes
        self.index = index
        self.is_trained = getattr(index, 'is_trained', True)
        self.ntotal = 0  # Start with no vectors
        self.d = index.d  # Vector dimension from underlying index
        self._using_remote = False

        # Get local index from wrapped index if available
        base_index = index._local_index if hasattr(index, '_local_index') else index

        # Create FAISS IndexIDMap2
        try:
            self._local_index = faiss.IndexIDMap2(base_index)
        except Exception as e:
            raise RuntimeError(f"Failed to create IndexIDMap2: {e}")

        # Initialize bidirectional mapping dictionaries
        self._id_map = {}  # Maps internal indices to user IDs
        self._rev_id_map = {}  # Maps user IDs to internal indices

    def replace_vector(self, id_val, vector: np.ndarray) -> None:
        """
        Replace a vector with a new one, preserving its ID.

        Args:
            id_val: ID of the vector to replace
            vector: New vector data

        Raises:
            ValueError: If the ID is not found or the vector has wrong dimensionality
        """
        # Validate vector dimension
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)

        if vector.shape[1] != self.d:
            raise ValueError(f"Vector has wrong dimension: {vector.shape[1]}, expected {self.d}")

        # Convert to float32 if needed
        vector = vector.astype(np.float32) if vector.dtype != np.float32 else vector

        # Convert ID to proper format
        id_array = np.array([id_val], dtype=np.int64)

        # Delegate to local FAISS index
        self._local_index.add_with_ids(vector, id_array)  # IndexIDMap2 will replace existing vectors

    def update_vectors(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """
        Update multiple vectors at once, preserving their IDs.

        Args:
            ids: Array of IDs to update
            vectors: New vector data, shape (n, d)

        Raises:
            ValueError: If any ID is not found or vectors have wrong shape
        """
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of IDs ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
            )

        # Check if all IDs exist
        missing = []
        for id_val in ids:
            if id_val not in self._rev_id_map:
                missing.append(id_val)

        if missing:
            raise ValueError(f"IDs not found: {missing}")

        # Rebuild the index
        all_known_ids = np.array(list(self._rev_id_map.keys()))

        # First, collect all vectors to keep (excluding those to be updated)
        keep_mask = np.ones(len(all_known_ids), dtype=bool)

        # Find indices of IDs to update
        for id_val in ids:
            idx = np.where(all_known_ids == id_val)[0][0]
            keep_mask[idx] = False

        # Get IDs and vectors to keep
        keep_ids = all_known_ids[keep_mask]

        # Get all vectors through reconstruction
        if len(keep_ids) > 0:
            if hasattr(self.index, 'reconstruct_n'):
                keep_internal_indices = np.array([
                    self._rev_id_map[id_val] for id_val in keep_ids
                ])
                keep_vectors = self.index.reconstruct_n(keep_internal_indices)
            else:
                # Fall back to single reconstruction
                keep_vectors = np.zeros((len(keep_ids), self.d), dtype=np.float32)
                for i, id_val in enumerate(keep_ids):
                    keep_vectors[i] = self.reconstruct(id_val)

            # Combine kept vectors with new vectors
            combined_vectors = np.vstack([keep_vectors, vectors])
            combined_ids = np.concatenate([keep_ids, ids])
        else:
            # Only update vectors
            combined_vectors = vectors
            combined_ids = ids

        # Reset and rebuild the index
        old_index = self.index
        self.reset()
        self.index = old_index
        self.index.reset()

        # Re-add all vectors
        self.add_with_ids(combined_vectors, combined_ids)

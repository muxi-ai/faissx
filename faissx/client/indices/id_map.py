#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IndexIDMap and IndexIDMap2 implementations for FAISSx.

These classes provide a drop-in replacement for FAISS IndexIDMap and IndexIDMap2,
supporting both local and remote execution modes.
"""

import logging
import numpy as np
import uuid
from typing import Tuple

# Import the client utilities
from faissx.client.client import get_client

# Set up logging
logger = logging.getLogger(__name__)


class IndexIDMap:
    """
    A drop-in replacement for FAISS IndexIDMap supporting both local and remote execution.

    IndexIDMap allows associating custom IDs with vectors, which can be useful
    when vectors represent entities with existing identifiers.

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

        Raises:
            RuntimeError: If there's an error creating the index
        """
        # Import the actual faiss module
        import faiss as native_faiss

        # Store the underlying index and its properties
        self.index = index
        self.is_trained = getattr(index, "is_trained", True)
        self.ntotal = 0  # Start with no vectors
        self.d = index.d  # Vector dimension from underlying index

        # Initialize bidirectional mapping dictionaries
        self._id_map = {}  # Maps internal indices to user IDs
        self._rev_id_map = {}  # Maps user IDs to internal indices

        # Store vectors for reconstruction
        self._vectors_by_id = {}

        # Default to local mode
        self._using_remote = False

        # Check if client exists (remote mode)
        client = get_client()
        if client is not None:
            try:
                # Remote mode is active
                self._using_remote = True
                self.client = client

                # Get the base index ID
                if hasattr(index, "index_id"):
                    self.base_index_id = index.index_id
                else:
                    raise ValueError("Base index must have an index_id for remote mode")

                # Generate unique name for the index
                self.name = f"index-idmap-{uuid.uuid4().hex[:8]}"

                # Create ID map on server
                try:
                    response = self.client.create_index(
                        name=self.name,
                        dimension=self.d,
                        index_type=f"IDMap:{self.base_index_id}",
                    )

                    self.index_id = response.get("index_id", self.name)
                    logging.info(f"Created remote IndexIDMap with ID {self.index_id}")
                    return
                except Exception as e:
                    # If the server doesn't support IDMap, raise a clear error
                    if "Unsupported index type: IDMap" in str(e):
                        raise RuntimeError(
                            "The server does not support IndexIDMap operations. "
                            "Please use local mode for IndexIDMap."
                        )
                    # For other errors, re-raise them
                    raise
            except (ValueError, RuntimeError):
                # Re-raise runtime errors without fallback
                raise
            except Exception as e:
                # Any other exception should result in local mode
                logging.warning(f"Using local mode for IndexIDMap due to error: {e}")
                self._using_remote = False

        # If we get here, we're in local mode
        # Get local index from wrapped index if available
        base_index = index._local_index if hasattr(index, "_local_index") else index

        # Create FAISS IndexIDMap
        try:
            self._local_index = native_faiss.IndexIDMap(base_index)
        except Exception as e:
            raise RuntimeError(f"Failed to create IndexIDMap: {e}")

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
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Convert IDs to int64 if needed
        ids_array = ids.astype(np.int64) if ids.dtype != np.int64 else ids

        if self._using_remote:
            # Use remote server for add_with_ids
            try:
                # Convert vectors to list for serialization
                vectors_list = vectors.tolist()
                ids_list = ids_array.tolist()

                request = {
                    "action": "add_with_ids",
                    "index_id": self.index_id,
                    "vectors": vectors_list,
                    "ids": ids_list,
                }

                response = self.client._send_request(request)

                # If successful, update our tracking information
                if response.get("success", False):
                    added_count = response.get("count", 0)

                    # Update local tracking of ID mappings
                    for i, id_val in enumerate(ids):
                        id_val_int = int(id_val)  # Convert to int to use as key
                        self._id_map[self.ntotal + i] = id_val_int
                        self._rev_id_map[id_val_int] = self.ntotal + i

                        # Store vectors for reconstruction
                        self._vectors_by_id[id_val_int] = vectors[i].copy()

                    # Update total count
                    self.ntotal += added_count
            except Exception as e:
                raise RuntimeError(f"Failed to add vectors with IDs: {e}")
        else:
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
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if self._using_remote:
            # Use remote server for search
            try:
                result = self.client.search(self.index_id, vectors, k)

                n = x.shape[0]  # Number of query vectors
                search_results = result.get("results", [])

                # Initialize output arrays with default values
                distances = np.full((n, k), float("inf"), dtype=np.float32)
                indices = np.full((n, k), -1, dtype=np.int64)

                # Process results for each query vector
                for i in range(min(n, len(search_results))):
                    result_data = search_results[i]
                    result_distances = result_data.get("distances", [])
                    result_indices = result_data.get("indices", [])

                    # Fill in results for this query vector
                    for j in range(min(k, len(result_distances))):
                        distances[i, j] = result_distances[j]
                        indices[i, j] = result_indices[
                            j
                        ]  # Server returns user IDs directly

                return distances, indices
            except Exception as e:
                raise RuntimeError(f"Search failed in remote mode: {e}")
        else:
            # Use local FAISS index to search
            return self._local_index.search(vectors, k)

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

        if self._using_remote:
            # Use remote server for remove_ids
            try:
                request = {
                    "action": "remove_ids",
                    "index_id": self.index_id,
                    "ids": ids_array.tolist(),
                }

                response = self.client._send_request(request)

                # If successful, update our tracking information
                if response.get("success", False):
                    removed_count = response.get("count", 0)

                    # Update local tracking of ID mappings
                    for id_val in ids:
                        id_val_int = int(id_val)
                        if id_val_int in self._rev_id_map:
                            internal_idx = self._rev_id_map[id_val_int]
                            del self._id_map[internal_idx]
                            del self._rev_id_map[id_val_int]

                            # Remove from vector storage
                            if id_val_int in self._vectors_by_id:
                                del self._vectors_by_id[id_val_int]

                    # Update total count
                    self.ntotal -= removed_count
            except Exception as e:
                raise RuntimeError(f"Failed to remove IDs: {e}")
        else:
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
        self, x: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Convert to float32 if needed
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if self._using_remote:
            # Use remote server for range_search
            try:
                result = self.client.range_search(self.index_id, vectors, radius)

                if not result.get("success", False):
                    error = result.get("error", "Unknown error")
                    raise RuntimeError(f"Range search failed: {error}")

                # Process results
                search_results = result.get("results", [])
                n_queries = len(search_results)

                # Calculate total number of results across all queries
                total_results = sum(res.get("count", 0) for res in search_results)

                # Initialize arrays
                lims = np.zeros(n_queries + 1, dtype=np.int64)
                distances = np.zeros(total_results, dtype=np.float32)
                indices = np.zeros(total_results, dtype=np.int64)

                # Fill arrays with results
                offset = 0
                for i, res in enumerate(search_results):
                    # Set limit boundary for this query
                    lims[i] = offset

                    # Get results for this query
                    result_distances = res.get("distances", [])
                    result_indices = res.get(
                        "indices", []
                    )  # These are user IDs from server
                    count = len(result_distances)

                    # Copy data to output arrays
                if count > 0:
                    # Convert to numpy array and assign to output
                    distances[offset:offset + count] = np.array(
                        result_distances, dtype=np.float32
                    )
                    indices[offset:offset + count] = np.array(
                        result_indices, dtype=np.int64
                    )
                offset += count

                # Set final boundary
                lims[n_queries] = offset

                return lims, distances, indices
            except Exception as e:
                raise RuntimeError(f"Range search failed in remote mode: {e}")
        else:
            # Use local FAISS index for range search
            if hasattr(self._local_index, "range_search"):
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

        if self._using_remote:
            # Check local cache first
            if id_val in self._vectors_by_id:
                return self._vectors_by_id[id_val]

            # Use remote server for reconstruction
            try:
                request = {
                    "action": "reconstruct",
                    "index_id": self.index_id,
                    "id": id_val,
                }

                response = self.client._send_request(request)

                if response.get("success", False):
                    vector = np.array(response.get("vector", []), dtype=np.float32)

                    # Cache the vector for future use
                    self._vectors_by_id[id_val] = vector

                    return vector
                else:
                    raise ValueError(f"ID {id_val} not found in index")
            except Exception as e:
                raise RuntimeError(f"Reconstruction failed in remote mode: {e}")
        else:
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
        - reconstruct_n(offset, n) - offset = starting index and n is how many to reconstruct

        Args:
            ids: Array of IDs to reconstruct or starting index
            n: Number of vectors to reconstruct (optional if ids is an array)

        Returns:
            Array of reconstructed vectors
        """
        # Handle both calling formats
        if isinstance(ids, (list, np.ndarray)) and n is None:
            n = len(ids)

            # Reconstruct vectors one by one
            vectors = np.zeros((n, self.d), dtype=np.float32)
            for i, id_val in enumerate(ids):
                try:
                    vectors[i] = self.reconstruct(id_val)
                except (ValueError, RuntimeError):
                    # If reconstruction fails, fill with zeros
                    vectors[i].fill(0)
            return vectors

        elif isinstance(ids, int):
            if self._using_remote:
                # For remote mode, handle this format by reconstructing individual vectors
                vectors = np.zeros((n, self.d), dtype=np.float32)
                for i in range(n):
                    try:
                        vectors[i] = self.reconstruct(ids + i)
                    except (ValueError, RuntimeError):
                        # If reconstruction fails, fill with zeros
                        vectors[i].fill(0)
                return vectors
            else:
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
        if self._using_remote:
            # Training should be done on the base index, not the IDMap wrapper
            if hasattr(self.index, "train"):
                self.index.train(x)
                self.is_trained = self.index.is_trained
        else:
            if hasattr(self.index, "train"):
                self.index.train(x)
                self.is_trained = getattr(self.index, "is_trained", True)

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        if self._using_remote:
            try:
                # Create new index with modified name
                new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

                # Recreate the IDMap index with the same base index
                response = self.client.create_index(
                    name=new_name,
                    dimension=self.d,
                    index_type=f"IDMap:{self.base_index_id}",
                )

                self.index_id = response.get("index_id", new_name)
                self.name = new_name

                # Reset all local state
                self.ntotal = 0
                self._id_map = {}
                self._rev_id_map = {}
                self._vectors_by_id = {}
            except Exception as e:
                raise RuntimeError(f"Failed to reset remote IndexIDMap: {e}")
        else:
            self._local_index.reset()
            self.ntotal = 0
            self._id_map = {}
            self._rev_id_map = {}
            self._vectors_by_id = {}
            self.is_trained = getattr(self.index, "is_trained", True)


class IndexIDMap2(IndexIDMap):
    """
    A drop-in replacement for FAISS IndexIDMap2 supporting both local and remote execution.

    IndexIDMap2 is a variant of IndexIDMap that supports replacement of existing vectors.
    """

    def __init__(self, index):
        """
        Initialize the IndexIDMap2 with the given underlying index.

        Args:
            index: The FAISS index to wrap (e.g., IndexFlatL2, IndexIVFFlat)
        """
        # Import the actual faiss module
        import faiss as native_faiss

        # Initialize base attributes
        self.index = index
        self.is_trained = getattr(index, "is_trained", True)
        self.ntotal = 0  # Start with no vectors
        self.d = index.d  # Vector dimension from underlying index

        # Initialize mapping dictionaries and vector cache
        self._id_map = {}  # Maps internal indices to user IDs
        self._rev_id_map = {}  # Maps user IDs to internal indices
        self._vectors_by_id = {}  # Cache for vector reconstruction

        # Default to local mode
        self._using_remote = False

        # Check if client exists (remote mode)
        client = get_client()
        if client is not None:
            try:
                # Remote mode is active
                self._using_remote = True
                self.client = client

                # Get the base index ID
                if hasattr(index, "index_id"):
                    self.base_index_id = index.index_id
                else:
                    raise ValueError("Base index must have an index_id for remote mode")

                # Generate unique name for the index
                self.name = f"index-idmap2-{uuid.uuid4().hex[:8]}"

                # Create ID map on server
                try:
                    response = self.client.create_index(
                        name=self.name,
                        dimension=self.d,
                        index_type=f"IDMap2:{self.base_index_id}",
                    )

                    self.index_id = response.get("index_id", self.name)
                    logging.info(f"Created remote IndexIDMap2 with ID {self.index_id}")
                    return
                except Exception as e:
                    # If the server doesn't support IDMap2, raise a clear error
                    if "Unsupported index type: IDMap2" in str(e):
                        raise RuntimeError(
                            "The server does not support IndexIDMap2 operations. "
                            "Please use local mode for IndexIDMap2."
                        )
                    # For other errors, re-raise them
                    raise
            except (ValueError, RuntimeError):
                # Re-raise runtime errors without fallback
                raise
            except Exception as e:
                # Any other exception should result in local mode
                logging.warning(f"Using local mode for IndexIDMap2 due to error: {e}")
                self._using_remote = False

        # If we get here, we're in local mode
        # Get local index from wrapped index if available
        base_index = index._local_index if hasattr(index, "_local_index") else index

        # Create FAISS IndexIDMap2
        try:
            self._local_index = native_faiss.IndexIDMap2(base_index)
        except Exception as e:
            raise RuntimeError(f"Failed to create IndexIDMap2: {e}")

        # Initialize bidirectional mapping dictionaries
        self._id_map = {}  # Maps internal indices to user IDs
        self._rev_id_map = {}  # Maps user IDs to internal indices
        self._vectors_by_id = {}

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
            raise ValueError(
                f"Vector has wrong dimension: {vector.shape[1]}, expected {self.d}"
            )

        # Convert to float32 if needed
        vector = vector.astype(np.float32) if vector.dtype != np.float32 else vector

        # Convert ID to proper format
        id_array = np.array([id_val], dtype=np.int64)

        if self._using_remote:
            # Use remote server for replace_vector
            try:
                request = {
                    "action": "replace_vector",
                    "index_id": self.index_id,
                    "id": int(id_val),
                    "vector": vector.reshape(-1).tolist(),
                }

                response = self.client._send_request(request)

                # If successful, update our local cache
                if response.get("success", False):
                    # Cache the new vector
                    self._vectors_by_id[int(id_val)] = vector.reshape(-1)
            except Exception as e:
                raise RuntimeError(f"Failed to replace vector: {e}")
        else:
            # Delegate to local FAISS index
            # IndexIDMap2 replaces vectors with the same ID
            self._local_index.add_with_ids(vector, id_array)

            # Update local cache
            self._vectors_by_id[int(id_val)] = vector.reshape(-1)

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

        if self._using_remote:
            # Update vectors one by one for now
            # This could be optimized with a batch update API in the future
            for i, id_val in enumerate(ids):
                self.replace_vector(id_val, vectors[i:i + 1])
        else:
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
                if hasattr(self.index, "reconstruct_n"):
                    keep_internal_indices = np.array(
                        [self._rev_id_map[id_val] for id_val in keep_ids]
                    )
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

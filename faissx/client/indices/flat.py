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
FAISSx IndexFlatL2 implementation.

This module provides a client-side implementation of the FAISS IndexFlatL2 class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).

The IndexFlatL2 is the simplest index type, storing vectors in memory and
computing exact Euclidean (L2) distances during search. It's suitable for
small to medium datasets where accuracy is more important than speed.
"""

import uuid
import numpy as np
from typing import Tuple, Any, Optional, Type

# Import the base module which provides access to FAISS
from .base import logger, FAISSxBaseIndex
from ..client import get_client

# Default parameter values
DEFAULT_BATCH_SIZE = 1000
DEFAULT_SEARCH_BATCH_SIZE = 100
DEFAULT_RECONSTRUCT_BATCH_SIZE = 100
DEFAULT_K_FACTOR = 1.0


class IndexFlatL2(FAISSxBaseIndex):
    """
    Proxy implementation of FAISS IndexFlatL2.

    This class provides a drop-in replacement for the FAISS IndexFlatL2 class,
    but can operate in either local mode (using FAISS directly) or remote mode
    (using the FAISSx server).

    IndexFlatL2 uses exact Euclidean (L2) distance computation, which provides
    100% accurate results but scales linearly with dataset size. It stores
    vectors in raw form without compression.

    Features:
    - Exact L2 distance computation
    - Works with both local and remote backends
    - GPU acceleration in local mode (when available)
    - Efficient batch processing for large operations

    Example:
        ```python
        # Create a 128-dimensional flat index
        index = IndexFlatL2(128)

        # Add vectors
        vectors = np.random.random((1000, 128)).astype('float32')
        index.add(vectors)

        # Search for nearest neighbors
        query = np.random.random((1, 128)).astype('float32')
        distances, indices = index.search(query, k=5)
        ```
    """

    def __init__(self, d: int, use_gpu: bool = False) -> None:
        """
        Initialize the index with the given dimensionality.

        Args:
            d: Vector dimension
            use_gpu: Whether to use GPU acceleration (local mode only)
        """
        super().__init__()

        # Import the actual faiss module at the top-level scope to ensure it's available
        import faiss as native_faiss

        # Store core parameters
        self.d: int = d  # Vector dimension
        self.ntotal: int = 0  # Total number of vectors in index
        self.is_trained: bool = True  # Flat indices don't need training
        self._local_index: Optional[Any] = None  # Local FAISS index instance
        self._use_gpu: bool = use_gpu  # GPU acceleration flag
        self._gpu_resources: Optional[Any] = None  # GPU resources for local mode

        # Generate unique identifier for the index
        self.name: str = f"index-flat-l2-{uuid.uuid4().hex[:8]}"
        self.index_id: str = self.name

        # Check if client exists and its mode
        client = get_client()

        if client is not None and client.mode == "remote":
            # Remote mode - create index on server
            logger.info(f"Creating remote IndexFlatL2 on server {client.server}")
            self._create_remote_index(client, d)
        else:
            # Local mode - create index directly
            logger.info(f"Creating local IndexFlatL2 with dimension {d}")
            self._create_local_index(d, use_gpu, native_faiss)

    def _create_local_index(self, d: int, use_gpu: bool, native_faiss: Any) -> None:
        """
        Create a local FAISS index.

        Initializes a flat index using the native FAISS library. If GPU acceleration is requested
        and available, the index will be moved to GPU memory for faster computation.

        Args:
            d: Vector dimension
            use_gpu: Whether to use GPU acceleration
            native_faiss: The imported FAISS module
        """
        self._local_index = native_faiss.IndexFlatL2(d)

        if use_gpu:
            try:
                if native_faiss.get_num_gpus() > 0:
                    self._gpu_resources = native_faiss.StandardGpuResources()
                    self._local_index = native_faiss.index_cpu_to_gpu(
                        self._gpu_resources, 0, self._local_index
                    )
                    logger.info(f"Using GPU-accelerated flat index for {self.name}")
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f"GPU requested but FAISS GPU support not available: {e}"
                )
                self._use_gpu = False

    def _create_remote_index(self, client: Any, d: int) -> None:
        """
        Create a remote flat index on the server.

        Sends a request to the FAISSx server to create a new flat index with L2 distance
        metric. Properly handles the server response and stores the assigned index ID.

        Args:
            client: The FAISSx client connection
            d: Vector dimension for the new index

        Raises:
            RuntimeError: If the server fails to create the index
        """
        try:
            logger.debug(f"Creating remote index {self.name} with dimension {d}")
            response = client.create_index(self.name, d, "L2")
            logger.debug(f"Server response: {response}")

            if isinstance(response, dict):
                self.index_id = response.get("index_id", self.name)
            else:
                logger.warning(f"Unexpected server response format: {response}")
                self.index_id = self.name

            logger.info(f"Created remote index with ID: {self.index_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to create remote flat index: {e}")

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        This method adds new vectors to the index. The vectors will be accessible
        for searching immediately after they are added. In local mode, vectors are
        added directly to the FAISS index. In remote mode, vectors are sent to the
        FAISSx server in batches.

        Args:
            x: Vectors to add, shape (n, d) where n is the number of vectors
               and d is the dimension (must match the index dimension)

        Raises:
            ValueError: If vector shape doesn't match index dimension
            RuntimeError: If adding vectors to the remote index fails

        Example:
            ```python
            # Add 1000 vectors of dimension 128
            vectors = np.random.random((1000, 128)).astype('float32')
            index.add(vectors)
            ```
        """
        self.register_access()

        # Validate input dimensions
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        client = get_client()

        if client is not None and client.mode == "remote":
            self._add_remote(client, x)
        else:
            self._add_local(x)

    def _add_local(self, vectors: np.ndarray) -> None:
        """
        Add vectors to local index.

        Adds the provided vectors to the local FAISS index and updates the total vector count.

        Args:
            vectors: Vectors to add, shape (n, d), already validated for correct shape
        """
        logger.debug(f"Adding {len(vectors)} vectors to local index {self.name}")
        self._local_index.add(vectors)
        self.ntotal = self._local_index.ntotal

    def _add_remote(self, client: Any, vectors: np.ndarray) -> None:
        """
        Add vectors to remote index with batch processing.

        Handles splitting large vector sets into more manageable batches for sending
        to the server. This prevents timeouts and memory issues when working with
        very large datasets.

        Args:
            client: The FAISSx client connection
            vectors: Vectors to add, shape (n, d), already validated

        Note:
            Uses the batch_size parameter to determine optimal batch size
        """
        logger.debug(f"Adding {len(vectors)} vectors to remote index {self.index_id}")

        try:
            batch_size = self.get_parameter("batch_size")
        except ValueError:
            batch_size = DEFAULT_BATCH_SIZE

        if len(vectors) <= batch_size:
            self._add_remote_batch(client, vectors)
            return

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:min(i + batch_size, len(vectors))]
            self._add_remote_batch(client, batch)

    def _add_remote_batch(self, client: Any, vectors: np.ndarray) -> None:
        """
        Add a batch of vectors to remote index.

        Sends a single batch of vectors to the FAISSx server and processes the response.
        Updates the total vector count based on the server's confirmation.

        Args:
            client: The FAISSx client connection
            vectors: Batch of vectors to add

        Raises:
            RuntimeError: If the server fails to add vectors or returns an error
        """
        try:
            response = client.add_vectors(self.index_id, vectors)
            logger.debug(f"Server response: {response}")

            if isinstance(response, dict) and response.get("success", False):
                self.ntotal += response.get("count", 0)
            elif not isinstance(response, dict):
                logger.warning(f"Unexpected response format from server: {response}")
                self.ntotal += len(vectors)
        except Exception as e:
            logger.error(f"Error adding vectors to remote index: {e}")
            raise RuntimeError(f"Failed to add vectors to remote index: {e}")

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.

        Performs exact nearest neighbor search using Euclidean (L2) distance.
        Returns the k closest vectors to each query vector, along with their distances.

        In local mode, this directly calls the FAISS index's search method.
        In remote mode, this sends queries to the FAISSx server with automatic batching
        for large query sets.

        Args:
            x: Query vectors, shape (n, d) where n is the number of queries
               and d is the dimension (must match the index dimension)
            k: Number of nearest neighbors to return per query

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - distances: Array of shape (n, k) containing L2 distances to each neighbor
                  (sorted from smallest to largest)
                - indices: Array of shape (n, k) containing the indices of each neighbor
                  (corresponding to the order vectors were added to the index)

        Raises:
            ValueError: If query vector shape doesn't match index dimension or k <= 0
            RuntimeError: If the search operation fails on the remote server

        Example:
            ```python
            # Search for 5 nearest neighbors
            query = np.random.random((1, 128)).astype('float32')
            distances, indices = index.search(query, k=5)

            # Print results
            for i in range(5):
                print(f"Neighbor {i}: index={indices[0, i]}, distance={distances[0, i]}")
            ```

        Note:
            The 'k_factor' parameter can be used to expand the internal search
            and then re-rank the results:
            ```python
            index.set_parameter("k_factor", 2.0)  # Search for 2*k vectors internally
            ```
        """
        self.register_access()

        # Validate input dimensions
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Validate k parameter
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        # Handle empty index case
        if self.ntotal == 0:
            n = x.shape[0]
            distances = np.full((n, k), float("inf"), dtype=np.float32)
            indices = np.full((n, k), -1, dtype=np.int64)
            return distances, indices

        # Get k_factor parameter for search expansion
        try:
            k_factor = max(1.0, self.get_parameter("k_factor"))
        except ValueError:
            k_factor = DEFAULT_K_FACTOR

        # Calculate internal k and determine if reranking is needed
        internal_k = min(int(k * k_factor), self.ntotal)
        need_reranking = k_factor > 1.0 and internal_k > k

        client = get_client()

        if client is not None and client.mode == "remote":
            return self._search_remote(client, x, k, internal_k, need_reranking)
        else:
            return self._search_local(x, k, internal_k, need_reranking)

    def _search_local(
        self, query_vectors: np.ndarray, k: int, internal_k: int, need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in local index.

        Performs k-nearest neighbor search using the local FAISS index, with optional
        re-ranking of results.

        Args:
            query_vectors: Query vectors, shape (n, d), already validated
            k: Number of results to return per query
            internal_k: Internal k value (possibly increased by k_factor)
            need_reranking: Whether to rerank results

        Returns:
            Tuple of distances and indices arrays, both shape (n, k)
        """
        logger.debug(
            f"Searching {len(query_vectors)} vectors in local index {self.name}"
        )

        distances, indices = self._local_index.search(query_vectors, internal_k)

        if need_reranking:
            distances = distances[:, :k]
            indices = indices[:, :k]

        return distances, indices

    def _search_remote(
        self,
        client: Any,
        query_vectors: np.ndarray,
        k: int,
        internal_k: int,
        need_reranking: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in remote index.

        Performs k-nearest neighbor search using the remote FAISSx server. Handles
        batching of large query sets for better performance and reliability.

        Args:
            client: The FAISSx client connection
            query_vectors: Query vectors, shape (n, d), already validated
            k: Number of results to return per query
            internal_k: Internal k value (possibly increased by k_factor)
            need_reranking: Whether to rerank results

        Returns:
            Tuple of distances and indices arrays, both shape (n, k)
        """
        logger.debug(
            f"Searching {len(query_vectors)} vectors in remote index {self.index_id}"
        )

        # Get the batch size for search operations
        try:
            batch_size = self.get_parameter("batch_size")
        except ValueError:
            batch_size = DEFAULT_SEARCH_BATCH_SIZE

        if batch_size <= 0:
            batch_size = DEFAULT_SEARCH_BATCH_SIZE

        # If queries fit in a single batch, search directly
        if len(query_vectors) <= batch_size:
            return self._search_remote_batch(
                client, query_vectors, k, internal_k, need_reranking
            )

        # Process in batches
        all_distances = []
        all_indices = []

        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:min(i + batch_size, len(query_vectors))]
            distances, indices = self._search_remote_batch(
                client, batch, k, internal_k, need_reranking
            )
            all_distances.append(distances)
            all_indices.append(indices)

        # Combine results
        if len(all_distances) == 1:
            return all_distances[0], all_indices[0]
        else:
            return np.vstack(all_distances), np.vstack(all_indices)

    def _search_remote_batch(
        self,
        client: Any,
        query_vectors: np.ndarray,
        k: int,
        internal_k: int,
        need_reranking: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search a batch of queries in remote index.

        Args:
            client: The FAISSx client
            query_vectors: Batch of query vectors
            k: Number of results to return
            internal_k: Internal k value (possibly increased by k_factor)
            need_reranking: Whether to rerank results

        Returns:
            Tuple of distances and indices arrays

        Raises:
            RuntimeError: If the server returns an error or invalid response
        """
        try:
            # Send request to server
            response = client.search(self.index_id, query_vectors, internal_k)

            # Log response
            logger.debug(f"Server response: {response}")

            # Initialize default return values
            n = query_vectors.shape[0]
            distances = np.full((n, k), float("inf"), dtype=np.float32)
            indices = np.full((n, k), -1, dtype=np.int64)

            # Validate response format
            if not isinstance(response, dict):
                logger.warning(f"Unexpected response type: {type(response)}")
                return distances, indices

            if "results" not in response:
                logger.warning("Missing 'results' key in response")
                return distances, indices

            # Extract results list
            search_results = response["results"]
            if not isinstance(search_results, list):
                logger.warning(
                    f"Invalid results format, expected list: {type(search_results)}"
                )
                return distances, indices

            # Process results for each query
            for i in range(min(n, len(search_results))):
                result_data = search_results[i]

                # Skip if result data is invalid
                if not isinstance(result_data, dict):
                    logger.warning(f"Invalid result data for query {i}: {result_data}")
                    continue

                # Extract distances and indices if available
                if "distances" in result_data and "indices" in result_data:
                    result_distances = np.array(result_data["distances"])
                    result_indices = np.array(result_data["indices"])

                    # Apply reranking if needed
                    if need_reranking:
                        result_distances = result_distances[:k]
                        result_indices = result_indices[:k]

                    # Copy results to output arrays
                    max_j = min(k, len(result_distances))
                    distances[i, :max_j] = result_distances[:max_j]
                    indices[i, :max_j] = result_indices[:max_j]
                else:
                    logger.warning(f"Missing distances or indices for query {i}")

            return distances, indices
        except Exception as e:
            logger.error(f"Error searching remote index: {e}")
            raise RuntimeError(f"Failed to search remote index: {e}")

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        # Register access for memory management
        self.register_access()

        client = get_client()

        if client is not None and client.mode == "remote":
            self._reset_remote(client)
        else:
            self._reset_local()

    def _reset_local(self) -> None:
        """Reset the local index."""
        logger.debug(f"Resetting local index {self.name}")

        if self._local_index is not None:
            self._local_index.reset()
            self.ntotal = 0

        # Reset parameters
        self.reset_parameters()

    def _reset_remote(self, client: Any) -> None:
        """Reset the remote index."""
        logger.debug(f"Resetting remote index {self.index_id}")

        try:
            # First try to delete the index
            response = client.delete_index(self.name)
            logger.debug(f"Delete index response: {response}")

            # Then recreate it
            response = client.create_index(self.name, self.d, "L2")
            logger.debug(f"Create index response: {response}")

            if isinstance(response, dict):
                self.index_id = response.get("index_id", self.name)
            else:
                # If response is not a dict, use the name as ID
                logger.warning(f"Unexpected server response format: {response}")
                self.index_id = self.name

            self.ntotal = 0
        except Exception as e:
            logger.error(f"Error resetting remote index: {e}")
            raise RuntimeError(f"Failed to reset remote index: {e}")

        # Reset parameters
        self.reset_parameters()

    def range_search(
        self, x: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for all vectors within the specified radius.

        Unlike k-nearest neighbor search, range search returns all vectors within
        a fixed distance threshold from each query vector. The number of results may
        vary for each query, from zero to potentially all vectors in the index.

        This method handles both local and remote execution, with automatic batching
        for large query sets in remote mode.

        Args:
            x: Query vectors, shape (n, d) where n is the number of queries
               and d is the dimension (must match the index dimension)
            radius: Maximum distance threshold (any vector with L2 distance <= radius is returned)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - lims: Array of shape (n+1) giving the boundaries of results for each query.
                  For query i, the results are in positions lims[i]:lims[i+1] in the
                  distances and indices arrays.
                - distances: Array of shape (sum_of_results) containing all distances,
                  where sum_of_results is the total number of neighbors found.
                - indices: Array of shape (sum_of_results) containing all indices.

        Raises:
            ValueError: If query vector shape doesn't match index dimension
            RuntimeError: If the range search operation fails

        Example:
            ```python
            # Find all vectors within radius 0.5
            query = np.random.random((1, 128)).astype('float32')
            lims, distances, indices = index.range_search(query, radius=0.5)

            # Process results for the first query
            start, end = lims[0], lims[1]
            results = list(zip(indices[start:end], distances[start:end]))
            print(f"Found {len(results)} neighbors within radius 0.5")
            ```
        """
        # Register access for memory management
        self.register_access()

        # Validate input dimensions
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        client = get_client()

        if client is not None and client.mode == "remote":
            return self._range_search_remote(client, x, radius)
        else:
            return self._range_search_local(x, radius)

    def _range_search_local(
        self, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform range search in local index.

        Args:
            query_vectors: Query vectors
            radius: Search radius

        Returns:
            Tuple of lims, distances, and indices arrays
        """
        logger.debug(
            f"Range searching {len(query_vectors)} vectors in local index {self.name}"
        )
        return self._local_index.range_search(query_vectors, radius)

    def _range_search_remote(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform range search in remote index.

        Handles batching of queries for efficient processing on the server.

        Args:
            client: The FAISSx client
            query_vectors: Query vectors
            radius: Search radius

        Returns:
            Tuple of lims, distances, and indices arrays

        Raises:
            RuntimeError: If the server returns an error
        """
        logger.debug(
            f"Range searching {len(query_vectors)} vectors in remote index {self.index_id}"
        )

        # Get the batch size for search operations
        try:
            batch_size = self.get_parameter("batch_size")
        except ValueError:
            batch_size = DEFAULT_SEARCH_BATCH_SIZE

        if batch_size <= 0:
            batch_size = DEFAULT_SEARCH_BATCH_SIZE

        # For small query sets, process in a single batch
        if len(query_vectors) <= batch_size:
            return self._range_search_remote_batch(client, query_vectors, radius)

        # For large query sets, process in batches and combine results
        all_lims = [0]  # First element is always 0
        all_distances = []
        all_indices = []

        # Process each batch
        for i in range(0, len(query_vectors), batch_size):
            # Extract the current batch
            batch = query_vectors[i:min(i + batch_size, len(query_vectors))]

            # Process the batch
            lims, distances, indices = self._range_search_remote_batch(
                client, batch, radius
            )

            # Adjust lims for concatenation (except first entry which is always 0)
            # We need to offset the lims by the current count of results
            if all_distances:
                lims = lims[1:] + all_lims[-1]

            # Add the result boundaries, distances, and indices to our overall results
            all_lims.extend(lims[1:])
            all_distances.extend(distances)
            all_indices.extend(indices)

        # Convert result lists to numpy arrays
        return np.array(all_lims), np.array(all_distances), np.array(all_indices)

    def _range_search_remote_batch(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform range search for a batch of queries in remote index.

        Args:
            client: The FAISSx client
            query_vectors: Batch of query vectors
            radius: Search radius

        Returns:
            Tuple of lims, distances, and indices arrays
        """
        try:
            # Send request to server
            response = client.range_search(self.index_id, query_vectors, radius)

            # Log response
            logger.debug(f"Server response: {response}")

            # Process response
            if isinstance(response, dict):
                lims = np.array(response.get("lims", [0]))
                distances = np.array(response.get("distances", []))
                indices = np.array(response.get("indices", []))
                return lims, distances, indices
            else:
                # Handle unexpected response format
                logger.warning(f"Unexpected range search response format: {response}")
                n = query_vectors.shape[0]
                return np.array([0] * (n + 1)), np.array([]), np.array([])
        except Exception as e:
            logger.error(f"Error in range search on remote index: {e}")
            raise RuntimeError(f"Failed to perform range search on remote index: {e}")

    def reconstruct(self, i: int) -> np.ndarray:
        """
        Reconstruct a vector from the index.

        Retrieves the raw vector at the specified index position. For flat indices,
        this simply returns the stored vector without any decoding or transformation.

        This method handles both local and remote execution automatically.

        Args:
            i: Index of the vector to reconstruct (in the range [0, ntotal))

        Returns:
            np.ndarray: Reconstructed vector of dimension d

        Raises:
            IndexError: If index is out of bounds
            RuntimeError: If reconstruction fails on the server

        Example:
            ```python
            # Retrieve the vector at index 5
            vector = index.reconstruct(5)
            print(f"Vector dimension: {len(vector)}")
            ```
        """
        # Register access for memory management
        self.register_access()

        # Validate index is in bounds
        if i < 0 or i >= self.ntotal:
            raise IndexError(f"Index {i} is out of bounds [0, {self.ntotal})")

        client = get_client()

        if client is not None and client.mode == "remote":
            return self._reconstruct_remote(client, i)
        else:
            return self._reconstruct_local(i)

    def _reconstruct_local(self, i: int) -> np.ndarray:
        """
        Reconstruct a vector from local index.

        For flat indices, this directly returns the stored vector at the specified index position.
        Since flat indices store raw vectors without compression, reconstruction is accurate
        and efficient.

        Args:
            i: Vector index to reconstruct

        Returns:
            The reconstructed vector as a numpy array of shape (d,)
        """
        logger.debug(f"Reconstructing vector {i} from local index {self.name}")
        return self._local_index.reconstruct(i)

    def _reconstruct_remote(self, client: Any, i: int) -> np.ndarray:
        """
        Reconstruct a vector from remote index.

        Requests the vector at the specified index from the remote server. For flat indices, the server
        returns the exact original vector. This method handles server communication, error checking,
        and response parsing.

        Args:
            client: The FAISSx client connection
            i: Vector index to reconstruct

        Returns:
            The reconstructed vector as a numpy array of shape (d,)

        Raises:
            RuntimeError: If the server returns an error or if communication fails
        """
        logger.debug(f"Reconstructing vector {i} from remote index {self.index_id}")

        try:
            # Send request to server
            response = client.reconstruct(self.index_id, i)

            # Log response
            logger.debug(f"Server response: {response}")

            # Process response
            if isinstance(response, dict) and "vector" in response:
                return np.array(response["vector"])
            else:
                # Handle unexpected response format
                logger.warning(f"Unexpected reconstruct response format: {response}")
                raise RuntimeError(
                    f"Invalid reconstruct response from server: {response}"
                )
        except Exception as e:
            logger.error(f"Error reconstructing vector from remote index: {e}")
            raise RuntimeError(f"Failed to reconstruct vector from remote index: {e}")

    def reconstruct_n(self, i0: int, ni: int) -> np.ndarray:
        """
        Reconstruct a batch of vectors from the index.

        Retrieves a continuous range of raw vectors from the index.
        For flat indices, this simply returns the stored vectors.
        Batching is handled automatically for remote requests.

        This method is more efficient than calling reconstruct() multiple times when retrieving
        many vectors, as it reduces the number of server round-trips in remote mode.

        Args:
            i0: First index to reconstruct (starting point)
            ni: Number of vectors to reconstruct (count)

        Returns:
            np.ndarray: Array of shape (ni, d) containing the reconstructed vectors

        Raises:
            IndexError: If any index in the range is out of bounds
            RuntimeError: If reconstruction fails on the server

        Example:
            ```python
            # Retrieve 10 vectors starting from index 20
            vectors = index.reconstruct_n(20, 10)
            print(f"Retrieved {len(vectors)} vectors of dimension {vectors.shape[1]}")
            ```
        """
        # Register access for memory management
        self.register_access()

        # Validate index range is in bounds
        if i0 < 0 or i0 + ni > self.ntotal:
            raise IndexError(
                f"Index range [{i0}, {i0+ni}) is out of bounds [0, {self.ntotal})"
            )

        client = get_client()

        if client is not None and client.mode == "remote":
            return self._reconstruct_n_remote(client, i0, ni)
        else:
            return self._reconstruct_n_local(i0, ni)

    def _reconstruct_n_local(self, i0: int, ni: int) -> np.ndarray:
        """
        Reconstruct a range of vectors from local index.

        Direct pass-through to the local FAISS index's reconstruct_n method, which retrieves
        a continuous block of vectors from the index. For flat indices, this is very efficient
        as it simply returns slices of the internally stored vectors.

        Args:
            i0: First index in the range to reconstruct
            ni: Number of vectors to reconstruct

        Returns:
            Array of reconstructed vectors with shape (ni, d)
        """
        logger.debug(
            f"Reconstructing vectors {i0}:{i0+ni} from local index {self.name}"
        )
        return self._local_index.reconstruct_n(i0, ni)

    def _reconstruct_n_remote(self, client: Any, i0: int, ni: int) -> np.ndarray:
        """
        Reconstruct a range of vectors from remote index.

        Handles batching of large requests to avoid overwhelming the server. For flat indices,
        reconstruction is straightforward as the vectors are stored in their original form.

        The method implements a batching strategy to:
        1. Break large requests into smaller batches
        2. Process each batch with the server
        3. Combine the results into a single continuous array

        Args:
            client: The FAISSx client connection
            i0: First index to reconstruct
            ni: Number of vectors to reconstruct

        Returns:
            Array of reconstructed vectors with shape (ni, d)

        Raises:
            RuntimeError: If the server returns an error
        """
        logger.debug(
            f"Reconstructing vectors {i0}:{i0+ni} from remote index {self.index_id}"
        )

        # Get the batch size for reconstruct operations
        try:
            batch_size = self.get_parameter("reconstruct_batch_size")
        except ValueError:
            batch_size = DEFAULT_RECONSTRUCT_BATCH_SIZE

        if batch_size <= 0:
            batch_size = DEFAULT_RECONSTRUCT_BATCH_SIZE

        # For small batches, use a direct call
        if ni <= batch_size:
            return self._reconstruct_n_remote_batch(client, i0, ni)

        # For larger batches, break up into smaller pieces
        result = []

        # Process each batch
        for i in range(i0, i0 + ni, batch_size):
            # Calculate the size of this batch (may be smaller for the last batch)
            current_ni = min(batch_size, i0 + ni - i)

            # Process the batch
            batch_result = self._reconstruct_n_remote_batch(client, i, current_ni)
            result.append(batch_result)

        # Combine all batch results
        return np.vstack(result)

    def _reconstruct_n_remote_batch(self, client: Any, i0: int, ni: int) -> np.ndarray:
        """
        Reconstruct a batch of vectors from remote index.

        Args:
            client: The FAISSx client
            i0: First index
            ni: Number of vectors

        Returns:
            Array of reconstructed vectors
        """
        try:
            # Send request to server
            response = client.reconstruct_n(self.index_id, i0, ni)

            # Log response
            logger.debug(f"Server response: {response}")

            # Process response
            if isinstance(response, dict) and "vectors" in response:
                return np.array(response["vectors"])
            else:
                # Handle unexpected response format
                logger.warning(f"Unexpected reconstruct_n response format: {response}")
                raise RuntimeError(
                    f"Invalid reconstruct_n response from server: {response}"
                )
        except Exception as e:
            logger.error(f"Error reconstructing vectors from remote index: {e}")
            raise RuntimeError(f"Failed to reconstruct vectors from remote index: {e}")

    def __enter__(self) -> 'IndexFlatL2':
        """
        Context manager entry point.

        Allows using the index with a 'with' statement, which ensures proper cleanup
        of resources even if an exception occurs.

        Returns:
            self: The current IndexFlatL2 instance for use in a with statement.

        Example:
            ```python
            with IndexFlatL2(128) as index:
                # Use the index
                index.add(vectors)
                distances, indices = index.search(query, k=5)
            # Index resources are automatically cleaned up
            ```
        """
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any
    ) -> None:
        """
        Context manager exit point.

        Automatically cleans up resources when exiting the 'with' block.

        Args:
            exc_type: Type of exception if one occurred, None otherwise.
            exc_val: Exception instance if one occurred, None otherwise.
            exc_tb: Traceback if an exception occurred, None otherwise.
        """
        self.close()

    def close(self) -> None:
        """
        Clean up resources used by the index.

        This method releases GPU resources if they were allocated and clears the local
        index reference. Should be called when the index is no longer needed to prevent
        resource leaks.

        After calling close(), the index should not be used anymore. For temporary
        use of an index with automatic cleanup, use the context manager pattern with
        a 'with' statement instead.
        """
        # Release GPU resources if they were allocated
        if self._use_gpu and self._gpu_resources is not None:
            self._gpu_resources = None

        # Clear the local index reference
        self._local_index = None

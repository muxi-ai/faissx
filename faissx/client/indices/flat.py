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
"""

import uuid
import numpy as np
from typing import Tuple, Any

# Import the base module which provides access to FAISS
from .base import logger

from ..client import get_client
from .base import FAISSxBaseIndex


class IndexFlatL2(FAISSxBaseIndex):
    """
    Proxy implementation of FAISS IndexFlatL2.

    This class provides a drop-in replacement for the FAISS IndexFlatL2 class,
    but can operate in either local mode (using FAISS directly) or remote mode
    (using the FAISSx server).
    """

    def __init__(self, d: int, use_gpu: bool = False):
        """
        Initialize the index with the given dimensionality.

        Args:
            d: Vector dimension
            use_gpu: Whether to use GPU acceleration (local mode only)
        """
        # Initialize base class
        super().__init__()

        # Import the actual faiss module at the top-level scope to ensure it's available
        import faiss as native_faiss

        # Store core parameters
        self.d = d  # Vector dimension
        self.ntotal = 0  # Total number of vectors in index
        self.is_trained = True  # Flat indices don't need training
        self._local_index = None  # Local FAISS index instance
        self._use_gpu = use_gpu  # GPU acceleration flag
        self._gpu_resources = None  # GPU resources for local mode

        # Generate unique identifier for the index
        self.name = f"index-flat-l2-{uuid.uuid4().hex[:8]}"
        self.index_id = self.name

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

        Args:
            d: Vector dimension
            use_gpu: Whether to use GPU acceleration
            native_faiss: The imported FAISS module
        """
        # Create a FAISS index directly
        self._local_index = native_faiss.IndexFlatL2(d)

        # Move to GPU if requested
        if use_gpu:
            try:
                # Try to use GPU resources if available
                if native_faiss.get_num_gpus() > 0:
                    self._gpu_resources = native_faiss.StandardGpuResources()
                    self._local_index = native_faiss.index_cpu_to_gpu(
                        self._gpu_resources, 0, self._local_index
                    )
                    logger.info(f"Using GPU-accelerated flat index for {self.name}")
            except (ImportError, AttributeError) as e:
                logger.warning(f"GPU requested but FAISS GPU support not available: {e}")
                self._use_gpu = False

    def _create_remote_index(self, client: Any, d: int) -> None:
        """
        Create a remote flat index on the server.

        Args:
            client: The FAISSx client
            d: Vector dimension
        """
        try:
            # Create index on server
            logger.debug(f"Creating remote index {self.name} with dimension {d}")
            response = client.create_index(self.name, d, "L2")

            # Log response
            logger.debug(f"Server response: {response}")

            # Parse response
            if isinstance(response, dict):
                self.index_id = response.get("index_id", self.name)
            else:
                # If response is not a dict, use the name as ID
                logger.warning(f"Unexpected server response format: {response}")
                self.index_id = self.name

            logger.info(f"Created remote index with ID: {self.index_id}")
        except Exception as e:
            # Raise a clear error instead of falling back to local mode
            raise RuntimeError(
                f"Failed to create remote flat index: {e}"
            )

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x: Vectors to add, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension
        """
        # Register access for memory management
        self.register_access()

        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        client = get_client()

        if client is not None and client.mode == "remote":
            self._add_remote(client, x)
        else:
            self._add_local(x)

    def _add_local(self, vectors: np.ndarray) -> None:
        """
        Add vectors to local index.

        Args:
            vectors: Vectors to add
        """
        logger.debug(f"Adding {len(vectors)} vectors to local index {self.name}")
        self._local_index.add(vectors)
        self.ntotal = self._local_index.ntotal

    def _add_remote(self, client: Any, vectors: np.ndarray) -> None:
        """
        Add vectors to remote index.

        Args:
            client: The FAISSx client
            vectors: Vectors to add
        """
        logger.debug(f"Adding {len(vectors)} vectors to remote index {self.index_id}")

        # Get the batch size for adding vectors
        batch_size = self.get_parameter('batch_size')
        if batch_size <= 0:
            batch_size = 1000  # Default batch size if not set

        # If vectors fit in a single batch, add directly
        if len(vectors) <= batch_size:
            self._add_remote_batch(client, vectors)
            return

        # Process in larger batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:min(i + batch_size, len(vectors))]
            self._add_remote_batch(client, batch)

    def _add_remote_batch(self, client: Any, vectors: np.ndarray) -> None:
        """
        Add a batch of vectors to remote index.

        Args:
            client: The FAISSx client
            vectors: Batch of vectors to add
        """
        try:
            # Send request to server
            response = client.add_vectors(self.index_id, vectors)

            # Log response
            logger.debug(f"Server response: {response}")

            # Update total count if successful
            if isinstance(response, dict) and response.get("success", False):
                added_count = response.get("count", 0)
                self.ntotal += added_count
            elif not isinstance(response, dict):
                # Handle case when response is not a dictionary
                logger.warning(f"Unexpected response format from server: {response}")
                # Assume we added all vectors
                self.ntotal += len(vectors)
        except Exception as e:
            logger.error(f"Error adding vectors to remote index: {e}")
            raise RuntimeError(f"Failed to add vectors to remote index: {e}")

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.

        Args:
            x: Query vectors, shape (n, d)
            k: Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Distances array of shape (n, k)
                - Indices array of shape (n, k)

        Raises:
            ValueError: If query vector shape doesn't match index dimension
        """
        # Register access for memory management
        self.register_access()

        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        # Handle empty index case
        if self.ntotal == 0:
            n = x.shape[0]
            # Split long line to avoid linter errors
            distances = np.full((n, k), float('inf'), dtype=np.float32)
            indices = np.full((n, k), -1, dtype=np.int64)
            return distances, indices

        # Apply k_factor parameter if set
        k_factor = self.get_parameter('k_factor')
        if k_factor <= 1.0:
            k_factor = 1.0

        internal_k = min(int(k * k_factor), self.ntotal)
        need_reranking = (k_factor > 1.0 and internal_k > k)

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

        Args:
            query_vectors: Query vectors
            k: Number of results to return
            internal_k: Internal k value (possibly increased by k_factor)
            need_reranking: Whether to rerank results

        Returns:
            Tuple of distances and indices arrays
        """
        logger.debug(f"Searching {len(query_vectors)} vectors in local index {self.name}")

        distances, indices = self._local_index.search(query_vectors, internal_k)

        # If k_factor was applied, rerank and trim results
        if need_reranking:
            distances = distances[:, :k]
            indices = indices[:, :k]

        return distances, indices

    def _search_remote(
        self, client: Any, query_vectors: np.ndarray, k: int,
        internal_k: int, need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in remote index.

        Args:
            client: The FAISSx client
            query_vectors: Query vectors
            k: Number of results to return
            internal_k: Internal k value (possibly increased by k_factor)
            need_reranking: Whether to rerank results

        Returns:
            Tuple of distances and indices arrays
        """
        logger.debug(
            f"Searching {len(query_vectors)} vectors in remote index {self.index_id}"
        )

        # Get the batch size for search operations
        batch_size = self.get_parameter('batch_size')
        if batch_size <= 0:
            batch_size = 100  # Default if not set

        # If queries fit in a single batch, search directly
        if len(query_vectors) <= batch_size:
            return self._search_remote_batch(client, query_vectors, k, internal_k, need_reranking)

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
        self, client: Any, query_vectors: np.ndarray, k: int,
        internal_k: int, need_reranking: bool
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
        """
        try:
            # Send request to server
            response = client.search(self.index_id, query_vectors, internal_k)

            # Log response
            logger.debug(f"Server response: {response}")

            # Initialize default return values
            n = query_vectors.shape[0]
            distances = np.full((n, k), float('inf'), dtype=np.float32)
            indices = np.full((n, k), -1, dtype=np.int64)

            # Process response
            if not isinstance(response, dict) or "results" not in response:
                logger.warning(f"Unexpected search response format: {response}")
                return distances, indices

            # Extract results list
            search_results = response["results"]
            if not isinstance(search_results, list):
                logger.warning(f"Invalid results format, expected list: {search_results}")
                return distances, indices

            # Process results for each query
            for i in range(min(n, len(search_results))):
                result_data = search_results[i]
                if not isinstance(result_data, dict):
                    continue

                # Extract distances and indices
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

        Args:
            x: Query vectors, shape (n, d)
            radius: Maximum distance threshold

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - lims: array of shape (n+1) giving the boundaries of results for each query
                - distances: array of shape (sum_of_results) containing all distances
                - indices: array of shape (sum_of_results) containing all indices

        Raises:
            ValueError: If query vector shape doesn't match index dimension
        """
        # Register access for memory management
        self.register_access()

        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

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
        logger.debug(f"Range searching {len(query_vectors)} vectors in local index {self.name}")
        return self._local_index.range_search(query_vectors, radius)

    def _range_search_remote(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform range search in remote index.

        Args:
            client: The FAISSx client
            query_vectors: Query vectors
            radius: Search radius

        Returns:
            Tuple of lims, distances, and indices arrays
        """
        logger.debug(f"Range searching {len(query_vectors)} vectors in remote index {self.index_id}")

        # Get the batch size for search operations
        batch_size = self.get_parameter('batch_size')
        if batch_size <= 0:
            batch_size = 100  # Default if not set

        # Process single batch directly
        if len(query_vectors) <= batch_size:
            return self._range_search_remote_batch(client, query_vectors, radius)

        # Handle multiple batches
        all_lims = [0]
        all_distances = []
        all_indices = []

        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:min(i + batch_size, len(query_vectors))]
            lims, distances, indices = self._range_search_remote_batch(
                client, batch, radius
            )

            # Adjust lims for concatenation (except first entry which is always 0)
            if all_distances:
                lims = lims[1:] + all_lims[-1]

            all_lims.extend(lims[1:])
            all_distances.extend(distances)
            all_indices.extend(indices)

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

        Args:
            i: Index of the vector to reconstruct

        Returns:
            Reconstructed vector

        Raises:
            IndexError: If index is out of bounds
        """
        # Register access for memory management
        self.register_access()

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

        Args:
            i: Vector index

        Returns:
            Reconstructed vector
        """
        logger.debug(f"Reconstructing vector {i} from local index {self.name}")
        return self._local_index.reconstruct(i)

    def _reconstruct_remote(self, client: Any, i: int) -> np.ndarray:
        """
        Reconstruct a vector from remote index.

        Args:
            client: The FAISSx client
            i: Vector index

        Returns:
            Reconstructed vector
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
                raise RuntimeError(f"Invalid reconstruct response from server: {response}")
        except Exception as e:
            logger.error(f"Error reconstructing vector from remote index: {e}")
            raise RuntimeError(f"Failed to reconstruct vector from remote index: {e}")

    def reconstruct_n(self, i0: int, ni: int) -> np.ndarray:
        """
        Reconstruct a batch of vectors from the index.

        Args:
            i0: First index to reconstruct
            ni: Number of vectors to reconstruct

        Returns:
            Reconstructed vectors as array of shape (ni, d)

        Raises:
            IndexError: If any index is out of bounds
        """
        # Register access for memory management
        self.register_access()

        if i0 < 0 or i0 + ni > self.ntotal:
            raise IndexError(f"Index range [{i0}, {i0+ni}) is out of bounds [0, {self.ntotal})")

        client = get_client()

        if client is not None and client.mode == "remote":
            return self._reconstruct_n_remote(client, i0, ni)
        else:
            return self._reconstruct_n_local(i0, ni)

    def _reconstruct_n_local(self, i0: int, ni: int) -> np.ndarray:
        """
        Reconstruct a range of vectors from local index.

        Args:
            i0: First index
            ni: Number of vectors

        Returns:
            Array of reconstructed vectors
        """
        logger.debug(f"Reconstructing vectors {i0}:{i0+ni} from local index {self.name}")
        return self._local_index.reconstruct_n(i0, ni)

    def _reconstruct_n_remote(self, client: Any, i0: int, ni: int) -> np.ndarray:
        """
        Reconstruct a range of vectors from remote index.

        Args:
            client: The FAISSx client
            i0: First index
            ni: Number of vectors

        Returns:
            Array of reconstructed vectors
        """
        logger.debug(f"Reconstructing vectors {i0}:{i0+ni} from remote index {self.index_id}")

        # Get the batch size for reconstruct operations
        batch_size = self.get_parameter('reconstruct_batch_size')
        if batch_size <= 0:
            batch_size = 100  # Default batch size

        # For small batches, use a direct call
        if ni <= batch_size:
            return self._reconstruct_n_remote_batch(client, i0, ni)

        # For larger batches, break up into smaller pieces
        result = []

        for i in range(i0, i0 + ni, batch_size):
            current_ni = min(batch_size, i0 + ni - i)
            batch_result = self._reconstruct_n_remote_batch(client, i, current_ni)
            result.append(batch_result)

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
                raise RuntimeError(f"Invalid reconstruct_n response from server: {response}")
        except Exception as e:
            logger.error(f"Error reconstructing vectors from remote index: {e}")
            raise RuntimeError(f"Failed to reconstruct vectors from remote index: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Clean up resources."""
        if self._use_gpu and self._gpu_resources is not None:
            # Clean up GPU resources
            self._gpu_resources = None
        self._local_index = None

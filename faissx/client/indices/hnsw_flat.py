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
FAISSx IndexHNSWFlat implementation.

This module provides a client-side implementation of the FAISS IndexHNSWFlat class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).
"""

import uuid
import numpy as np
from typing import Tuple, Any, Optional

from ..client import get_client
from .base import logger, FAISSxBaseIndex


# Create a class to hold HNSW parameters with the same API as FAISS
class HNSWParameters:
    """
    Proxy for FAISS HNSW parameters.

    This class mimics the behavior of FAISS HNSW parameter access.
    """
    def __init__(self, parent_index):
        self.parent_index = parent_index
        self._efSearch = 16  # Default in FAISS
        self._efConstruction = 40  # Default in FAISS
        self._M = parent_index.M if hasattr(parent_index, 'M') else 32  # Should match parent's M

    @property
    def efSearch(self):
        """Get the efSearch parameter"""
        if not self.parent_index._using_remote and self.parent_index._local_index is not None:
            # Pass through to the real FAISS index if in local mode
            return self.parent_index._local_index.hnsw.efSearch
        return self._efSearch

    @efSearch.setter
    def efSearch(self, value):
        """Set the efSearch parameter"""
        self._efSearch = value
        if not self.parent_index._using_remote and self.parent_index._local_index is not None:
            self.parent_index._local_index.hnsw.efSearch = value

    @property
    def efConstruction(self):
        """Get the efConstruction parameter"""
        if not self.parent_index._using_remote and self.parent_index._local_index is not None:
            # Pass through to the real FAISS index if in local mode
            return self.parent_index._local_index.hnsw.efConstruction
        return self._efConstruction

    @efConstruction.setter
    def efConstruction(self, value):
        """Set the efConstruction parameter"""
        self._efConstruction = value
        if not self.parent_index._using_remote and self.parent_index._local_index is not None:
            self.parent_index._local_index.hnsw.efConstruction = value

    @property
    def M(self):
        """Get the M parameter (read-only like in FAISS)"""
        # Simply return the M value from the parent index
        # This is more reliable than trying to access it through the local index
        return self.parent_index.M


class IndexHNSWFlat(FAISSxBaseIndex):
    """
    Proxy implementation of FAISS IndexHNSWFlat.

    This class mimics the behavior of FAISS IndexHNSWFlat, which uses Hierarchical
    Navigable Small World graphs for efficient approximate similarity search. It offers
    excellent search performance with good accuracy, particularly for high-dimensional
    data.

    When running in local mode with CUDA-capable GPUs, it will automatically use
    GPU acceleration if available.

    Attributes:
        d (int): Vector dimension
        M (int): Number of connections per node in the HNSW graph
        metric_type (str): Distance metric type ('L2' or 'IP')
        is_trained (bool): Always True for HNSW index
        ntotal (int): Total number of vectors in the index
        name (str): Unique identifier for the index
        index_id (str): Server-side index identifier (when in remote mode)
        _vector_mapping (dict): Maps local indices to server indices (remote mode only)
        _next_idx (int): Next available local index (remote mode only)
        _local_index: Local FAISS index (local mode only)
        _using_remote (bool): Whether we're using remote or local implementation
        _gpu_resources: GPU resources if using GPU (local mode only)
        _use_gpu (bool): Whether we're using GPU acceleration (local mode only)
        hnsw: Access to HNSW-specific parameters
    """

    def __init__(self, d: int, M: int = 32, metric=None):
        """
        Initialize the HNSW index with specified parameters.

        Args:
            d (int): Vector dimension
            M (int): Number of connections per node (higher = better accuracy, more memory)
            metric: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        # Initialize base class
        super().__init__()

        # Try to import faiss locally to avoid module-level dependency
        try:
            import faiss as local_faiss
            METRIC_L2 = local_faiss.METRIC_L2
            METRIC_INNER_PRODUCT = local_faiss.METRIC_INNER_PRODUCT
        except ImportError:
            # Define fallback constants when faiss isn't available
            METRIC_L2 = 0
            METRIC_INNER_PRODUCT = 1
            local_faiss = None

        # Set default metric if not provided
        if metric is None:
            metric = METRIC_L2

        # Store core parameters
        self.d = d
        self.M = M
        # Convert metric type to string representation for remote mode
        self.metric_type = "IP" if metric == METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained = True  # HNSW doesn't need training
        self.ntotal = 0

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None
        self._local_index = None

        # Generate unique name for the index
        self.name = f"index-hnsw-flat-{uuid.uuid4().hex[:8]}"
        self.index_id = self.name

        # Initialize vector mapping for remote mode
        self._vector_mapping = {}  # Maps server-side indices to local indices for faster lookup
        self._next_idx = 0  # Counter for local indices

        # Check if client exists and its mode
        client = get_client()

        if client is not None and client.mode == "remote":
            # Remote mode
            logger.info(f"Creating remote IndexHNSWFlat on server {client.server}")
            self._create_remote_index(client, d, M)
        else:
            # Local mode
            logger.info(f"Creating local IndexHNSWFlat index {self.name}")
            self._create_local_index(d, M, metric)

        # Initialize hnsw property
        self.hnsw = HNSWParameters(self)

    def _get_index_type_string(self, M: Optional[int] = None) -> str:
        """
        Get standardized string representation of index type.

        Args:
            M: Connections per layer parameter to use instead of self.M

        Returns:
            String representation of index type
        """
        # Use internal M if none provided
        if M is None:
            M = self.M

        # Create base type string
        index_type = f"HNSW{M}"

        # Add metric type suffix if needed
        if self.metric_type == "IP":
            index_type = f"{index_type}_IP"

        return index_type

    def _create_local_index(self, d: int, M: int, metric: Any) -> None:
        """
        Create a local FAISS HNSW index.

        Args:
            d (int): Vector dimension
            M (int): Connections per layer parameter
            metric: Distance metric type
        """
        try:
            import faiss

            # Try to use GPU if available
            gpu_available = False
            try:
                import faiss.contrib.gpu  # type: ignore

                ngpus = faiss.get_num_gpus()
                gpu_available = ngpus > 0
            except (ImportError, AttributeError) as e:
                logger.warning(f"GPU support not available: {e}")
                gpu_available = False

            if gpu_available:
                # GPU is available, create resources and GPU index
                self._use_gpu = True
                self._gpu_resources = faiss.StandardGpuResources()

                # Create CPU index first (HNSW has limited GPU operations)
                cpu_index = faiss.IndexHNSWFlat(d, M, metric)

                # Convert to GPU index - note that many HNSW operations still run on CPU
                try:
                    self._local_index = faiss.index_cpu_to_gpu(
                        self._gpu_resources, 0, cpu_index
                    )
                    logger.info(f"Using GPU-accelerated HNSW index for {self.name} (search only)")
                except Exception as e:
                    # If GPU conversion fails, fall back to CPU
                    self._local_index = cpu_index
                    self._use_gpu = False
                    logger.warning(
                        f"Failed to create GPU HNSW index: {e}, using CPU instead"
                    )
            else:
                # No GPUs available, use CPU version
                self._local_index = faiss.IndexHNSWFlat(d, M, metric)

            self.index_id = self.name  # Use name as ID for consistency
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")

    def _create_remote_index(self, client: Any, d: int, M: int) -> None:
        """
        Create a remote HNSW index on the server.

        Args:
            client: FAISSx client instance
            d (int): Vector dimension
            M (int): Connections per layer parameter
        """
        try:
            # Get index type string
            index_type = self._get_index_type_string(M)

            # Create index on server
            logger.debug(f"Creating remote index {self.name} with type {index_type}")
            response = client.create_index(self.name, d, index_type)

            # Parse response
            if isinstance(response, dict):
                self.index_id = response.get("index_id", self.name)
            else:
                logger.warning(f"Unexpected server response format: {response}")
                self.index_id = self.name

        except Exception as e:
            raise RuntimeError(
                f"Failed to create remote HNSW index: {e}. "
                f"Server may not support HNSW indices with type {self._get_index_type_string(M)}."
            )

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Prepare vectors for indexing or search.

        Args:
            vectors: Input vectors as numpy array

        Returns:
            Normalized array with proper dtype
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        # Convert to float32 if needed (FAISS requirement)
        return vectors.astype(np.float32) if vectors.dtype != np.float32 else vectors

    def _map_server_to_local_indices(self, server_indices: list) -> np.ndarray:
        """
        Convert server-side indices to local indices.

        Args:
            server_indices: List of server-side indices

        Returns:
            Array of corresponding local indices, -1 for not found
        """
        local_indices = np.full(len(server_indices), -1, dtype=np.int64)

        for i, server_idx in enumerate(server_indices):
            for local_idx, info in self._vector_mapping.items():
                if info.get("server_idx") == server_idx:
                    local_indices[i] = local_idx
                    break

        return local_indices

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension
        """
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Prepare vectors
        vectors = self._prepare_vectors(x)

        client = get_client()

        if client is not None and client.mode == "remote":
            self._add_remote(client, vectors)
        else:
            self._add_local(vectors)

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
        Add vectors to remote index with batch processing.

        Args:
            client: The FAISSx client
            vectors: Vectors to add
        """
        logger.debug(f"Adding {len(vectors)} vectors to remote index {self.index_id}")

        # Get the batch size for adding vectors
        try:
            batch_size = self.get_parameter('batch_size')
        except ValueError:
            batch_size = 1000  # Default batch size

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

            # Update tracking for added vectors
            if isinstance(response, dict) and response.get("success", False):
                added_count = response.get("count", 0)

                # Create mapping for each added vector
                for i in range(added_count):
                    self._vector_mapping[self._next_idx] = {
                        "local_idx": self._next_idx,
                        "server_idx": self.ntotal + i,
                    }
                    self._next_idx += 1

                self.ntotal += added_count
            else:
                # Handle error or unexpected response
                error_msg = (response.get("error", "Unknown server error")
                             if isinstance(response, dict) else "Invalid response format")
                logger.warning(f"Error adding vectors: {error_msg}")
                raise RuntimeError(f"Failed to add vectors: {error_msg}")

        except Exception as e:
            logger.error(f"Error adding vectors to remote index: {e}")
            raise RuntimeError(f"Failed to add vectors to remote index: {e}")

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
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Prepare vectors
        query_vectors = self._prepare_vectors(x)

        # For HNSW, we can sometimes get better results by requesting more neighbors
        # than we actually need, then re-ranking them
        try:
            need_reranking = self.get_parameter('rerank_results')
        except ValueError:
            need_reranking = False

        try:
            search_factor = self.get_parameter('search_factor')
        except ValueError:
            search_factor = 1.0

        internal_k = int(k * search_factor) if need_reranking else k

        client = get_client()

        if client is not None and client.mode == "remote":
            return self._search_remote(client, query_vectors, k, internal_k, need_reranking)
        else:
            return self._search_local(query_vectors, k, internal_k, need_reranking)

    def _search_local(
        self, query_vectors: np.ndarray, k: int, internal_k: int, need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using local FAISS index.

        Args:
            query_vectors: Prepared query vectors
            k: Number of results requested
            internal_k: Internal search width (may be larger than k)
            need_reranking: Whether to rerank results

        Returns:
            Tuple of (distances, indices)
        """
        logger.debug(f"Searching local index {self.name} for {len(query_vectors)} queries, k={k}")

        # Use the actual k value - reranking would be done by FAISS internally
        distances, indices = self._local_index.search(query_vectors, k)
        return distances, indices

    def _search_remote(
        self, client: Any, query_vectors: np.ndarray, k: int,
        internal_k: int, need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using remote index with batch processing.

        Args:
            client: FAISSx client
            query_vectors: Prepared query vectors
            k: Number of results requested
            internal_k: Internal search width
            need_reranking: Whether to rerank results

        Returns:
            Tuple of (distances, indices)
        """
        logger.debug(f"Searching remote index {self.index_id} for {len(query_vectors)} queries, k={k}")

        # Get batch size for search operations
        try:
            batch_size = self.get_parameter('search_batch_size')
        except ValueError:
            batch_size = 100  # Default batch size for search

        # If queries fit in a single batch, search directly
        if len(query_vectors) <= batch_size:
            return self._search_remote_batch(
                client, query_vectors, k, internal_k, need_reranking
            )

        # Process in batches
        n = len(query_vectors)
        all_distances = np.full((n, k), float("inf"), dtype=np.float32)
        all_indices = np.full((n, k), -1, dtype=np.int64)

        for i in range(0, n, batch_size):
            batch = query_vectors[i:min(i + batch_size, n)]
            batch_distances, batch_indices = self._search_remote_batch(
                client, batch, k, internal_k, need_reranking
            )

            # Copy batch results to output arrays
            batch_size_actual = len(batch)
            all_distances[i:i+batch_size_actual] = batch_distances
            all_indices[i:i+batch_size_actual] = batch_indices

        return all_distances, all_indices

    def _search_remote_batch(
        self, client: Any, query_vectors: np.ndarray, k: int,
        internal_k: int, need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search a batch of queries using remote index.

        Args:
            client: FAISSx client
            query_vectors: Batch of query vectors
            k: Number of results requested
            internal_k: Internal search width
            need_reranking: Whether to rerank results

        Returns:
            Tuple of (distances, indices)
        """
        try:
            # Set up search parameters
            # Uncomment and use if needed for specific HNSW parameters
            # params = {
            #     "efSearch": self.hnsw.efSearch,
            # }

            # Request more results if reranking is needed
            actual_k = internal_k if need_reranking else k

            # Perform search
            result = client.search(
                self.index_id,
                query_vectors=query_vectors,
                k=actual_k,
                # params=params
            )

            # Check for errors
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                raise RuntimeError(f"Search failed: {error}")

            n = len(query_vectors)  # Number of query vectors
            search_results = result.get("results", [])

            # Initialize output arrays
            distances = np.full((n, k), float("inf"), dtype=np.float32)
            indices = np.full((n, k), -1, dtype=np.int64)

            # Process results for each query vector
            for i in range(min(n, len(search_results))):
                result_data = search_results[i]
                result_distances = result_data.get("distances", [])
                result_indices = result_data.get("indices", [])

                # Rerank if needed
                if need_reranking and len(result_distances) > k:
                    # Simple reranking by distance (server should return sorted results)
                    result_distances = result_distances[:k]
                    result_indices = result_indices[:k]

                # Fill in results for this query vector
                for j in range(min(k, len(result_distances))):
                    distances[i, j] = result_distances[j]

                    # Map server index back to local index
                    server_idx = result_indices[j]
                    for local_idx, info in self._vector_mapping.items():
                        if info.get("server_idx") == server_idx:
                            indices[i, j] = local_idx
                            break

            return distances, indices

        except Exception as e:
            logger.error(f"Error during remote search: {e}")
            raise RuntimeError(f"Search failed: {e}")

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
                - indices: array of shape (sum_of_results) containing all indices

        Raises:
            ValueError: If query vector shape doesn't match index dimension
            RuntimeError: If range search fails or isn't supported by the index type
        """
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Prepare vectors
        query_vectors = self._prepare_vectors(x)

        client = get_client()

        if client is not None and client.mode == "remote":
            return self._range_search_remote(client, query_vectors, radius)
        else:
            return self._range_search_local(query_vectors, radius)

    def _range_search_local(
        self, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Range search using local FAISS index.

        Args:
            query_vectors: Prepared query vectors
            radius: Distance threshold

        Returns:
            Tuple of (lims, distances, indices)
        """
        logger.debug(f"Range searching local index {self.name} with radius={radius}")

        if hasattr(self._local_index, "range_search"):
            return self._local_index.range_search(query_vectors, radius)
        else:
            raise RuntimeError("Local FAISS index does not support range_search")

    def _range_search_remote(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Range search using remote index with batch processing.

        Args:
            client: FAISSx client
            query_vectors: Prepared query vectors
            radius: Distance threshold

        Returns:
            Tuple of (lims, distances, indices)
        """
        logger.debug(f"Range searching remote index {self.index_id} with radius={radius}")

        # Get batch size for range search operations
        try:
            batch_size = self.get_parameter('search_batch_size')
        except ValueError:
            batch_size = 100  # Default batch size

        # Perform search using batch approach
        if len(query_vectors) <= batch_size:
            return self._range_search_remote_batch(
                client, query_vectors, radius
            )

        # Process in batches
        n_queries = len(query_vectors)
        all_results = []

        for i in range(0, n_queries, batch_size):
            batch = query_vectors[i:min(i + batch_size, n_queries)]
            try:
                batch_result = client.range_search(self.index_id, batch, radius)

                if not batch_result.get("success", False):
                    error = batch_result.get("error", "Unknown error")
                    raise RuntimeError(f"Range search failed: {error}")

                all_results.extend(batch_result.get("results", []))
            except Exception as e:
                logger.error(f"Range search batch {i//batch_size} failed: {e}")
                raise

        # Calculate total number of results across all queries
        total_results = sum(
            res.get("count", 0) for res in all_results
        )

        # Initialize arrays
        lims = np.zeros(n_queries + 1, dtype=np.int64)
        distances = np.zeros(total_results, dtype=np.float32)
        indices = np.zeros(total_results, dtype=np.int64)

        # Fill arrays with results
        offset = 0
        for i, res in enumerate(all_results):
            # Set limit boundary for this query
            lims[i] = offset

            # Get results for this query
            result_distances = res.get("distances", [])
            result_indices = res.get("indices", [])
            count = len(result_distances)

            # Copy data to output arrays
            if count > 0:
                distances[offset:offset + count] = np.array(result_distances, dtype=np.float32)

                # Map server indices back to local indices
                mapped_indices = self._map_server_to_local_indices(result_indices)
                indices[offset:offset + count] = mapped_indices
                offset += count

        # Set final boundary
        lims[n_queries] = offset

        return lims, distances, indices

    def _range_search_remote_batch(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Range search a batch of queries using remote index.

        Args:
            client: FAISSx client
            query_vectors: Batch of query vectors
            radius: Distance threshold

        Returns:
            Tuple of (lims, distances, indices)
        """
        try:
            # Set up search parameters
            # Uncomment and use if needed for specific HNSW parameters
            # params = {
            #     "efSearch": self.hnsw.efSearch,
            # }

            # Perform range search
            result = client.range_search(self.index_id, query_vectors, radius)

            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                raise RuntimeError(f"Range search failed: {error}")

            # Process results
            search_results = result.get("results", [])
            n_queries = len(search_results)

            # Calculate total number of results
            total_results = sum(
                res.get("count", 0) for res in search_results
            )

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
                result_indices = res.get("indices", [])
                count = len(result_distances)

                # Copy data to output arrays
                if count > 0:
                    distances[offset:offset + count] = np.array(result_distances, dtype=np.float32)

                    # Map server indices back to local indices
                    mapped_indices = self._map_server_to_local_indices(result_indices)
                    indices[offset:offset + count] = mapped_indices
                    offset += count

            # Set final boundary
            lims[n_queries] = offset

            return lims, distances, indices

        except Exception as e:
            logger.error(f"Error during remote range search: {e}")
            raise RuntimeError(f"Range search failed: {e}")

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
        """Reset local index"""
        if hasattr(self._local_index, "reset"):
            self._local_index.reset()
            self.ntotal = 0
        else:
            # Recreate the index if reset is not supported
            try:
                import faiss
                metric = faiss.METRIC_INNER_PRODUCT if self.metric_type == "IP" else faiss.METRIC_L2
                self._create_local_index(self.d, self.M, metric)
                self.ntotal = 0
            except Exception as e:
                logger.error(f"Error resetting local index: {e}")
                raise RuntimeError(f"Failed to reset index: {e}")

    def _reset_remote(self, client: Any) -> None:
        """
        Reset remote index by creating a new one.

        Args:
            client: FAISSx client
        """
        try:
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            index_type = self._get_index_type_string()

            response = client.create_index(
                name=new_name, dimension=self.d, index_type=index_type
            )

            # Handle different response formats
            if isinstance(response, dict):
                self.index_id = response.get("index_id", new_name)
            else:
                # For string responses, use the name directly
                logger.debug(f"Got string response: {response}")
                self.index_id = new_name

            self.name = new_name
            logger.debug(f"Successfully created new index: {self.index_id}")

        except Exception as e:
            logger.warning(f"Failed to create new index during reset: {e}. Trying alternative method.")

            # Try a different approach - create with a completely unique name
            try:
                # Generate a totally unique name
                unique_name = f"index-hnsw-{uuid.uuid4().hex[:12]}"
                index_type = self._get_index_type_string()

                logger.debug(f"Attempting to create index with unique name: {unique_name}")
                response = client.create_index(
                    name=unique_name, dimension=self.d, index_type=index_type
                )

                if isinstance(response, dict):
                    self.index_id = response.get("index_id", unique_name)
                else:
                    # For string responses, use the name directly
                    self.index_id = unique_name

                self.name = unique_name
                logger.debug(f"Successfully created alternative index: {self.index_id}")

            except Exception as e2:
                logger.error(f"Failed all reset attempts: {e2}")
                raise RuntimeError(f"Failed to reset index: {e2}")

        # Reset all local state
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def close(self) -> None:
        """Clean up resources."""
        # Clean up GPU resources if used
        if self._use_gpu and self._gpu_resources is not None:
            self._gpu_resources = None

        # Local index will be cleaned up by garbage collector
        pass

    def __del__(self) -> None:
        """Clean up when the object is deleted."""
        self.close()

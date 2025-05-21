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
FAISSx IndexIVFFlat implementation.

This module provides a client-side implementation of the FAISS IndexIVFFlat class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).
"""

import uuid
import numpy as np
from typing import Tuple, Any

from ..client import get_client
from .base import logger, FAISSxBaseIndex

# Import needed for type hints
from .flat import IndexFlatL2


class IndexIVFFlat(FAISSxBaseIndex):
    """
    Proxy implementation of FAISS IndexIVFFlat.

    This class mimics the behavior of FAISS IndexIVFFlat, which uses inverted file
    indexing for efficient similarity search. It divides the vector space into partitions
    (clusters) for faster search, requiring a training step before use.

    When running in local mode with CUDA-capable GPUs, it will automatically use
    GPU acceleration if available.

    Attributes:
        d (int): Vector dimension
        nlist (int): Number of clusters/partitions
        metric_type (str): Distance metric type ('L2' or 'IP')
        is_trained (bool): Whether the index has been trained
        ntotal (int): Total number of vectors in the index
        name (str): Unique identifier for the index
        index_id (str): Server-side index identifier (when in remote mode)
        _vector_mapping (dict): Maps local indices to server indices (remote mode only)
        _next_idx (int): Next available local index (remote mode only)
        _local_index: Local FAISS index (local mode only)
        _gpu_resources: GPU resources if using GPU (local mode only)
        _use_gpu (bool): Whether we're using GPU acceleration (local mode only)
        _nprobe (int): Number of clusters to search (default: 1)
    """

    def __init__(self, quantizer, d: int, nlist: int, metric_type=None):
        """
        Initialize the inverted file index with specified parameters.

        Args:
            quantizer: Quantizer object that defines the centroids (usually IndexFlatL2)
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions
            metric_type: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        super().__init__()  # Initialize base class

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

        # Use default metric if not provided
        if metric_type is None:
            metric_type = METRIC_L2

        # Store core parameters
        self.d = d
        self.nlist = nlist
        # Convert metric type to string representation for remote mode
        self.metric_type = "IP" if metric_type == METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained = False
        self.ntotal = 0
        self._nprobe = 1  # Default number of probes

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None
        self._local_index = None

        # Generate unique name for the index
        self.name = f"index-ivf-flat-{uuid.uuid4().hex[:8]}"
        self.index_id = self.name

        # Initialize vector mapping for remote mode
        self._vector_mapping = {}  # Maps local indices to server-side information
        self._next_idx = 0  # Counter for local indices

        # Check if client exists and its mode
        client = get_client()

        # Explicit check for remote mode instead of just checking if client exists
        if client is not None and client.mode == "remote":
            # Remote mode
            logger.info(f"Creating remote IndexIVFFlat on server {client.server}")
            self._create_remote_index(client, quantizer, d, nlist)
        else:
            # Local mode
            logger.info(f"Creating local IndexIVFFlat index {self.name}")
            self._create_local_index(quantizer, d, nlist, metric_type)

    def _get_index_type_string(self) -> str:
        """
        Get standardized string representation of the IVF index type.

        Returns:
            String representation of index type
        """
        # Create the index type string based on nlist and metric type
        index_type = f"IVF{self.nlist}"

        # Add metric type suffix if needed
        if self.metric_type == "IP":
            index_type = f"{index_type}_IP"

        return index_type

    def _parse_server_response(self, response: Any, default_value: Any) -> Any:
        """
        Parse server response with consistent error handling.

        Args:
            response: Server response to parse
            default_value: Default value to use if response isn't a dict

        Returns:
            Parsed value from response or default value
        """
        if isinstance(response, dict):
            return response.get("index_id", default_value)
        else:
            logger.warning(f"Unexpected server response format: {response}")
            return default_value

    def _create_local_index(self, quantizer, d: int, nlist: int, metric_type: Any) -> None:
        """
        Create a local FAISS inverted file index.

        Args:
            quantizer: Quantizer object that defines the centroids
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions
            metric_type: Distance metric type
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

                # Create CPU index first
                if isinstance(quantizer, IndexFlatL2) and getattr(quantizer, "_use_gpu", False):
                    # If the quantizer is already on GPU, get the CPU version
                    cpu_quantizer = faiss.index_gpu_to_cpu(quantizer._local_index)
                else:
                    # Otherwise, use the provided quantizer directly
                    cpu_quantizer = (
                        quantizer._local_index
                        if hasattr(quantizer, "_local_index")
                        else quantizer
                    )

                # Create CPU index
                cpu_index = faiss.IndexIVFFlat(cpu_quantizer, d, nlist, metric_type)

                # Convert to GPU index
                try:
                    self._local_index = faiss.index_cpu_to_gpu(
                        self._gpu_resources, 0, cpu_index
                    )
                    logger.info(f"Using GPU-accelerated IVF index for {self.name}")
                except Exception as e:
                    # If GPU conversion fails, fall back to CPU
                    self._local_index = cpu_index
                    self._use_gpu = False
                    logger.warning(
                        f"Failed to create GPU IVF index: {e}, using CPU instead"
                    )
            else:
                # No GPUs available, use CPU version
                self._local_index = faiss.IndexIVFFlat(
                    (
                        quantizer._local_index
                        if hasattr(quantizer, "_local_index")
                        else quantizer
                    ),
                    d,
                    nlist,
                    metric_type,
                )

            self.index_id = self.name  # Use name as ID for consistency
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local FAISS IVF index: {e}")

    def _create_remote_index(self, client: Any, quantizer, d: int, nlist: int) -> None:
        """
        Create a remote IVF index on the server.

        Args:
            client: FAISSx client instance
            quantizer: Quantizer object that defines the centroids
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions
        """
        try:
            # Get index type string
            index_type = self._get_index_type_string()

            # Create index on server
            logger.debug(f"Creating remote index {self.name} with type {index_type}")
            response = client.create_index(
                name=self.name, dimension=d, index_type=index_type
            )

            # Log the raw response for debugging
            logger.debug(f"Server response: {response}")

            # Parse response to get index ID
            self.index_id = self._parse_server_response(response, self.name)
            logger.info(f"Created remote index with ID: {self.index_id}")
        except Exception as e:
            # Raise a clear error instead of falling back to local mode
            raise RuntimeError(
                f"Failed to create remote IVF index: {e}. "
                f"Server may not support IVF indices with type {index_type}."
            )

    # Add nprobe property getter and setter to handle it as an attribute
    @property
    def nprobe(self):
        """Get the current nprobe value"""
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value):
        """Set the nprobe value and update the local index if present"""
        self.set_nprobe(value)

    def train(self, x: np.ndarray) -> None:
        """
        Train the index with the provided vectors.

        Args:
            x (np.ndarray): Training vectors, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or already trained
            RuntimeError: If remote training operation fails
        """
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        client = get_client()

        # Explicit check for remote mode
        if client is not None and client.mode == "remote":
            self._train_remote(client, vectors)
        else:
            self._train_local(vectors)

    def _train_local(self, vectors: np.ndarray) -> None:
        """Train the local index with the provided vectors."""
        logger.debug(f"Training local index {self.name} with {len(vectors)} vectors")

        # Use local FAISS implementation directly
        self._local_index.train(vectors)
        self.is_trained = self._local_index.is_trained

    def _train_remote(self, client: Any, vectors: np.ndarray) -> None:
        """Train the remote index with the provided vectors."""
        logger.debug(f"Training remote index {self.index_id} with {len(vectors)} vectors")

        try:
            result = client.train_index(self.index_id, vectors)

            # Check for explicit error response
            if not isinstance(result, dict) or not result.get("success", False):
                error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                raise RuntimeError(f"Remote training failed: {error_msg}")

            # Update local state based on training result
            self.is_trained = result.get("is_trained", True)
        except Exception as e:
            # Ensure all errors are properly propagated
            raise RuntimeError(f"Remote training operation failed: {e}")

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or index not trained
            RuntimeError: If remote add operation fails
        """
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        client = get_client()

        # Explicit check for remote mode
        if client is not None and client.mode == "remote":
            self._add_remote(client, vectors)
        else:
            self._add_local(vectors)

    def _add_local(self, vectors: np.ndarray) -> None:
        """Add vectors to local index."""
        logger.debug(f"Adding {len(vectors)} vectors to local index {self.name}")

        # Use local FAISS implementation directly
        self._local_index.add(vectors)
        self.ntotal = self._local_index.ntotal

    def _add_remote(self, client: Any, vectors: np.ndarray) -> None:
        """Add vectors to remote index."""
        logger.debug(f"Adding {len(vectors)} vectors to remote index {self.index_id}")

        # Get batch size parameter
        batch_size = self.get_parameter('batch_size')
        if batch_size <= 0:
            batch_size = 1000  # Default if not set or invalid

        # If vectors fit in a single batch, add them directly
        if len(vectors) <= batch_size:
            self._add_remote_batch(client, vectors)
            return

        # Otherwise, process in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:min(i+batch_size, len(vectors))]
            self._add_remote_batch(client, batch)

    def _add_remote_batch(self, client: Any, vectors: np.ndarray) -> None:
        """Add a batch of vectors to the remote index."""
        try:
            result = client.add_vectors(self.index_id, vectors)

            # Log response
            logger.debug(f"Server response: {result}")

            # Update local tracking if addition was successful
            if isinstance(result, dict) and result.get("success", False):
                added_count = result.get("count", 0)
                # Create mapping for each added vector
                for i in range(added_count):
                    local_idx = self._next_idx
                    server_idx = self.ntotal + i
                    self._vector_mapping[local_idx] = {
                        "local_idx": local_idx,
                        "server_idx": server_idx,
                    }
                    self._next_idx += 1

                self.ntotal += added_count
            elif not isinstance(result, dict):
                # Handle non-dict responses
                logger.warning(f"Unexpected response format from server: {result}")
                # Assume we added all vectors as a fallback
                for i in range(len(vectors)):
                    local_idx = self._next_idx
                    server_idx = self.ntotal + i
                    self._vector_mapping[local_idx] = {
                        "local_idx": local_idx,
                        "server_idx": server_idx,
                    }
                    self._next_idx += 1

                self.ntotal += len(vectors)
        except Exception as e:
            # Ensure all errors are properly propagated
            raise RuntimeError(f"Remote add operation failed: {e}")

    def set_nprobe(self, nprobe: int) -> None:
        """
        Set the number of clusters to visit during search (nprobe).

        Higher values of nprobe will give more accurate results at the cost of
        slower search. For IVF indices, nprobe should be between 1 and nlist.

        Args:
            nprobe (int): Number of clusters to search (between 1 and nlist)

        Raises:
            ValueError: If nprobe is less than 1 or greater than nlist
        """
        # Register access for memory management
        self.register_access()

        if nprobe < 1:
            raise ValueError(f"nprobe must be at least 1, got {nprobe}")
        if nprobe > self.nlist:
            raise ValueError(
                f"nprobe must not exceed nlist ({self.nlist}), got {nprobe}"
            )

        self._nprobe = nprobe

        # If using local implementation, update the index directly
        client = get_client()
        if client is None or client.mode == "local":
            if self._local_index is not None:
                self._local_index.nprobe = nprobe

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
            RuntimeError: If index is not trained or remote operation fails
        """
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Get k_factor parameter for oversampling, if set
        k_factor = self.get_parameter('k_factor')
        if k_factor <= 1.0:
            k_factor = 1.0

        # Calculate internal_k with k_factor and clamp to ntotal
        internal_k = min(int(k * k_factor), max(1, self.ntotal))
        need_reranking = (k_factor > 1.0 and internal_k > k)

        client = get_client()

        # Explicit check for remote mode
        if client is not None and client.mode == "remote":
            return self._search_remote(client, query_vectors, k, internal_k, need_reranking)
        else:
            return self._search_local(query_vectors, k, internal_k, need_reranking)

    def _search_local(self, query_vectors: np.ndarray, k: int, internal_k: int,
                     need_reranking: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Search in local index."""
        logger.debug(f"Searching {len(query_vectors)} vectors in local index {self.name}")

        # Set nprobe for local index before searching
        self._local_index.nprobe = self._nprobe

        # Use local FAISS implementation directly
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
        """Search in remote index."""
        logger.debug(f"Searching {len(query_vectors)} vectors in remote index {self.index_id}")

        # Get batch size parameter
        batch_size = self.get_parameter('batch_size')
        if batch_size <= 0:
            batch_size = 100  # Default if not set or invalid

        # If queries fit in a single batch, search directly
        if len(query_vectors) <= batch_size:
            return self._search_remote_batch(client, query_vectors, k, internal_k, need_reranking)

        # Otherwise, process in batches and combine results
        all_distances = []
        all_indices = []

        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:min(i+batch_size, len(query_vectors))]
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
        """Search a batch of queries in the remote index."""
        try:
            # Include nprobe parameter in the search request
            result = client.search(
                self.index_id,
                query_vectors=query_vectors,
                k=internal_k,
                params={"nprobe": self._nprobe}  # Send nprobe parameter to server
            )

            # Log response
            logger.debug(f"Server response: {result}")

            n = query_vectors.shape[0]  # Number of query vectors

            # Initialize output arrays with default values
            distances = np.full((n, k), float("inf"), dtype=np.float32)
            indices = np.full((n, k), -1, dtype=np.int64)

            # Process results based on response format
            if not isinstance(result, dict) or "results" not in result:
                logger.warning(f"Unexpected search response format: {result}")
                return distances, indices

            # Extract results list
            search_results = result["results"]
            if not isinstance(search_results, list):
                logger.warning(f"Invalid results format, expected list: {search_results}")
                return distances, indices

            # Process results for each query vector
            for i in range(min(n, len(search_results))):
                result_data = search_results[i]

                if not isinstance(result_data, dict):
                    logger.warning(f"Invalid result data format for query {i}: {result_data}")
                    continue

                result_distances = result_data.get("distances", [])
                result_indices = result_data.get("indices", [])

                # Number of results for this query
                num_results = min(k, len(result_distances))

                # Fill in results for this query vector
                for j in range(num_results):
                    distances[i, j] = result_distances[j]
                    server_idx = result_indices[j]

                    # Map server index back to local index
                    found = False
                    for local_idx, info in self._vector_mapping.items():
                        if info["server_idx"] == server_idx:
                            indices[i, j] = local_idx
                            found = True
                            break

                    # Keep -1 if mapping not found
                    if not found:
                        indices[i, j] = -1

            return distances, indices
        except Exception as e:
            # Ensure all errors are properly propagated
            raise RuntimeError(f"Remote search operation failed: {e}")

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
            RuntimeError: If range search fails or isn't supported in remote mode
        """
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        client = get_client()

        # Explicit check for remote mode
        if client is not None and client.mode == "remote":
            return self._range_search_remote(client, query_vectors, radius)
        else:
            return self._range_search_local(query_vectors, radius)

    def _range_search_local(
        self, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Range search in local index."""
        logger.debug(f"Range searching {len(query_vectors)} vectors in local index {self.name}")

        # Set nprobe for local index before searching
        self._local_index.nprobe = self._nprobe

        # Use local FAISS implementation directly
        if hasattr(self._local_index, "range_search"):
            return self._local_index.range_search(query_vectors, radius)
        else:
            raise RuntimeError("Local FAISS index does not support range_search")

    def _range_search_remote(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Range search in remote index."""
        logger.debug(f"Range searching {len(query_vectors)} vectors in remote index {self.index_id}")

        # Get batch size parameter
        batch_size = self.get_parameter('batch_size')
        if batch_size <= 0:
            batch_size = 100  # Default if not set or invalid

        # If queries fit in a single batch, search directly
        if len(query_vectors) <= batch_size:
            return self._range_search_remote_batch(client, query_vectors, radius)

        # Otherwise, process in batches and combine results
        all_lims = [0]  # Start with 0 for the first boundary
        all_distances = []
        all_indices = []

        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:min(i+batch_size, len(query_vectors))]
            lims, distances, indices = self._range_search_remote_batch(client, batch, radius)

            # Adjust lims to account for previously added results
            if len(all_distances) > 0:
                total_results = len(all_distances)
                # Skip the first element of lims (it's always 0)
                adjusted_lims = lims[1:] + total_results
                all_lims.extend(adjusted_lims)
            else:
                all_lims.extend(lims[1:])

            all_distances.extend(distances)
            all_indices.extend(indices)

        # Convert to numpy arrays
        return (
            np.array(all_lims, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            np.array(all_indices, dtype=np.int64),
        )

    def _range_search_remote_batch(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Range search a batch of queries in the remote index."""
        try:
            # Add nprobe parameter to the search request
            result = client.range_search(
                self.index_id,
                query_vectors,
                radius,
                params={"nprobe": self._nprobe}
            )

            # Log response
            logger.debug(f"Server response: {result}")

            if not isinstance(result, dict) or not result.get("success", False):
                error = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                raise RuntimeError(f"Range search failed in remote mode: {error}")

            # Process results
            search_results = result.get("results", [])
            n_queries = len(search_results)

            # Calculate total number of results across all queries
            total_results = sum(res.get("count", 0) for res in search_results)

            # Initialize arrays
            lims = np.zeros(n_queries + 1, dtype=np.int64)
            distances = []
            indices = []

            # Fill arrays with results
            offset = 0
            for i, res in enumerate(search_results):
                # Set limit boundary for this query
                lims[i] = offset

                # Get results for this query
                result_distances = res.get("distances", [])
                result_indices = res.get("indices", [])
                count = len(result_distances)

                # Add data to output arrays
                if count > 0:
                    distances.extend(result_distances)

                    # Map server indices back to local indices
                    for server_idx in result_indices:
                        found = False
                        for local_idx, info in self._vector_mapping.items():
                            if info["server_idx"] == server_idx:
                                indices.append(local_idx)
                                found = True
                                break

                        if not found:
                            indices.append(-1)

                    offset += count

            # Set final boundary
            lims[n_queries] = offset

            return lims, np.array(distances, dtype=np.float32), np.array(indices, dtype=np.int64)
        except Exception as e:
            # Ensure all errors are properly propagated
            raise RuntimeError(f"Range search failed in remote mode: {e}")

    def reset(self) -> None:
        """
        Reset the index to its initial state, removing all vectors but keeping training.

        This method removes all vectors from the index but preserves the training state.
        After calling reset(), you don't need to retrain the index.

        Raises:
            RuntimeError: If remote reset operation fails
        """
        # Register access for memory management
        self.register_access()

        # Remember if the index was trained before reset
        was_trained = self.is_trained

        client = get_client()

        # Explicit check for remote mode
        if client is not None and client.mode == "remote":
            self._reset_remote(client, was_trained)
        else:
            self._reset_local(was_trained)

    def _reset_local(self, was_trained: bool) -> None:
        """Reset the local index."""
        logger.debug(f"Resetting local index {self.name}")

        # Reset local FAISS index
        self._local_index.reset()
        self.ntotal = 0
        # Restore the trained state
        self.is_trained = was_trained

    def _reset_remote(self, client: Any, was_trained: bool) -> None:
        """Reset the remote index."""
        logger.debug(f"Resetting remote index {self.index_id}")

        try:
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            index_type = self._get_index_type_string()

            response = client.create_index(
                name=new_name, dimension=self.d, index_type=index_type
            )

            if not isinstance(response, dict) or not response.get("success", False):
                error_msg = response.get("error", "Unknown error") if isinstance(response, dict) else str(response)
                raise RuntimeError(f"Failed to create new index during reset: {error_msg}")

            # Update index information
            self.index_id = self._parse_server_response(response, new_name)
            self.name = new_name

            # Don't reset training state
            self.is_trained = was_trained

            # Reset all local state
            self.ntotal = 0
            self._vector_mapping = {}
            self._next_idx = 0
        except Exception as e:
            # Ensure all errors are properly propagated
            raise RuntimeError(f"Remote reset operation failed: {e}")

    def __enter__(self):
        """Support context manager interface."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        self.close()

    def close(self) -> None:
        """
        Release resources associated with this index.

        This method should be called when you're done using the index to free resources.
        """
        # Clean up GPU resources if used
        if self._use_gpu and self._gpu_resources is not None:
            self._gpu_resources = None
            self._use_gpu = False

        # Clear index to free memory
        self._local_index = None

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        self.close()

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
FAISSx IndexScalarQuantizer implementation.

This module provides a client-side implementation of the FAISS IndexScalarQuantizer class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).

Scalar Quantization compresses vectors by quantizing each dimension independently,
offering a good balance between memory usage and search accuracy.
"""

import uuid
import numpy as np
from typing import Tuple, Any, Dict, Optional, List

from ..client import get_client
from .base import logger, FAISSxBaseIndex


class IndexScalarQuantizer(FAISSxBaseIndex):
    """
    Proxy implementation of FAISS IndexScalarQuantizer.

    This class mimics the behavior of FAISS IndexScalarQuantizer, which uses scalar
    quantization for efficient memory usage while maintaining search accuracy. It's
    a good compromise between the high memory usage of flat indices and the lower
    precision of product quantization.

    When running in local mode with CUDA-capable GPUs, it will automatically use
    GPU acceleration if available.

    Attributes:
        d (int): Vector dimension
        qtype (int): Quantizer type (see faiss.ScalarQuantizer constants)
        metric_type (str): Distance metric type ('L2' or 'IP')
        is_trained (bool): Whether the index has been trained
        ntotal (int): Total number of vectors in the index
        name (str): Unique identifier for the index
        index_id (str): Server-side index identifier (when in remote mode)
    """

    def __init__(self, d: int, qtype: Optional[Any] = None, metric_type: Optional[Any] = None):
        """
        Initialize the scalar quantizer index with specified parameters.

        Args:
            d (int): Vector dimension
            qtype: Scalar quantizer type (if None, uses default QT_8bit)
            metric_type: Distance metric, either faiss.METRIC_L2 or
                        faiss.METRIC_INNER_PRODUCT
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
        self.d: int = d
        self.qtype: Any = qtype
        # Convert metric type to string representation for remote mode
        self.metric_type: str = "IP" if metric_type == METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained: bool = True  # Scalar quantizer doesn't need explicit training
        self.ntotal: int = 0

        # Initialize GPU-related attributes
        self._use_gpu: bool = False
        self._gpu_resources: Optional[Any] = None
        self._local_index: Optional[Any] = None

        # Generate unique name for the index
        self.name: str = f"index-sq-{uuid.uuid4().hex[:8]}"
        self.index_id: str = self.name

        # Initialize vector mapping for remote mode
        self._vector_mapping: Dict[int, int] = {}  # Maps server-side indices to local indices
        self._next_idx: int = 0  # Counter for local indices

        # Check if client exists and its mode
        client = get_client()

        # Explicit check for remote mode instead of just checking if client exists
        if client is not None and client.mode == "remote":
            # Remote mode
            logger.info(f"Creating remote IndexScalarQuantizer on server {client.server}")
            self._create_remote_index(client, d, qtype)
        else:
            # Local mode
            logger.info(f"Creating local IndexScalarQuantizer index {self.name}")
            self._create_local_index(d, qtype, metric_type)

    def _get_index_type_string(self, qtype: Optional[Any] = None) -> str:
        """
        Get standardized string representation of index type.

        Args:
            qtype: Quantizer type to convert to string

        Returns:
            String representation of index type
        """
        # Use internal qtype if none provided
        if qtype is None:
            qtype = self.qtype

        # Handle different qtype formats
        if isinstance(qtype, str):
            # If qtype is already a string (e.g., "SQ8"), use it directly
            qtype_str = qtype
        elif qtype is None:
            # Default to SQ8 if none specified
            qtype_str = "SQ8"
        else:
            # Convert integer constant to string representation
            # Map common quantizer types to string representations
            qtype_map = {
                1: "SQ8",  # QT_8bit
                2: "SQ4",  # QT_4bit
                5: "SQ16"  # QT_fp16
            }
            qtype_str = qtype_map.get(qtype, f"SQ{qtype}")

        # Ensure the qtype_str has the "SQ" prefix if it doesn't already
        if not qtype_str.startswith("SQ"):
            qtype_str = f"SQ{qtype_str}"

        # Add metric type suffix if needed
        if self.metric_type == "IP":
            qtype_str = f"{qtype_str}_IP"

        return qtype_str

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

    def _create_local_index(self, d: int, qtype: Any, metric_type: Any) -> None:
        """
        Create a local FAISS scalar quantizer index.

        Args:
            d (int): Vector dimension
            qtype: Quantizer type
            metric_type: Distance metric type

        Raises:
            RuntimeError: If index creation fails
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

            # Set default qtype if not specified
            if qtype is None:
                qtype = faiss.ScalarQuantizer.QT_8bit

            if gpu_available:
                # GPU is available, create resources and GPU index
                self._use_gpu = True
                self._gpu_resources = faiss.StandardGpuResources()

                # Create CPU index first
                cpu_index = faiss.IndexScalarQuantizer(d, qtype, metric_type)

                # Convert to GPU index
                try:
                    self._local_index = faiss.index_cpu_to_gpu(
                        self._gpu_resources, 0, cpu_index
                    )
                    logger.info(f"Using GPU-accelerated SQ index for {self.name}")
                except Exception as e:
                    # If GPU conversion fails, fall back to CPU
                    self._local_index = cpu_index
                    self._use_gpu = False
                    logger.warning(
                        f"Failed to create GPU SQ index: {e}, using CPU instead"
                    )
            else:
                # No GPUs available, use CPU version
                self._local_index = faiss.IndexScalarQuantizer(d, qtype, metric_type)

            self.index_id = self.name  # Use name as ID for consistency
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")

    def _create_remote_index(self, client: Any, d: int, qtype: Any) -> None:
        """
        Create a remote scalar quantizer index on the server.

        Args:
            client: FAISSx client instance
            d (int): Vector dimension
            qtype: Quantizer type

        Raises:
            RuntimeError: If remote index creation fails
        """
        try:
            # Get index type string
            index_type = self._get_index_type_string(qtype)

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
                f"Failed to create remote scalar quantizer index: {e}. "
                f"Server may not support SQ indices with type {index_type}."
            )

    def _map_server_to_local_indices(self, server_indices: List[int]) -> np.ndarray:
        """
        Map server indices to local indices using the mapping dictionary.

        Args:
            server_indices: List of server-side indices

        Returns:
            Array of local indices
        """
        local_indices = []
        for server_idx in server_indices:
            local_idx = self._vector_mapping.get(server_idx, -1)
            local_indices.append(local_idx)
        return np.array(local_indices, dtype=np.int64)

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

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        client = get_client()

        # Explicit check for remote mode instead of just checking if client exists
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

        # Make sure the index is trained before adding vectors
        if not self._local_index.is_trained:
            # For scalar quantizer, some require training
            self._local_index.train(vectors)

        # Use local FAISS implementation directly
        self._local_index.add(vectors)
        self.ntotal = self._local_index.ntotal

    def _add_remote(self, client: Any, vectors: np.ndarray) -> None:
        """
        Add vectors to remote index.

        Args:
            client: FAISSx client
            vectors: Vectors to add
        """
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
        """
        Add a batch of vectors to the remote index.

        Args:
            client: FAISSx client
            vectors: Batch of vectors to add
        """
        # Add vectors to remote index
        result = client.add_vectors(self.index_id, vectors)

        # Log response
        logger.debug(f"Server response: {result}")

        # Update local tracking if addition was successful
        if isinstance(result, dict) and result.get("success", False):
            added_count = result.get("count", 0)
            # Create mapping for each added vector
            for i in range(added_count):
                server_idx = self.ntotal + i
                local_idx = self._next_idx
                # Store mapping from server index to local index
                self._vector_mapping[server_idx] = local_idx
                self._next_idx += 1

            self.ntotal += added_count
        elif not isinstance(result, dict):
            # Handle non-dict responses (e.g., string)
            logger.warning(f"Unexpected response format from server: {result}")
            # Assume we added all vectors as a fallback
            for i in range(len(vectors)):
                server_idx = self.ntotal + i
                local_idx = self._next_idx
                self._vector_mapping[server_idx] = local_idx
                self._next_idx += 1

            self.ntotal += len(vectors)

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

        # Get k_factor parameter for oversampling, if set
        k_factor = self.get_parameter('k_factor')
        if k_factor <= 1.0:
            k_factor = 1.0

        # Calculate internal_k with k_factor and clamp to ntotal
        internal_k = min(int(k * k_factor), max(1, self.ntotal))
        need_reranking = (k_factor > 1.0 and internal_k > k)

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        client = get_client()

        # Explicit check for remote mode instead of just checking if client exists
        if client is not None and client.mode == "remote":
            return self._search_remote(client, query_vectors, k, internal_k, need_reranking)
        else:
            return self._search_local(query_vectors, k, internal_k, need_reranking)

    def _search_local(
        self,
        query_vectors: np.ndarray,
        k: int,
        internal_k: int,
        need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in local index.

        Args:
            query_vectors: Vectors to query
            k: Number of results to return
            internal_k: Number of results to retrieve internally (may be larger than k)
            need_reranking: Whether to rerank results

        Returns:
            Tuple of (distances, indices)
        """
        logger.debug(f"Searching {len(query_vectors)} vectors in local index {self.name}")

        # Use local FAISS implementation directly
        distances, indices = self._local_index.search(query_vectors, internal_k)

        # If k_factor was applied, rerank and trim results
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
        need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in remote index.

        Args:
            client: FAISSx client
            query_vectors: Vectors to query
            k: Number of results to return
            internal_k: Number of results to retrieve internally
            need_reranking: Whether to rerank results

        Returns:
            Tuple of (distances, indices)
        """
        logger.debug(f"Searching {len(query_vectors)} vectors in remote index {self.index_id}")

        # Get batch size parameter
        batch_size = self.get_parameter('batch_size')
        if batch_size <= 0:
            batch_size = 100  # Default if not set or invalid

        # If queries fit in a single batch, search directly
        if len(query_vectors) <= batch_size:
            return self._search_remote_batch(
                client, query_vectors, k, internal_k, need_reranking
            )

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
        self,
        client: Any,
        query_vectors: np.ndarray,
        k: int,
        internal_k: int,
        need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search a batch of queries in the remote index.

        Args:
            client: FAISSx client
            query_vectors: Vectors to query
            k: Number of results to return
            internal_k: Number of results to retrieve internally
            need_reranking: Whether to rerank results

        Returns:
            Tuple of (distances, indices)
        """
        # Perform search on remote index
        result = client.search(self.index_id, query_vectors=query_vectors, k=internal_k)

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
                continue

            result_distances = result_data.get("distances", [])
            result_indices = result_data.get("indices", [])

            # Limit to k results
            max_j = min(k, len(result_distances))

            # Fill distances directly
            distances[i, :max_j] = result_distances[:max_j]

            # Map server indices to local indices
            for j in range(max_j):
                server_idx = result_indices[j]
                indices[i, j] = self._vector_mapping.get(server_idx, -1)

        return distances, indices

    def range_search(
        self, x: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for vectors within a specified radius of query vectors.

        Args:
            x (np.ndarray): Query vectors, shape (n, d)
            radius (float): Maximum distance threshold for results

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - lims: Boundaries of results for each query, shape (n+1)
                - distances: Distances for each result, shape (sum_of_results)
                - indices: Indices for each result, shape (sum_of_results)

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

        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        client = get_client()

        # Explicit check for remote mode instead of just checking if client exists
        if client is not None and client.mode == "remote":
            return self._range_search_remote(client, query_vectors, radius)
        else:
            return self._range_search_local(query_vectors, radius)

    def _range_search_local(
        self,
        query_vectors: np.ndarray,
        radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute range search on the local index.

        Args:
            query_vectors: Query vectors
            radius: Maximum distance threshold

        Returns:
            Tuple of lims, distances, and indices arrays

        Raises:
            RuntimeError: If range search fails
        """
        logger.debug(f"Range searching {len(query_vectors)} vectors in local index {self.name}")

        if self._local_index is None:
            raise RuntimeError("Local index not initialized")

        try:
            # Try to use the range_search method if it exists
            if hasattr(self._local_index, 'range_search'):
                lims, distances, indices = self._local_index.range_search(query_vectors, radius)
                return lims, distances, indices
            else:
                # Fallback implementation for indices that don't support range_search directly
                raise NotImplementedError(
                    "Range search not supported by this index type in local mode"
                )
        except Exception as e:
            logger.error(f"Error in local range search: {e}")
            raise RuntimeError(f"Range search failed: {e}")

    def _range_search_remote(
        self,
        client: Any,
        query_vectors: np.ndarray,
        radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute range search on the remote index.

        Args:
            client: The FAISSx client
            query_vectors: Query vectors
            radius: Maximum distance threshold

        Returns:
            Tuple of lims, distances, and indices arrays
        """
        logger.debug(
            f"Range searching {len(query_vectors)} vectors in remote index {self.index_id}"
        )

        # Get batch size parameter
        batch_size = self.get_parameter('batch_size')
        if batch_size <= 0:
            batch_size = 100  # Default if not set

        # If queries fit in a single batch, search directly
        if len(query_vectors) <= batch_size:
            return self._range_search_remote_batch(client, query_vectors, radius)

        # Process in batches
        all_lims = [0]  # Start with 0 for the first offset
        all_distances = []
        all_indices = []

        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:min(i + batch_size, len(query_vectors))]
            lims, distances, indices = self._range_search_remote_batch(client, batch, radius)

            # Adjust lims to account for previous results
            offset = all_lims[-1]
            adjusted_lims = lims[1:] + offset  # Skip the first lim (0) and add offset
            all_lims.extend(adjusted_lims)

            # Add distances and indices
            all_distances.extend(distances)
            all_indices.extend(indices)

        return (
            np.array(all_lims, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            np.array(all_indices, dtype=np.int64)
        )

    def _range_search_remote_batch(
        self,
        client: Any,
        query_vectors: np.ndarray,
        radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute range search for a batch of queries on the remote index.

        Args:
            client: The FAISSx client
            query_vectors: Batch of query vectors
            radius: Maximum distance threshold

        Returns:
            Tuple of lims, distances, and indices arrays
        """
        try:
            # Send request to server
            response = client.range_search(self.index_id, query_vectors, radius)

            # Log response
            logger.debug(f"Range search server response: {response}")

            # Initialize default return values
            n = query_vectors.shape[0]
            lims = np.zeros(n + 1, dtype=np.int64)
            distances = np.array([], dtype=np.float32)
            indices = np.array([], dtype=np.int64)

            # Process response
            if not isinstance(response, dict) or "results" not in response:
                logger.warning(f"Unexpected range search response format: {response}")
                return lims, distances, indices

            # Extract results list
            search_results = response.get("results", [])
            if not isinstance(search_results, list):
                logger.warning(f"Invalid results format, expected list: {search_results}")
                return lims, distances, indices

            # Process results
            if isinstance(search_results, list) and len(search_results) > 0:
                # Check if results contain lims, distances, and indices
                if ("lims" in search_results[0] and
                        "distances" in search_results[0] and
                        "indices" in search_results[0]):
                    # Extract and combine results
                    lims = np.array(search_results[0]["lims"], dtype=np.int64)
                    distances = np.array(search_results[0]["distances"], dtype=np.float32)
                    indices = np.array(search_results[0]["indices"], dtype=np.int64)
                else:
                    # Alternative format: process individual per-query results
                    total_results = 0
                    for i, result in enumerate(search_results):
                        if (isinstance(result, dict) and
                                "distances" in result and "indices" in result):
                            result_distances = result.get("distances", [])
                            result_indices = result.get("indices", [])

                            lims[i + 1] = lims[i] + len(result_distances)
                            distances = np.concatenate(
                                (distances, np.array(result_distances, dtype=np.float32))
                            )
                            indices = np.concatenate(
                                (indices, np.array(result_indices, dtype=np.int64))
                            )

                            total_results += len(result_distances)

                    # Complete lims array
                    lims[n] = total_results

            return lims, distances, indices

        except Exception as e:
            logger.error(f"Error in remote range search: {e}")
            raise RuntimeError(f"Range search failed on remote index: {e}")

    def reset(self) -> None:
        """
        Reset the index to its initial state, removing all vectors.
        """
        # Register access for memory management
        self.register_access()

        client = get_client()

        # Explicit check for remote mode instead of just checking if client exists
        if client is not None and client.mode == "remote":
            self._reset_remote(client)
        else:
            self._reset_local()

    def _reset_local(self) -> None:
        """
        Reset local index.

        This removes all vectors but keeps the index configuration.
        """
        logger.debug(f"Resetting local index {self.name}")
        # Reset local FAISS index
        if self._local_index is not None:
            self._local_index.reset()
        self.ntotal = 0

    def _reset_remote(self, client: Any) -> None:
        """
        Reset remote index by creating a new one.

        Args:
            client: FAISSx client
        """
        logger.debug(f"Resetting remote index {self.index_id}")

        try:
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"
            logger.debug(f"Creating new index {new_name} to replace {self.name}")

            # Get index type string
            index_type = self._get_index_type_string()

            response = client.create_index(
                name=new_name, dimension=self.d, index_type=index_type
            )

            # Log the raw response for debugging
            logger.debug(f"Server response: {response}")

            # Parse response
            if isinstance(response, dict):
                self.index_id = response.get("index_id", new_name)
                self.name = new_name
            else:
                # If response is not a dict, use the new name as ID
                logger.warning(f"Unexpected server response format: {response}")
                self.index_id = new_name
                self.name = new_name

            logger.info(f"Reset index with new name: {self.name}")
        except Exception as e:
            # Recreate with same name if error occurs
            logger.warning(f"Failed to create new index during reset: {e}")
            logger.debug(f"Trying to recreate index with same name: {self.name}")

            # Get index type string
            index_type = self._get_index_type_string()

            response = client.create_index(
                name=self.name, dimension=self.d, index_type=index_type
            )

            # Log the raw response for debugging
            logger.debug(f"Server response: {response}")

            # Parse response
            if isinstance(response, dict):
                self.index_id = response.get("index_id", self.name)
            else:
                # If response is not a dict, use the name as ID
                logger.warning(f"Unexpected server response format: {response}")
                self.index_id = self.name

        # Reset all local state
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def __enter__(self) -> 'IndexScalarQuantizer':
        """
        Context manager entry.

        Returns:
            Self for use in context manager
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> None:
        """
        Context manager exit.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self.close()

    def close(self) -> None:
        """
        Clean up resources.

        This is particularly important for freeing GPU resources.
        """
        if self._use_gpu and self._gpu_resources is not None:
            self._gpu_resources = None
        self._local_index = None

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        self.close()

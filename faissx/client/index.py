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
import faiss

# Define constants for when faiss isn't fully available
if not hasattr(faiss, 'METRIC_L2'):
    class FaissConstants:
        METRIC_L2 = 0
        METRIC_INNER_PRODUCT = 1
    faiss = FaissConstants()

from .client import get_client


class IndexFlatL2:
    """
    Proxy implementation of FAISS IndexFlatL2

    This class mimics the behavior of FAISS IndexFlatL2. It uses the local FAISS
    implementation by default, but can use the remote FAISSx service when explicitly
    configured via configure(). It maintains a mapping between local and server-side
    indices to ensure consistent indexing across operations when using remote mode.

    When running in local mode with CUDA-capable GPUs, it will automatically use
    GPU acceleration if available.

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
        _gpu_resources: GPU resources if using GPU (local mode only)
        _use_gpu (bool): Whether we're using GPU acceleration (local mode only)
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

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None

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

            # Check if GPU is available and can be used
            try:
                # Import GPU-specific module
                import faiss.contrib.gpu
                ngpus = faiss.get_num_gpus()

                if ngpus > 0:
                    # GPU is available, create resources and GPU index
                    self._use_gpu = True
                    self._gpu_resources = faiss.StandardGpuResources()

                    # Create CPU index first
                    cpu_index = faiss.IndexFlatL2(d)

                    # Convert to GPU index
                    self._local_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)

                    import logging
                    logging.info(f"Using GPU-accelerated index for {self.name}")
                else:
                    # No GPUs available
                    self._local_index = faiss.IndexFlatL2(d)
            except (ImportError, AttributeError):
                # GPU support not available in this FAISS build
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

        # Get vector count to be added
        n_vectors = vectors.shape[0]

        # For smaller batches, use regular add_vectors
        if n_vectors <= 1000:
            # Add vectors to remote index (remote mode)
            result = self.client.add_vectors(self.index_id, vectors)
        else:
            # For larger batches, use the batch version with automatic chunking
            result = self.client.batch_add_vectors(self.index_id, vectors)

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

        # Get query count
        n_queries = query_vectors.shape[0]

        # For smaller query batches, use regular search
        if n_queries <= 100:
            # Search via remote index
            result = self.client.search(self.index_id, query_vectors, k)
        else:
            # For larger query batches, use the batch version with automatic chunking
            result = self.client.batch_search(self.index_id, query_vectors, k)

        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Search failed: {error}")

        # Process results
        search_results = result.get("results", [])
        n = x.shape[0]  # Number of query vectors

        # Initialize output arrays with default values
        distances = np.full((n, k), float('inf'), dtype=np.float32)
        indices = np.full((n, k), -1, dtype=np.int64)  # Use -1 as sentinel for not found

        # Fill in results from the search
        for i, res in enumerate(search_results):
            result_distances = res.get("distances", [])
            result_indices = res.get("indices", [])
            num_results = min(len(result_distances), k)

            if num_results > 0:
                # Fill distances directly
                distances[i, :num_results] = result_distances[:num_results]

                # Map server indices back to local indices
                for j, server_idx in enumerate(result_indices[:num_results]):
                    # Look up the server index in our vector mapping
                    found = False
                    for local_idx, info in self._vector_mapping.items():
                        if info["server_idx"] == server_idx:
                            indices[i, j] = local_idx
                            found = True
                            break

                    if not found:
                        indices[i, j] = -1  # Not found in local mapping

        return distances, indices

    def range_search(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            if hasattr(self._local_index, 'range_search'):
                return self._local_index.range_search(query_vectors, radius)
            else:
                raise RuntimeError("Local FAISS index does not support range_search")

        # Get query count
        n_queries = query_vectors.shape[0]

        # For smaller query batches, use regular range search
        if n_queries <= 100:
            # Search via remote index
            result = self.client.range_search(self.index_id, query_vectors, radius)
        else:
            # For larger query batches, use the batch version with automatic chunking
            result = self.client.batch_range_search(self.index_id, query_vectors, radius)

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
            result_indices = res.get("indices", [])
            count = len(result_distances)

            # Copy data to output arrays
            if count > 0:
                distances[offset:offset+count] = np.array(result_distances, dtype=np.float32)

                # Map server indices back to local indices
                mapped_indices = np.zeros(count, dtype=np.int64)
                for j, server_idx in enumerate(result_indices):
                    found = False
                    for local_idx, info in self._vector_mapping.items():
                        if info["server_idx"] == server_idx:
                            mapped_indices[j] = local_idx
                            found = True
                            break

                    if not found:
                        mapped_indices[j] = -1

                indices[offset:offset+count] = mapped_indices
                offset += count

        # Set final boundary
        lims[n_queries] = offset

        return lims, distances, indices

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
        # Clean up GPU resources if used
        if self._use_gpu and self._gpu_resources is not None:
            # Nothing explicit needed as StandardGpuResources has its own cleanup in __del__
            self._gpu_resources = None

        # Local index will be cleaned up by garbage collector
        pass


class IndexIVFFlat:
    """
    Proxy implementation of FAISS IndexIVFFlat

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
        _using_remote (bool): Whether we're using remote or local implementation
        _gpu_resources: GPU resources if using GPU (local mode only)
        _use_gpu (bool): Whether we're using GPU acceleration (local mode only)
    """

    def __init__(self, quantizer, d: int, nlist: int, metric_type=faiss.METRIC_L2):
        """
        Initialize the inverted file index with specified parameters.

        Args:
            quantizer: Quantizer object that defines the centroids (usually IndexFlatL2)
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions
            metric_type: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        # Store core parameters
        self.d = d
        self.nlist = nlist
        # Convert metric type to string representation
        self.metric_type = "IP" if metric_type == faiss.METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained = False
        self.ntotal = 0

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None

        # Generate unique name for the index
        self.name = f"index-ivf-flat-{uuid.uuid4().hex[:8]}"

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
                self._local_index = None

                # Determine index type identifier
                index_type = f"IVF{nlist}"
                if self.metric_type == "IP":
                    index_type = f"{index_type}_IP"

                # Create index on server
                response = self.client.create_index(
                    name=self.name,
                    dimension=self.d,
                    index_type=index_type
                )

                self.index_id = response.get("index_id", self.name)
                self.is_trained = response.get("is_trained", False)

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

            # Check if GPU is available and can be used
            try:
                # Import GPU-specific module
                import faiss.contrib.gpu
                ngpus = faiss.get_num_gpus()

                if ngpus > 0:
                    # GPU is available, create resources
                    self._use_gpu = True
                    self._gpu_resources = faiss.StandardGpuResources()

                    # Create CPU index first
                    if isinstance(quantizer, IndexFlatL2) and quantizer._use_gpu:
                        # If the quantizer is already on GPU, get the CPU version
                        cpu_quantizer = faiss.index_gpu_to_cpu(quantizer._local_index)
                    else:
                        # Otherwise, use the provided quantizer directly
                        cpu_quantizer = quantizer._local_index if hasattr(quantizer, '_local_index') else quantizer

                    cpu_index = faiss.IndexIVFFlat(cpu_quantizer, d, nlist, metric_type)

                    # Convert to GPU index
                    self._local_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)

                    import logging
                    logging.info(f"Using GPU-accelerated IVF index for {self.name}")
                else:
                    # No GPUs available, use CPU version
                    self._local_index = faiss.IndexIVFFlat(quantizer._local_index, d, nlist, metric_type)
            except (ImportError, AttributeError):
                # GPU support not available in this FAISS build
                self._local_index = faiss.IndexIVFFlat(quantizer._local_index, d, nlist, metric_type)

            self.index_id = self.name  # Use name as ID for consistency
        except ImportError as e:
            raise ImportError(f"Failed to import FAISS for local mode: {e}")

    def train(self, x: np.ndarray) -> None:
        """
        Train the index with the provided vectors.

        Args:
            x (np.ndarray): Training vectors, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or already trained
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            self._local_index.train(vectors)
            self.is_trained = self._local_index.is_trained
            return

        # Train the remote index
        result = self.client.train_index(self.index_id, vectors)

        # Update local state based on training result
        if result.get("success", False):
            self.is_trained = result.get("is_trained", True)

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or index not trained
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

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

        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")

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

    def range_search(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            if hasattr(self._local_index, 'range_search'):
                return self._local_index.range_search(query_vectors, radius)
            else:
                raise RuntimeError("Local FAISS index does not support range_search")

        # Search via remote index
        result = self.client.range_search(self.index_id, query_vectors, radius)

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
            result_indices = res.get("indices", [])
            count = len(result_distances)

            # Copy data to output arrays
            if count > 0:
                distances[offset:offset+count] = np.array(result_distances, dtype=np.float32)

                # Map server indices back to local indices
                mapped_indices = np.zeros(count, dtype=np.int64)
                for j, server_idx in enumerate(result_indices):
                    found = False
                    for local_idx, info in self._vector_mapping.items():
                        if info["server_idx"] == server_idx:
                            mapped_indices[j] = local_idx
                            found = True
                            break

                    if not found:
                        mapped_indices[j] = -1

                indices[offset:offset+count] = mapped_indices
                offset += count

        # Set final boundary
        lims[n_queries] = offset

        return lims, distances, indices

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        if not self._using_remote:
            # Reset local FAISS index
            self._local_index.reset()
            self.ntotal = 0
            self.is_trained = False
            return

        # Remote mode reset
        try:
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            index_type = f"IVF{self.nlist}"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=new_name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", new_name)
            self.name = new_name
            self.is_trained = False
        except Exception:
            # Recreate with same name if error occurs
            index_type = f"IVF{self.nlist}"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=self.name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", self.name)
            self.is_trained = False

        # Reset all local state
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        # Clean up GPU resources if used
        if self._use_gpu and self._gpu_resources is not None:
            # Nothing explicit needed as StandardGpuResources has its own cleanup in __del__
            self._gpu_resources = None

        # Local index will be cleaned up by garbage collector
        pass


class IndexHNSWFlat:
    """
    Proxy implementation of FAISS IndexHNSWFlat

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
    """

    def __init__(self, d: int, M: int = 32, metric=faiss.METRIC_L2):
        """
        Initialize the HNSW index with specified parameters.

        Args:
            d (int): Vector dimension
            M (int): Number of connections per node (higher = better accuracy, more memory)
            metric: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        # Store core parameters
        self.d = d
        self.M = M
        # Convert metric type to string representation
        self.metric_type = "IP" if metric == faiss.METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained = True  # HNSW doesn't need training
        self.ntotal = 0

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None

        # Generate unique name for the index
        self.name = f"index-hnsw-flat-{uuid.uuid4().hex[:8]}"

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
                self._local_index = None

                # Determine index type identifier
                index_type = f"HNSW{M}"
                if self.metric_type == "IP":
                    index_type = f"{index_type}_IP"

                # Create index on server
                response = self.client.create_index(
                    name=self.name,
                    dimension=self.d,
                    index_type=index_type
                )

                self.index_id = response.get("index_id", self.name)

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

            # Check if GPU is available and can be used
            try:
                # Import GPU-specific module
                import faiss.contrib.gpu
                ngpus = faiss.get_num_gpus()

                if ngpus > 0:
                    # GPU is available, create resources
                    self._use_gpu = True
                    self._gpu_resources = faiss.StandardGpuResources()

                    # Note: HNSW is only partially supported on GPU
                    # Create CPU index first (all HNSW operations except search will run on CPU)
                    cpu_index = faiss.IndexHNSWFlat(d, M, metric)

                    # For HNSW, typically only search can be GPU-accelerated
                    # Convert to GPU index, but many operations will fall back to CPU
                    try:
                        self._local_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
                        import logging
                        logging.info(f"Using GPU-accelerated HNSW index for {self.name} (search only)")
                    except Exception as e:
                        # If GPU conversion fails, fall back to CPU
                        self._local_index = cpu_index
                        self._use_gpu = False
                        import logging
                        logging.warning(f"Failed to create GPU HNSW index: {e}, using CPU instead")
                else:
                    # No GPUs available, use CPU version
                    self._local_index = faiss.IndexHNSWFlat(d, M, metric)
            except (ImportError, AttributeError):
                # GPU support not available in this FAISS build
                self._local_index = faiss.IndexHNSWFlat(d, M, metric)

            self.index_id = self.name  # Use name as ID for consistency
        except ImportError as e:
            raise ImportError(f"Failed to import FAISS for local mode: {e}")

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

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

    def range_search(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            if hasattr(self._local_index, 'range_search'):
                return self._local_index.range_search(query_vectors, radius)
            else:
                raise RuntimeError("Local FAISS index does not support range_search")

        # Search via remote index
        result = self.client.range_search(self.index_id, query_vectors, radius)

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
            result_indices = res.get("indices", [])
            count = len(result_distances)

            # Copy data to output arrays
            if count > 0:
                distances[offset:offset+count] = np.array(result_distances, dtype=np.float32)

                # Map server indices back to local indices
                mapped_indices = np.zeros(count, dtype=np.int64)
                for j, server_idx in enumerate(result_indices):
                    found = False
                    for local_idx, info in self._vector_mapping.items():
                        if info["server_idx"] == server_idx:
                            mapped_indices[j] = local_idx
                            found = True
                            break

                    if not found:
                        mapped_indices[j] = -1

                indices[offset:offset+count] = mapped_indices
                offset += count

        # Set final boundary
        lims[n_queries] = offset

        return lims, distances, indices

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
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            index_type = f"HNSW{self.M}"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=new_name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", new_name)
            self.name = new_name
        except Exception:
            # Recreate with same name if error occurs
            index_type = f"HNSW{self.M}"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=self.name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", self.name)

        # Reset all local state
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        # Clean up GPU resources if used
        if self._use_gpu and self._gpu_resources is not None:
            # Nothing explicit needed as StandardGpuResources has its own cleanup in __del__
            self._gpu_resources = None

        # Local index will be cleaned up by garbage collector
        pass


class IndexPQ:
    """
    Proxy implementation of FAISS IndexPQ

    This class mimics the behavior of FAISS IndexPQ, which uses Product Quantization
    for efficient vector compression and similarity search. PQ significantly reduces
    the memory footprint of vectors while maintaining reasonable search accuracy.

    When running in local mode with CUDA-capable GPUs, it will automatically use
    GPU acceleration if available.

    Attributes:
        d (int): Vector dimension
        M (int): Number of subquantizers
        nbits (int): Number of bits per subquantizer (default 8)
        metric_type (str): Distance metric type ('L2' or 'IP')
        is_trained (bool): Whether the index has been trained
        ntotal (int): Total number of vectors in the index
        name (str): Unique identifier for the index
        index_id (str): Server-side index identifier (when in remote mode)
        _vector_mapping (dict): Maps local indices to server indices (remote mode only)
        _next_idx (int): Next available local index (remote mode only)
        _local_index: Local FAISS index (local mode only)
        _using_remote (bool): Whether we're using remote or local implementation
        _gpu_resources: GPU resources if using GPU (local mode only)
        _use_gpu (bool): Whether we're using GPU acceleration (local mode only)
    """

    def __init__(self, d: int, M: int = 8, nbits: int = 8, metric=faiss.METRIC_L2):
        """
        Initialize the PQ index with specified parameters.

        Args:
            d (int): Vector dimension (must be a multiple of M)
            M (int): Number of subquantizers
            nbits (int): Number of bits per subquantizer (default 8)
            metric: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        # Validate that dimension is a multiple of M
        if d % M != 0:
            raise ValueError(f"PQ requires dimension ({d}) to be a multiple of M ({M})")

        # Store core parameters
        self.d = d
        self.M = M
        self.nbits = nbits
        # Convert metric type to string representation
        self.metric_type = "IP" if metric == faiss.METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained = False
        self.ntotal = 0

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None

        # Generate unique name for the index
        self.name = f"index-pq-{uuid.uuid4().hex[:8]}"

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
                self._local_index = None

                # Determine index type identifier
                index_type = f"PQ{M}x{nbits}"
                if self.metric_type == "IP":
                    index_type = f"{index_type}_IP"

                # Create index on server
                response = self.client.create_index(
                    name=self.name,
                    dimension=self.d,
                    index_type=index_type
                )

                self.index_id = response.get("index_id", self.name)
                self.is_trained = response.get("is_trained", False)

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

            # Check if GPU is available and can be used
            try:
                # Import GPU-specific module
                import faiss.contrib.gpu
                ngpus = faiss.get_num_gpus()

                if ngpus > 0:
                    # GPU is available, create resources
                    self._use_gpu = True
                    self._gpu_resources = faiss.StandardGpuResources()

                    # Create CPU index first
                    cpu_index = faiss.IndexPQ(d, M, nbits, metric)

                    # Convert to GPU index
                    try:
                        self._local_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
                        import logging
                        logging.info(f"Using GPU-accelerated PQ index for {self.name}")
                    except Exception as e:
                        # If GPU conversion fails, fall back to CPU
                        self._local_index = cpu_index
                        self._use_gpu = False
                        import logging
                        logging.warning(f"Failed to create GPU PQ index: {e}, using CPU instead")
                else:
                    # No GPUs available, use CPU version
                    self._local_index = faiss.IndexPQ(d, M, nbits, metric)
            except (ImportError, AttributeError):
                # GPU support not available in this FAISS build
                self._local_index = faiss.IndexPQ(d, M, nbits, metric)

            self.index_id = self.name  # Use name as ID for consistency
        except ImportError as e:
            raise ImportError(f"Failed to import FAISS for local mode: {e}")

    def train(self, x: np.ndarray) -> None:
        """
        Train the index with the provided vectors.

        Args:
            x (np.ndarray): Training vectors, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or already trained
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            self._local_index.train(vectors)
            self.is_trained = self._local_index.is_trained
            return

        # Train the remote index
        result = self.client.train_index(self.index_id, vectors)

        # Update local state based on training result
        if result.get("success", False):
            self.is_trained = result.get("is_trained", True)

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or index not trained
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

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

        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")

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

    def range_search(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            if hasattr(self._local_index, 'range_search'):
                return self._local_index.range_search(query_vectors, radius)
            else:
                raise RuntimeError("Local FAISS index does not support range_search")

        # Search via remote index
        result = self.client.range_search(self.index_id, query_vectors, radius)

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
            result_indices = res.get("indices", [])
            count = len(result_distances)

            # Copy data to output arrays
            if count > 0:
                distances[offset:offset+count] = np.array(result_distances, dtype=np.float32)

                # Map server indices back to local indices
                mapped_indices = np.zeros(count, dtype=np.int64)
                for j, server_idx in enumerate(result_indices):
                    found = False
                    for local_idx, info in self._vector_mapping.items():
                        if info["server_idx"] == server_idx:
                            mapped_indices[j] = local_idx
                            found = True
                            break

                    if not found:
                        mapped_indices[j] = -1

                indices[offset:offset+count] = mapped_indices
                offset += count

        # Set final boundary
        lims[n_queries] = offset

        return lims, distances, indices

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        if not self._using_remote:
            # Reset local FAISS index
            self._local_index.reset()
            self.ntotal = 0
            self.is_trained = False
            return

        # Remote mode reset
        try:
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            index_type = f"PQ{self.M}x{self.nbits}"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=new_name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", new_name)
            self.name = new_name
            self.is_trained = False
        except Exception:
            # Recreate with same name if error occurs
            index_type = f"PQ{self.M}x{self.nbits}"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=self.name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", self.name)
            self.is_trained = False

        # Reset all local state
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        # Clean up GPU resources if used
        if self._use_gpu and self._gpu_resources is not None:
            # Nothing explicit needed as StandardGpuResources has its own cleanup in __del__
            self._gpu_resources = None

        # Local index will be cleaned up by garbage collector
        pass


class IndexIVFPQ:
    """
    Proxy implementation of FAISS IndexIVFPQ

    This class mimics the behavior of FAISS IndexIVFPQ, which combines inverted file
    indexing with product quantization for efficient similarity search. It offers
    both excellent search performance and memory efficiency, particularly for large
    high-dimensional datasets.

    When running in local mode with CUDA-capable GPUs, it will automatically use
    GPU acceleration if available.

    Attributes:
        d (int): Vector dimension
        nlist (int): Number of clusters/partitions for IVF
        M (int): Number of subquantizers for PQ
        nbits (int): Number of bits per subquantizer
        metric_type (str): Distance metric type ('L2' or 'IP')
        is_trained (bool): Whether the index has been trained
        ntotal (int): Total number of vectors in the index
        name (str): Unique identifier for the index
        index_id (str): Server-side index identifier (when in remote mode)
        _vector_mapping (dict): Maps local indices to server indices (remote mode only)
        _next_idx (int): Next available local index (remote mode only)
        _local_index: Local FAISS index (local mode only)
        _using_remote (bool): Whether we're using remote or local implementation
        _gpu_resources: GPU resources if using GPU (local mode only)
        _use_gpu (bool): Whether we're using GPU acceleration (local mode only)
    """

    def __init__(self, quantizer, d: int, M: int, nbits: int, nlist: int, metric_type=faiss.METRIC_L2):
        """
        Initialize the IVF-PQ index with specified parameters.

        Args:
            quantizer: Quantizer object that defines the centroids (usually IndexFlatL2)
            d (int): Vector dimension
            M (int): Number of subquantizers for PQ (must be a divisor of d)
            nbits (int): Number of bits per subquantizer (typically 8)
            nlist (int): Number of clusters/partitions for IVF
            metric_type: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        # Store core parameters
        self.d = d
        self.M = M
        self.nbits = nbits
        self.nlist = nlist

        # Validate parameters
        if d % M != 0:
            raise ValueError(f"Dimension {d} must be a multiple of M={M}")

        # Convert metric type to string representation
        self.metric_type = "IP" if metric_type == faiss.METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained = False
        self.ntotal = 0

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None

        # Generate unique name for the index
        self.name = f"index-ivfpq-{uuid.uuid4().hex[:8]}"

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
                self._local_index = None

                # Determine index type identifier
                index_type = f"IVFPQ{self.nlist}_{self.M}x{self.nbits}"
                if self.metric_type == "IP":
                    index_type = f"{index_type}_IP"

                # Create index on server
                response = self.client.create_index(
                    name=self.name,
                    dimension=self.d,
                    index_type=index_type
                )

                self.index_id = response.get("index_id", self.name)
                self.is_trained = response.get("is_trained", False)

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

            # Check if GPU is available and can be used
            try:
                # Import GPU-specific module
                import faiss.contrib.gpu
                ngpus = faiss.get_num_gpus()

                if ngpus > 0:
                    # GPU is available, create resources
                    self._use_gpu = True
                    self._gpu_resources = faiss.StandardGpuResources()

                    # Create CPU index first
                    if isinstance(quantizer, IndexFlatL2) and quantizer._use_gpu:
                        # If the quantizer is already on GPU, get the CPU version
                        cpu_quantizer = faiss.index_gpu_to_cpu(quantizer._local_index)
                    else:
                        # Otherwise, use the provided quantizer directly
                        cpu_quantizer = quantizer._local_index if hasattr(quantizer, '_local_index') else quantizer

                    cpu_index = faiss.IndexIVFPQ(cpu_quantizer, d, nlist, M, nbits, metric_type)

                    # Convert to GPU index
                    self._local_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)

                    import logging
                    logging.info(f"Using GPU-accelerated IVFPQ index for {self.name}")
                else:
                    # No GPUs available, use CPU version
                    self._local_index = faiss.IndexIVFPQ(quantizer._local_index, d, nlist, M, nbits, metric_type)
            except (ImportError, AttributeError):
                # GPU support not available in this FAISS build
                self._local_index = faiss.IndexIVFPQ(quantizer._local_index, d, nlist, M, nbits, metric_type)

            self.index_id = self.name  # Use name as ID for consistency
        except ImportError as e:
            raise ImportError(f"Failed to import FAISS for local mode: {e}")

    def train(self, x: np.ndarray) -> None:
        """
        Train the index with the provided vectors.

        Args:
            x (np.ndarray): Training vectors, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or already trained
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            self._local_index.train(vectors)
            self.is_trained = self._local_index.is_trained
            return

        # Train the remote index
        result = self.client.train_index(self.index_id, vectors)

        # Update local state based on training result
        if result.get("success", False):
            self.is_trained = result.get("is_trained", True)

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or index not trained
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

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

        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")

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

    def range_search(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            if hasattr(self._local_index, 'range_search'):
                return self._local_index.range_search(query_vectors, radius)
            else:
                raise RuntimeError("Local FAISS index does not support range_search")

        # Search via remote index
        result = self.client.range_search(self.index_id, query_vectors, radius)

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
            result_indices = res.get("indices", [])
            count = len(result_distances)

            # Copy data to output arrays
            if count > 0:
                distances[offset:offset+count] = np.array(result_distances, dtype=np.float32)

                # Map server indices back to local indices
                mapped_indices = np.zeros(count, dtype=np.int64)
                for j, server_idx in enumerate(result_indices):
                    found = False
                    for local_idx, info in self._vector_mapping.items():
                        if info["server_idx"] == server_idx:
                            mapped_indices[j] = local_idx
                            found = True
                            break

                    if not found:
                        mapped_indices[j] = -1

                indices[offset:offset+count] = mapped_indices
                offset += count

        # Set final boundary
        lims[n_queries] = offset

        return lims, distances, indices

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        if not self._using_remote:
            # Reset local FAISS index
            self._local_index.reset()
            self.ntotal = 0
            self.is_trained = False
            return

        # Remote mode reset
        try:
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            index_type = f"IVFPQ{self.nlist}_{self.M}x{self.nbits}"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=new_name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", new_name)
            self.name = new_name
            self.is_trained = False
        except Exception:
            # Recreate with same name if error occurs
            index_type = f"IVFPQ{self.nlist}_{self.M}x{self.nbits}"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=self.name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", self.name)
            self.is_trained = False

        # Reset all local state
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        # Clean up GPU resources if used
        if self._use_gpu and self._gpu_resources is not None:
            # Nothing explicit needed as StandardGpuResources has its own cleanup in __del__
            self._gpu_resources = None

        # Local index will be cleaned up by garbage collector
        pass


class IndexScalarQuantizer:
    """
    Proxy implementation of FAISS IndexScalarQuantizer

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
        _vector_mapping (dict): Maps local indices to server indices (remote mode only)
        _next_idx (int): Next available local index (remote mode only)
        _local_index: Local FAISS index (local mode only)
        _using_remote (bool): Whether we're using remote or local implementation
        _gpu_resources: GPU resources if using GPU (local mode only)
        _use_gpu (bool): Whether we're using GPU acceleration (local mode only)
    """

    def __init__(self, d: int, qtype=None, metric_type=faiss.METRIC_L2):
        """
        Initialize the scalar quantizer index with specified parameters.

        Args:
            d (int): Vector dimension
            qtype: Scalar quantizer type (if None, uses default QT_8bit)
            metric_type: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        # Store core parameters
        self.d = d

        # Use default quantizer type if not specified
        if qtype is None:
            # Use QT_8bit as default if available, or 0 as fallback
            self.qtype = getattr(faiss, "ScalarQuantizer.QT_8bit", 0)
        else:
            self.qtype = qtype

        # Convert metric type to string representation
        self.metric_type = "IP" if metric_type == faiss.METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained = True  # Scalar quantizer doesn't need training
        self.ntotal = 0

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None

        # Generate unique name for the index
        self.name = f"index-scalar-quantizer-{uuid.uuid4().hex[:8]}"

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
                self._local_index = None

                # Determine index type identifier
                index_type = "ScalarQ"
                if self.metric_type == "IP":
                    index_type = f"{index_type}_IP"

                # Create index on server
                response = self.client.create_index(
                    name=self.name,
                    dimension=self.d,
                    index_type=index_type
                )

                self.index_id = response.get("index_id", self.name)
                self.is_trained = True  # ScalarQuantizer is always trained

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

            # Check if GPU is available and can be used
            try:
                # Import GPU-specific module
                import faiss.contrib.gpu
                ngpus = faiss.get_num_gpus()

                if ngpus > 0:
                    # GPU is available, create resources
                    self._use_gpu = True
                    self._gpu_resources = faiss.StandardGpuResources()

                    # Create CPU index first
                    cpu_index = faiss.IndexScalarQuantizer(d, self.qtype, metric_type)

                    # Convert to GPU index
                    self._local_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)

                    import logging
                    logging.info(f"Using GPU-accelerated ScalarQuantizer index for {self.name}")
                else:
                    # No GPUs available, use CPU version
                    self._local_index = faiss.IndexScalarQuantizer(d, self.qtype, metric_type)
            except (ImportError, AttributeError):
                # GPU support not available in this FAISS build
                self._local_index = faiss.IndexScalarQuantizer(d, self.qtype, metric_type)

            self.index_id = self.name  # Use name as ID for consistency
        except ImportError as e:
            raise ImportError(f"Failed to import FAISS for local mode: {e}")

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

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

    def range_search(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            if hasattr(self._local_index, 'range_search'):
                return self._local_index.range_search(query_vectors, radius)
            else:
                raise RuntimeError("Local FAISS index does not support range_search")

        # Search via remote index
        result = self.client.range_search(self.index_id, query_vectors, radius)

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
            result_indices = res.get("indices", [])
            count = len(result_distances)

            # Copy data to output arrays
            if count > 0:
                distances[offset:offset+count] = np.array(result_distances, dtype=np.float32)

                # Map server indices back to local indices
                mapped_indices = np.zeros(count, dtype=np.int64)
                for j, server_idx in enumerate(result_indices):
                    found = False
                    for local_idx, info in self._vector_mapping.items():
                        if info["server_idx"] == server_idx:
                            mapped_indices[j] = local_idx
                            found = True
                            break

                    if not found:
                        mapped_indices[j] = -1

                indices[offset:offset+count] = mapped_indices
                offset += count

        # Set final boundary
        lims[n_queries] = offset

        return lims, distances, indices

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
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            index_type = "ScalarQ"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=new_name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", new_name)
            self.name = new_name
        except Exception:
            # Recreate with same name if error occurs
            index_type = "ScalarQ"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=self.name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", self.name)

        # Reset all local state
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        # Clean up GPU resources if used
        if self._use_gpu and self._gpu_resources is not None:
            # Nothing explicit needed as StandardGpuResources has its own cleanup in __del__
            self._gpu_resources = None

        # Local index will be cleaned up by garbage collector
        pass

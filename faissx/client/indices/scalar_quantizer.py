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
"""

import uuid
import numpy as np
from typing import Tuple
import logging

# Avoid module-level dependency on faiss
try:
    import faiss
    METRIC_L2 = faiss.METRIC_L2
    METRIC_INNER_PRODUCT = faiss.METRIC_INNER_PRODUCT
except ImportError:
    # Define constants for when faiss isn't available
    METRIC_L2 = 0
    METRIC_INNER_PRODUCT = 1

from ..client import get_client


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

    def __init__(self, d: int, qtype=None, metric_type=METRIC_L2):
        """
        Initialize the scalar quantizer index with specified parameters.

        Args:
            d (int): Vector dimension
            qtype: Scalar quantizer type (if None, uses default QT_8bit)
            metric_type: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        # Store core parameters
        self.d = d
        # Convert metric type to string representation
        self.metric_type = "IP" if metric_type == METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained = True  # Scalar quantizer doesn't need explicit training
        self.ntotal = 0

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None

        # Generate unique name for the index
        self.name = f"index-sq-{uuid.uuid4().hex[:8]}"

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

                # Determine index type identifier for scalar quantizer
                qtype_str = "SQ8" if qtype is None else f"SQ{qtype}"
                index_type = qtype_str
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
            logging.warning(f"Error initializing remote mode: {e}, falling back to local mode")

        # Use local FAISS implementation by default
        self._using_remote = False
        self._local_index = None

        # Import faiss again to ensure it's in the local scope
        try:
            import faiss

            # Try to use GPU if available
            gpu_available = False
            try:
                import faiss.contrib.gpu  # type: ignore
                ngpus = faiss.get_num_gpus()
                gpu_available = ngpus > 0
            except (ImportError, AttributeError) as e:
                logging.warning(f"GPU support not available: {e}")
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
                    logging.info(f"Using GPU-accelerated SQ index for {self.name}")
                except Exception as e:
                    # If GPU conversion fails, fall back to CPU
                    self._local_index = cpu_index
                    self._use_gpu = False
                    logging.warning(f"Failed to create GPU SQ index: {e}, using CPU instead")
            else:
                # No GPUs available, use CPU version
                self._local_index = faiss.IndexScalarQuantizer(d, qtype, metric_type)

            self.index_id = self.name  # Use name as ID for consistency
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")

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
        # Implementation for range_search
        raise NotImplementedError("Range search is not yet implemented for IndexScalarQuantizer")

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        if not self._using_remote:
            # Reset local FAISS index
            self._local_index.reset()
            self.ntotal = 0
            return

        # Remote mode reset - create a new index
        try:
            # Determine index type for recreation
            qtype_str = "SQ8"  # Default
            index_type = qtype_str
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"
            response = self.client.create_index(
                name=new_name,
                dimension=self.d,
                index_type=index_type
            )

            self.index_id = response.get("index_id", new_name)
            self.name = new_name
        except Exception:
            # Recreate with same name if error occurs
            qtype_str = "SQ8"  # Default
            index_type = qtype_str
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

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

from ..client import get_client
from .base import logger


class IndexScalarQuantizer:
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
        _vector_mapping (dict): Maps local indices to server indices (remote mode only)
        _next_idx (int): Next available local index (remote mode only)
        _local_index: Local FAISS index (local mode only)
        _gpu_resources: GPU resources if using GPU (local mode only)
        _use_gpu (bool): Whether we're using GPU acceleration (local mode only)
    """

    def __init__(self, d: int, qtype=None, metric_type=None):
        """
        Initialize the scalar quantizer index with specified parameters.

        Args:
            d (int): Vector dimension
            qtype: Scalar quantizer type (if None, uses default QT_8bit)
            metric_type: Distance metric, either faiss.METRIC_L2 or
                        faiss.METRIC_INNER_PRODUCT
        """
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
        # Convert metric type to string representation for remote mode
        self.metric_type = "IP" if metric_type == METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained = True  # Scalar quantizer doesn't need explicit training
        self.ntotal = 0

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None

        # Generate unique name for the index
        self.name = f"index-sq-{uuid.uuid4().hex[:8]}"

        # Initialize vector mapping for remote mode
        self._vector_mapping = {}  # Maps local indices to server-side information
        self._next_idx = 0  # Counter for local indices

        # Check if client exists and its mode
        client = get_client()

        if client is None or client.mode == "local":
            # Local mode
            logger.info(f"Creating local IndexScalarQuantizer index {self.name}")
            self._create_local_index(d, qtype, metric_type)
        else:
            # Remote mode
            logger.info(f"Creating remote IndexScalarQuantizer on server {client.server}")
            self._create_remote_index(client, d, qtype)

    def _create_local_index(self, d, qtype, metric_type):
        """Create a local FAISS scalar quantizer index."""
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

    def _create_remote_index(self, client, d, qtype):
        """Create a remote scalar quantizer index on the server."""
        try:
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

            # Determine the final index type
            index_type = qtype_str
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            # Create index on server
            logger.debug(f"Creating remote index {self.name} with type {index_type}")
            response = client.create_index(
                name=self.name, dimension=d, index_type=index_type
            )

            # Log the raw response for debugging
            logger.debug(f"Server response: {response}")

            # Handle different response formats
            if isinstance(response, dict):
                self.index_id = response.get("index_id", self.name)
            else:
                # If response is not a dict (maybe a string or other type), use the name as ID
                logger.warning(f"Unexpected server response format: {response}")
                self.index_id = self.name

            logger.info(f"Created remote index with ID: {self.index_id}")
        except Exception as e:
            # Raise a clear error instead of falling back to local mode
            raise RuntimeError(
                f"Failed to create remote scalar quantizer index: {e}. "
                f"Server may not support SQ indices with type {index_type}."
            )

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
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        client = get_client()

        if client is None or client.mode == "local":
            # Local mode
            logger.debug(f"Adding {len(vectors)} vectors to local index {self.name}")
            # Make sure the index is trained before adding vectors
            if not self._local_index.is_trained:
                # For scalar quantizer, some require training
                self._local_index.train(vectors)

            # Use local FAISS implementation directly
            self._local_index.add(vectors)
            self.ntotal = self._local_index.ntotal
        else:
            # Remote mode
            logger.debug(f"Adding {len(vectors)} vectors to remote index {self.index_id}")
            # Add vectors to remote index
            result = client.add_vectors(self.index_id, vectors)

            # Log response
            logger.debug(f"Server response: {result}")

            # Update local tracking if addition was successful
            if isinstance(result, dict) and result.get("success", False):
                added_count = result.get("count", 0)
                # Create mapping for each added vector
                for i in range(added_count):
                    self._vector_mapping[self._next_idx] = {
                        "local_idx": self._next_idx,
                        "server_idx": self.ntotal + i,
                    }
                    self._next_idx += 1

                self.ntotal += added_count
            elif not isinstance(result, dict):
                # Handle non-dict responses (e.g., string)
                logger.warning(f"Unexpected response format from server: {result}")
                # Assume we added all vectors as a fallback
                self.ntotal += len(vectors)
                for i in range(len(vectors)):
                    self._vector_mapping[self._next_idx] = {
                        "local_idx": self._next_idx,
                        "server_idx": self.ntotal - len(vectors) + i,
                    }
                    self._next_idx += 1

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
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        client = get_client()

        if client is None or client.mode == "local":
            # Local mode
            logger.debug(f"Searching {len(query_vectors)} vectors in local index {self.name}")
            return self._local_index.search(query_vectors, k)
        else:
            # Remote mode
            logger.debug(f"Searching {len(query_vectors)} vectors in remote index {self.index_id}")
            # Perform search on remote index
            result = client.search(self.index_id, query_vectors=query_vectors, k=k)

            # Log response
            logger.debug(f"Server response: {result}")

            n = x.shape[0]  # Number of query vectors

            # Initialize output arrays with default values
            distances = np.full((n, k), float("inf"), dtype=np.float32)
            idx = np.full((n, k), -1, dtype=np.int64)

            # Process results based on response format
            if (isinstance(result, dict) and
                "results" in result and
                isinstance(result["results"], list)):
                search_results = result["results"]

                # Process results for each query vector
                for i in range(min(n, len(search_results))):
                    result_data = search_results[i]
                    if isinstance(result_data, dict):
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
            else:
                # Alternative response format handling
                logger.warning(f"Unexpected search response format: {result}")

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
        raise NotImplementedError(
            "Range search is not yet implemented for IndexScalarQuantizer"
        )

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        client = get_client()

        if client is None or client.mode == "local":
            # Local mode
            logger.debug(f"Resetting local index {self.name}")
            # Reset local FAISS index
            self._local_index.reset()
            self.ntotal = 0
        else:
            # Remote mode
            logger.debug(f"Resetting remote index {self.index_id}")
            try:
                # Create new index with modified name
                new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"
                logger.debug(f"Creating new index {new_name} to replace {self.name}")

                # Determine index type using the same logic as in _create_remote_index
                qtype_str = "SQ8"  # Default
                if self.metric_type == "IP":
                    index_type = f"{qtype_str}_IP"
                else:
                    index_type = qtype_str

                response = client.create_index(
                    name=new_name, dimension=self.d, index_type=index_type
                )

                # Log the raw response for debugging
                logger.debug(f"Server response: {response}")

                # Handle different response formats
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

                # Determine index type using the same logic as in _create_remote_index
                qtype_str = "SQ8"  # Default
                if self.metric_type == "IP":
                    index_type = f"{qtype_str}_IP"
                else:
                    index_type = qtype_str

                response = client.create_index(
                    name=self.name, dimension=self.d, index_type=index_type
                )

                # Log the raw response for debugging
                logger.debug(f"Server response: {response}")

                # Handle different response formats
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

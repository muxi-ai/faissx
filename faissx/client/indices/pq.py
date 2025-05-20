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
FAISSx IndexPQ implementation.

This module provides a client-side implementation of the FAISS IndexPQ class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).
"""

from .base import uuid, np, Tuple, faiss, logging, get_client


class IndexPQ:
    """
    Proxy implementation of FAISS IndexPQ.

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
        # Convert metric type to string representation for remote mode
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

                # Determine index type identifier for server
                index_type = f"PQ{M}x{nbits}"
                if self.metric_type == "IP":
                    index_type = f"{index_type}_IP"

                # Create index on server
                response = self.client.create_index(
                    name=self.name, dimension=self.d, index_type=index_type
                )

                self.index_id = response.get("index_id", self.name)
                self.is_trained = response.get("is_trained", False)

                # Initialize local tracking of vectors for remote mode
                self._vector_mapping = (
                    {}
                )  # Maps local indices to server-side information
                self._next_idx = 0  # Counter for local indices
                return

        except Exception as e:
            logging.warning(
                f"Error initializing remote mode: {e}, falling back to local mode"
            )

        # Use local FAISS implementation by default
        self._using_remote = False
        self._local_index = None

        # Import local FAISS here to avoid module-level dependency
        try:
            # Check if GPU is available and can be used
            try:
                # Import GPU-specific module
                import faiss.contrib.gpu  # type: ignore

                ngpus = faiss.get_num_gpus()

                if ngpus > 0:
                    # GPU is available, create resources
                    self._use_gpu = True
                    self._gpu_resources = faiss.StandardGpuResources()

                    # Create CPU index first
                    cpu_index = faiss.IndexPQ(d, M, nbits, metric)

                    # Convert to GPU index
                    try:
                        self._local_index = faiss.index_cpu_to_gpu(
                            self._gpu_resources, 0, cpu_index
                        )
                        logging.info(f"Using GPU-accelerated PQ index for {self.name}")
                    except Exception as e:
                        # If GPU conversion fails, fall back to CPU
                        self._local_index = cpu_index
                        self._use_gpu = False
                        logging.warning(
                            f"Failed to create GPU PQ index: {e}, using CPU instead"
                        )
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
        # Implementation omitted for brevity
        pass

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or index not trained
        """
        # Implementation omitted for brevity
        pass

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
        # Implementation omitted for brevity
        pass

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
        # Implementation omitted for brevity
        pass

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        # Implementation omitted for brevity
        pass

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

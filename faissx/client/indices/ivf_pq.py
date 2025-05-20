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
FAISSx IndexIVFPQ implementation.

This module provides a client-side implementation of the FAISS IndexIVFPQ class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).
"""

from .base import np, Tuple, faiss


class IndexIVFPQ:
    """
    Proxy implementation of FAISS IndexIVFPQ.

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

    def __init__(
        self,
        quantizer,
        d: int,
        M: int,
        nbits: int,
        nlist: int,
        metric_type=faiss.METRIC_L2,
    ):
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
        # Implementation initialization (constructor)
        # Omitted for brevity
        pass

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

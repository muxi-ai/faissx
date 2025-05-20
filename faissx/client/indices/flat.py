"""
FAISSx IndexFlatL2 implementation.

This module provides a client-side implementation of the FAISS IndexFlatL2 class.
"""

import uuid
import numpy as np
from typing import Tuple, Optional

import faiss
import logging

from ..client import get_client
from .base import FAISSxBaseIndex

logger = logging.getLogger(__name__)


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
        super().__init__()  # Initialize base class

        self.d = d
        self.ntotal = 0
        self.is_trained = True  # Flat indices don't need training
        self._local_index = None
        self._use_gpu = use_gpu
        self._gpu_resources = None
        self.name = f"index-flat-l2-{uuid.uuid4().hex[:8]}"

        client = get_client()
        if client is None:
            # Local mode - create a FAISS index directly
            self._local_index = faiss.IndexFlatL2(d)

            # Move to GPU if requested
            if use_gpu:
                try:
                    import faiss.contrib.gpu
                    if faiss.get_num_gpus() > 0:
                        self._gpu_resources = faiss.StandardGpuResources()
                        self._local_index = faiss.index_cpu_to_gpu(
                            self._gpu_resources, 0, self._local_index
                        )
                except (ImportError, AttributeError):
                    logger.warning("GPU requested but FAISS GPU support not available")
        else:
            # Remote mode - create a FAISS index on the server
            client.create_index(self.name, d, "L2")

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
        if client is None:
            # Local mode
            self._local_index.add(x)
            self.ntotal = self._local_index.ntotal
        else:
            # Remote mode
            # Get the batch size for adding vectors
            batch_size = self.get_parameter('batch_size')

            # Process in batches to avoid overwhelming the server
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                client.add_vectors(self.name, batch)

            # Update the total count
            self.ntotal += len(x)

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

        # Apply k_factor parameter if set
        k_factor = self.get_parameter('k_factor')
        internal_k = min(int(k * k_factor), self.ntotal)
        need_reranking = (k_factor > 1.0 and internal_k > k)

        client = get_client()
        if client is None:
            # Local mode
            distances, indices = self._local_index.search(x, internal_k)

            # If k_factor was applied, rerank and trim results
            if need_reranking:
                distances = distances[:, :k]
                indices = indices[:, :k]

            return distances, indices
        else:
            # Remote mode
            # Get the batch size for search operations
            batch_size = self.get_parameter('batch_size')

            # Process in batches to avoid overwhelming the server
            all_distances = []
            all_indices = []

            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                distances, indices = client.search(self.name, batch, internal_k)

                # If k_factor was applied, rerank and trim results
                if need_reranking:
                    distances = distances[:, :k]
                    indices = indices[:, :k]

                all_distances.append(distances)
                all_indices.append(indices)

            # Combine results
            if len(all_distances) == 1:
                return all_distances[0], all_indices[0]
            else:
                return np.vstack(all_distances), np.vstack(all_indices)

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        client = get_client()
        if client is None:
            # Local mode
            if self._local_index is not None:
                self._local_index.reset()
                self.ntotal = 0
        else:
            # Remote mode
            # Delete and recreate the index on the server
            client.delete_index(self.name)
            client.create_index(self.name, self.d, "L2")
            self.ntotal = 0

        # Reset parameters
        self.reset_parameters()

    def range_search(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if client is None:
            # Local mode
            return self._local_index.range_search(x, radius)
        else:
            # Remote mode
            # Get the batch size for search operations
            batch_size = self.get_parameter('batch_size')

            # Process in batches
            if len(x) <= batch_size:
                return client.range_search(self.name, x, radius)
            else:
                # Handle large batches by breaking them up and combining results
                all_lims = [0]
                all_distances = []
                all_indices = []

                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    lims, distances, indices = client.range_search(self.name, batch, radius)

                    # Adjust lims for concatenation (except first entry which is always 0)
                    if all_distances:
                        lims = lims[1:] + all_lims[-1]

                    all_lims.extend(lims[1:])
                    all_distances.extend(distances)
                    all_indices.extend(indices)

                return np.array(all_lims), np.array(all_distances), np.array(all_indices)

    def reconstruct(self, i: int) -> np.ndarray:
        """
        Reconstruct a vector from the index.

        Args:
            i: Vector index

        Returns:
            Reconstructed vector

        Raises:
            RuntimeError: If index is empty
            IndexError: If index is out of bounds
        """
        # Register access for memory management
        self.register_access()

        if self.ntotal == 0:
            raise RuntimeError("Cannot reconstruct from an empty index")

        if i < 0 or i >= self.ntotal:
            raise IndexError(f"Index out of bounds: {i}, index has {self.ntotal} vectors")

        client = get_client()
        if client is None:
            # Local mode
            return self._local_index.reconstruct(i)
        else:
            # Remote mode
            return client.reconstruct(self.name, i)

    def reconstruct_n(self, i0: int, ni: int) -> np.ndarray:
        """
        Reconstruct a range of vectors from the index.

        Args:
            i0: First index to reconstruct
            ni: Number of vectors to reconstruct

        Returns:
            Reconstructed vectors, shape (ni, d)

        Raises:
            RuntimeError: If index is empty
            IndexError: If any index is out of bounds
        """
        # Register access for memory management
        self.register_access()

        if self.ntotal == 0:
            raise RuntimeError("Cannot reconstruct from an empty index")

        if i0 < 0 or i0 + ni > self.ntotal:
            raise IndexError(f"Index range out of bounds: {i0}:{i0+ni}, index has {self.ntotal} vectors")

        client = get_client()
        if client is None:
            # Local mode
            return self._local_index.reconstruct_n(i0, ni)
        else:
            # Remote mode
            # For small batches, use a direct call
            if ni <= 100:
                return client.reconstruct_n(self.name, i0, ni)
            else:
                # For larger batches, break up into smaller pieces
                batch_size = 100
                result = []

                for i in range(i0, i0 + ni, batch_size):
                    current_ni = min(batch_size, i0 + ni - i)
                    result.append(client.reconstruct_n(self.name, i, current_ni))

                return np.vstack(result)

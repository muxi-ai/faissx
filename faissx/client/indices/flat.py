"""
FAISSx IndexFlatL2 implementation.

This module provides a client-side implementation of the FAISS IndexFlatL2 class.
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
                import faiss.contrib.gpu
                ngpus = faiss.get_num_gpus()
                gpu_available = ngpus > 0
            except (ImportError, AttributeError) as e:
                logging.warning(f"GPU support not available: {e}")
                gpu_available = False

            if gpu_available:
                # GPU is available, create resources and GPU index
                self._use_gpu = True
                self._gpu_resources = faiss.StandardGpuResources()

                # Create CPU index first
                cpu_index = faiss.IndexFlatL2(d)

                # Convert to GPU index
                self._local_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
                logging.info(f"Using GPU-accelerated index for {self.name}")
            else:
                # No GPUs available
                self._local_index = faiss.IndexFlatL2(d)

            self.index_id = self.name  # Use name as ID for consistency
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")

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

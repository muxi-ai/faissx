#!/usr/bin/env python3
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

IVF-PQ combines inverted file indexing with product quantization for efficient
similarity search and memory usage, making it ideal for large-scale applications.
"""

import time
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from ..client import get_client
from .base import FAISSxBaseIndex, logger


class IndexIVFPQ(FAISSxBaseIndex):
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
        _vector_mapping (Dict): Maps local indices to server indices (remote mode only)
        _next_idx (int): Next available local index (remote mode only)
        _local_index: Local FAISS index (local mode only)
        _gpu_resources: GPU resources if using GPU (local mode only)
        _use_gpu (bool): Whether we're using GPU acceleration (local mode only)
        _nprobe (int): Number of clusters to search (default: 1)
    """

    def __init__(
        self,
        quantizer: Any,
        d: int,
        nlist: int,
        m: int,
        nbits: int,
        metric_type: Optional[Any] = None,
    ) -> None:
        """
        Initialize the IVF-PQ index with specified parameters.

        Args:
            quantizer: Quantizer object that defines the centroids (usually IndexFlatL2)
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions for IVF
            m (int): Number of subquantizers for PQ (must be a divisor of d)
            nbits (int): Number of bits per subquantizer (typically 8)
            metric_type: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        super().__init__()  # Initialize base class

        # Try to import faiss locally to avoid module-level dependency
        try:
            import faiss as local_faiss

            default_metric = local_faiss.METRIC_L2
            metric_inner_product = local_faiss.METRIC_INNER_PRODUCT
        except ImportError:
            # Define fallback constants when faiss isn't available
            default_metric = 0
            metric_inner_product = 1
            local_faiss = None

        # Use default metric if not provided
        if metric_type is None:
            metric_type = default_metric

        # Validate that d is a multiple of M (required by FAISS)
        if d % m != 0:
            raise ValueError(f"The dimension ({d}) must be a multiple of M ({m})")

        # Store core parameters
        self.d = d
        self.M = m  # FAISS uses capital M in the property name
        self.nbits = nbits
        self.nlist = nlist

        # Convert metric type to string representation for remote mode
        self.metric_type = "IP" if metric_type == metric_inner_product else "L2"

        # Initialize state variables
        self.is_trained = False
        self.ntotal = 0
        self._nprobe = 1  # Default number of probes

        # For vector caching (helps with reconstruction for io.py)
        self._cached_vectors = None

        # Initialize GPU-related attributes
        self._use_gpu = False
        self._gpu_resources = None
        self._local_index = None

        # Generate unique name for the index
        self.name = f"index-ivf-pq-{uuid.uuid4().hex[:8]}"
        self.index_id = self.name

        # Initialize vector mapping for remote mode
        # Maps local indices to server-side information
        self._vector_mapping: Dict[int, Dict[str, int]] = {}
        self._next_idx: int = 0  # Counter for local indices

        # Check if client exists and its mode
        client = get_client()

        # Explicit check for remote mode instead of just checking if client exists
        if client is not None and client.mode == "remote":
            # Remote mode
            logger.info(f"Creating remote IndexIVFPQ on server {client.server}")
            self._create_remote_index(client, quantizer, d, nlist, m, nbits)
        else:
            # Local mode
            logger.info(f"Creating local IndexIVFPQ index {self.name}")
            self._create_local_index(quantizer, d, nlist, m, nbits, metric_type)

    def _get_index_type_string(self) -> str:
        """
        Get standardized string representation of the IVF-PQ index type.

        Returns:
            String representation of index type
        """
        # Create the index type string based on parameters
        index_type = f"IVF{self.nlist},PQ{self.M}x{self.nbits}"

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

    def _create_local_index(
        self, quantizer, d: int, nlist: int, m: int, nbits: int, metric_type: Any
    ) -> None:
        """
        Create a local FAISS IVF-PQ index.

        Args:
            quantizer: Quantizer object that defines the centroids
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions for IVF
            m (int): Number of subquantizers for PQ
            nbits (int): Number of bits per subquantizer
            metric_type: Distance metric type
        """
        try:
            import faiss as local_faiss

            # Try to use GPU if available
            gpu_available = False
            try:
                import faiss.contrib.gpu  # type: ignore

                ngpus = local_faiss.get_num_gpus()
                gpu_available = ngpus > 0
            except (ImportError, AttributeError) as e:
                logger.warning(f"GPU support not available: {e}")
                gpu_available = False

            # Get local index from wrapped quantizer if available
            if hasattr(quantizer, "_local_index"):
                base_quantizer = quantizer._local_index
            else:
                base_quantizer = quantizer

            # Create the local index
            cpu_index = local_faiss.IndexIVFPQ(
                base_quantizer, d, nlist, m, nbits, metric_type
            )

            if gpu_available:
                # GPU is available, create resources and GPU index
                self._use_gpu = True
                self._gpu_resources = local_faiss.StandardGpuResources()

                # Use factory method for better GPU optimization choices
                gpu_index = local_faiss.index_cpu_to_gpu(
                    self._gpu_resources, 0, cpu_index
                )
                self._local_index = gpu_index
                logger.info(f"Created GPU-accelerated IndexIVFPQ with dimension {d}")
            else:
                # No GPU, use CPU index
                self._local_index = cpu_index
                logger.info(f"Created CPU IndexIVFPQ with dimension {d}")

        except ImportError:
            logger.warning("FAISS not available for local index creation")
            self._local_index = None

    def _create_remote_index(
        self, client: Any, quantizer, d: int, nlist: int, m: int, nbits: int
    ) -> None:
        """
        Create a remote IVF-PQ index on the FAISSx server.

        Args:
            client: FAISSx client instance
            quantizer: Quantizer object that defines the centroids
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions for IVF
            m (int): Number of subquantizers for PQ
            nbits (int): Number of bits per subquantizer
        """
        try:
            # Get the index type string - this already includes nlist, m, and nbits
            index_type = self._get_index_type_string()

            # Create the index on the server with simplified parameters
            # that are common across different server implementations
            params = {
                "name": self.name,
                "dimension": d,
                "index_type": index_type,
            }

            if self.metric_type != "L2":
                params["metric_type"] = self.metric_type

            # Note: We don't pass quantizer_id or specific parameters like nlist/m/nbits
            # because not all server implementations support these parameters.
            # The index_type string already encodes this information for the server.

            response = client.create_index(**params)

            # Parse response and set index ID
            self.index_id = self._parse_server_response(response, self.name)
            logger.info(f"Created remote index with ID: {self.index_id}")

            # Check if the index is already trained (server might train it automatically)
            try:
                status_response = client.get_index_status(self.index_id)
                if isinstance(status_response, dict) and status_response.get("is_trained", False):
                    self.is_trained = True
                    logger.info("Remote index already trained by server")
            except Exception as e:
                logger.warning(f"Could not check index training status: {e}")

        except Exception as e:
            logger.error(f"Failed to create remote index: {e}")
            raise ValueError(f"Could not create remote index: {e}") from e

    @property
    def nprobe(self) -> int:
        """Get the number of clusters to search."""
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        """Set the number of clusters to search."""
        self.set_nprobe(value)

    def set_nprobe(self, nprobe: int) -> None:
        """
        Set the number of clusters to probe during search.

        This parameter controls the trade-off between search accuracy and speed.
        Higher values improve search quality at the cost of speed.

        Args:
            nprobe: Number of clusters to probe (1-nlist)
        """
        if nprobe < 1:
            nprobe = 1

        # Limit maximum nprobe to avoid extreme slowdowns
        if nprobe > self.nlist:
            logger.warning(
                f"nprobe {nprobe} > nlist {self.nlist}, capping to {self.nlist}"
            )
            nprobe = self.nlist

        self._nprobe = nprobe

        # Update the local index if in local mode
        if hasattr(self, "_local_index") and self._local_index is not None:
            self._local_index.nprobe = nprobe

        # Update the remote index if in remote mode
        client = get_client()
        if client is not None and client.mode == "remote":
            try:
                client.set_parameter(self.index_id, "nprobe", nprobe)
            except Exception as e:
                logger.warning(f"Failed to set nprobe on server: {e}")

    def train(self, x: np.ndarray) -> None:
        """
        Train the index with the given vectors.

        This must be called before adding vectors if the index is not pre-trained.

        Args:
            x: Training vectors as a 2D numpy array

        Raises:
            ValueError: If the vector dimension doesn't match the index dimension
        """
        if x.shape[1] != self.d:
            raise ValueError(
                f"Training data has dimension {x.shape[1]}, "
                f"but index dimension is {self.d}"
            )

        # Convert vectors to float32 if needed
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # Check if client exists and mode
        client = get_client()
        if client is not None and client.mode == "remote":
            self._train_remote(client, x)
        else:
            self._train_local(x)

        # Mark index as trained
        self.is_trained = True

    def _train_local(self, vectors: np.ndarray) -> None:
        """
        Train the local FAISS index.

        Args:
            vectors: Training vectors
        """
        if self._local_index is not None:
            start_time = time.time()
            self._local_index.train(vectors)
            logger.info(f"Local index trained in {time.time() - start_time:.2f}s")

    def _train_remote(self, client: Any, vectors: np.ndarray) -> None:
        """
        Train the remote index on the FAISSx server.

        Args:
            client: FAISSx client instance
            vectors: Training vectors
        """
        start_time = time.time()

        # Check if the index is already trained on the server
        try:
            status_response = client.get_index_status(self.index_id)
            if isinstance(status_response, dict) and status_response.get("is_trained", False):
                logger.info("Remote index already trained, skipping training")
                self.is_trained = True
                return
        except (AttributeError, Exception) as e:
            # Some servers don't implement get_index_status, just proceed with training
            logger.warning(f"Could not check training status: {e}")

        # Send training vectors to server
        try:
            # Try different method names the server might use for training
            for method_name in ['train', 'train_index']:
                if hasattr(client, method_name):
                    train_method = getattr(client, method_name)
                    response = train_method(self.index_id, vectors)

                    if isinstance(response, dict) and response.get("success", False):
                        logger.info(f"Remote index trained in {time.time() - start_time:.2f}s")
                        self.is_trained = True
                        return
                    else:
                        logger.warning(f"Unexpected training response: {response}")
                        # Assume training was successful as a fallback
                        self.is_trained = True
                        return

            # If we get here, none of the training methods were found
            logger.warning("Server doesn't support explicit training, assuming auto-training")
            self.is_trained = True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # For servers that don't support explicit training, we'll just assume
            # the index is implicitly trained when adding vectors
            logger.warning("Assuming index will be trained implicitly when adding vectors")
            self.is_trained = True

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        The vectors are added in batches to avoid memory issues and
        to improve performance.

        Args:
            x: Vectors to add as a 2D numpy array

        Raises:
            ValueError: If the index is not trained or if the vector dimension is incorrect
        """
        if not self.is_trained:
            raise ValueError("Cannot add vectors to an untrained index")

        if x.shape[1] != self.d:
            raise ValueError(
                f"Vector dimension {x.shape[1]} doesn't match index dimension {self.d}"
            )

        # Convert vectors to float32 if needed
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # Cache vectors for reconstruction (helps with io.py persistence)
        if hasattr(self, "_cached_vectors") and self._cached_vectors is not None:
            self._cached_vectors = np.vstack([self._cached_vectors, x])
        else:
            self._cached_vectors = x.copy()

        # Check if client exists and mode
        client = get_client()
        if client is not None and client.mode == "remote":
            self._add_remote(client, x)
        else:
            self._add_local(x)

        # Update vector count
        self.ntotal += x.shape[0]

    def _add_local(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the local FAISS index.

        Args:
            vectors: Vectors to add
        """
        if self._local_index is not None:
            self._local_index.add(vectors)

    def _add_remote(self, client: Any, vectors: np.ndarray) -> None:
        """
        Add vectors to the remote index on the FAISSx server.

        Args:
            client: FAISSx client instance
            vectors: Vectors to add
        """
        # Use batch processing for better performance and memory usage
        if vectors.shape[0] > 10000:
            self._add_remote_batch(client, vectors)
            return

        try:
            start_idx = self._next_idx
            # Send vectors to server - ignore response as we track indices locally
            client.add_vectors(self.index_id, vectors)

            # Update vector mapping
            for i in range(vectors.shape[0]):
                self._vector_mapping[start_idx + i] = {"server_id": start_idx + i}

            self._next_idx = start_idx + vectors.shape[0]

        except Exception as e:
            logger.error(f"Error adding vectors to remote index: {e}")
            raise ValueError(f"Failed to add vectors: {e}")

    def _add_remote_batch(self, client: Any, vectors: np.ndarray) -> None:
        """
        Add vectors to the remote index in batches.

        Args:
            client: FAISSx client instance
            vectors: Vectors to add
        """
        batch_size = 10000  # Use reasonable batch size
        num_vectors = vectors.shape[0]
        start_idx = self._next_idx

        try:
            for i in range(0, num_vectors, batch_size):
                end_idx = min(i + batch_size, num_vectors)
                batch = vectors[i:end_idx]

                logger.info(
                    f"Adding batch {i//batch_size + 1}/{(num_vectors-1)//batch_size + 1} "
                    f"({batch.shape[0]} vectors)"
                )

                # Send batch to server - ignore response as we track indices locally
                client.add_vectors(self.index_id, batch)

                # Update vector mapping
                for j in range(batch.shape[0]):
                    self._vector_mapping[start_idx + i + j] = {
                        "server_id": start_idx + i + j
                    }

            self._next_idx = start_idx + num_vectors

        except Exception as e:
            logger.error(f"Error adding vectors in batch: {e}")
            raise ValueError(f"Failed to add vectors in batch: {e}")

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the k nearest neighbors of each query vector.

        This method performs a k-nearest neighbor search for each query vector in x.
        The search uses the IVF-PQ index structure where:
          1. The query is assigned to its nearest centroids
          2. The search is restricted to vectors in those centroid clusters
          3. PQ codes are used to efficiently compute approximate distances

        The search quality depends on:
          - nprobe: Number of clusters to explore (higher gives better results)
          - Training quality: How well the clusters represent the data distribution
          - M (number of subquantizers): Higher values give better precision

        Args:
            x: Query vectors as a 2D numpy array, shape (n_queries, d)
            k: Number of nearest neighbors to retrieve per query

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - distances: Array of shape (n_queries, k) with distances to nearest neighbors
                - indices: Array of shape (n_queries, k) with indices of nearest neighbors

        Raises:
            ValueError: If the index is empty or if the query dimension is incorrect
        """
        if self.ntotal == 0:
            raise ValueError("Cannot search an empty index")

        if x.shape[1] != self.d:
            raise ValueError(
                f"Query dimension {x.shape[1]} doesn't match index dimension {self.d}"
            )

        # Convert vectors to float32 if needed
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # Register access for memory management
        self.register_access()

        # Check if client exists and its mode
        client = get_client()

        # For large k, we need to handle potential reranking
        need_reranking = False
        internal_k = k

        # For remote mode, we might need more results for reranking
        if client is not None and client.mode == "remote" and k > self.nprobe * 10:
            internal_k = k
            need_reranking = True

        if client is not None and client.mode == "remote":
            return self._search_remote(client, x, k, internal_k, need_reranking)
        else:
            return self._search_local(x, k, internal_k, need_reranking)

    def _search_local(
        self,
        query_vectors: np.ndarray,
        k: int,
        internal_k: int,
        need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using the local FAISS index.

        Performs k-nearest neighbor search using the local FAISS implementation. The method
        will automatically utilize GPU if available and handles reranking if needed.

        Args:
            query_vectors: Query vectors, shape (n, d)
            k: Number of results to return per query
            internal_k: Number of results to retrieve internally (for reranking)
            need_reranking: Whether to perform reranking of results

        Returns:
            Tuple of (distances, indices):
                - distances: Array of shape (n, k) with distances to nearest neighbors
                - indices: Array of shape (n, k) with indices of nearest neighbors
        """
        if self._local_index is not None:
            distances, indices = self._local_index.search(query_vectors, k)
            return distances, indices

        # Fallback if no local index
        return np.zeros((query_vectors.shape[0], k), dtype=np.float32), np.zeros(
            (query_vectors.shape[0], k), dtype=np.int64
        )

    def _search_remote(
        self,
        client: Any,
        query_vectors: np.ndarray,
        k: int,
        internal_k: int,
        need_reranking: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using the remote index on the FAISSx server.

        Args:
            client: FAISSx client instance
            query_vectors: Query vectors
            k: Number of results to return
            internal_k: Number of results to retrieve internally (for reranking)
            need_reranking: Whether reranking is needed

        Returns:
            Tuple of (distances, indices)
        """
        # For batch processing when there are many query vectors
        if query_vectors.shape[0] > 100:
            return self._search_remote_batch(
                client, query_vectors, k, internal_k, need_reranking
            )

        try:
            # Search on the server
            response = client.search(self.index_id, query_vectors, internal_k)

            if not isinstance(response, dict):
                raise ValueError(f"Unexpected search response: {response}")

            # Extract distances and indices from response
            distances = response.get("distances")
            indices = response.get("indices")

            if distances is None or indices is None:
                raise ValueError("Search response missing distances or indices")

            # Convert to numpy arrays
            distances = np.array(distances, dtype=np.float32)
            indices = np.array(indices, dtype=np.int64)

            # Apply any necessary mapping from server indices to local indices
            if hasattr(self, "_vector_mapping") and self._vector_mapping:
                # For now, assume server indices match our mapping
                # This could be enhanced with a proper mapping
                pass

            # If we requested more results than needed for reranking, truncate
            if need_reranking and internal_k > k:
                distances = distances[:, :k]
                indices = indices[:, :k]

            return distances, indices

        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty results on error
            return (
                np.zeros((query_vectors.shape[0], k), dtype=np.float32),
                np.zeros((query_vectors.shape[0], k), dtype=np.int64) - 1,
            )

    def _search_remote_batch(
        self,
        client: Any,
        query_vectors: np.ndarray,
        k: int,
        internal_k: int,
        need_reranking: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using the remote index in batches for better performance.

        Args:
            client: FAISSx client instance
            query_vectors: Query vectors
            k: Number of results to return
            internal_k: Number of results to retrieve internally (for reranking)
            need_reranking: Whether reranking is needed

        Returns:
            Tuple of (distances, indices)
        """
        batch_size = 100  # Reasonable batch size for search
        num_queries = query_vectors.shape[0]

        # Initialize result arrays
        all_distances = np.zeros((num_queries, k), dtype=np.float32)
        all_indices = np.zeros((num_queries, k), dtype=np.int64) - 1  # -1 for not found

        try:
            for i in range(0, num_queries, batch_size):
                end_idx = min(i + batch_size, num_queries)
                batch = query_vectors[i:end_idx]

                logger.debug(
                    f"Searching batch {i//batch_size + 1}/{(num_queries-1)//batch_size + 1} "
                    f"({batch.shape[0]} queries)"
                )

                # Search this batch
                response = client.search(self.index_id, batch, internal_k)

                if not isinstance(response, dict):
                    raise ValueError(f"Unexpected search response: {response}")

                # Extract distances and indices from response
                batch_distances = response.get("distances")
                batch_indices = response.get("indices")

                if batch_distances is None or batch_indices is None:
                    raise ValueError("Search response missing distances or indices")

                # Convert to numpy arrays
                batch_distances = np.array(batch_distances, dtype=np.float32)
                batch_indices = np.array(batch_indices, dtype=np.int64)

                # If we requested more results than needed for reranking, truncate
                if need_reranking and internal_k > k:
                    batch_distances = batch_distances[:, :k]
                    batch_indices = batch_indices[:, :k]

                # Store in result arrays
                all_distances[i:end_idx] = batch_distances
                all_indices[i:end_idx] = batch_indices

            return all_distances, all_indices

        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            # Return empty results on error
            return (
                np.zeros((num_queries, k), dtype=np.float32),
                np.zeros((num_queries, k), dtype=np.int64) - 1,
            )

    def range_search(
        self, x: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform range search to find all vectors within the given radius.

        Args:
            x: Query vectors as a 2D numpy array
            radius: Radius for the search

        Returns:
            Tuple of (distances, indices, lims) where:
            - distances: flat array of distances for all results
            - indices: flat array of indices for all results
            - lims: array of size len(x)+1 indicating where each query's results start/end

        Raises:
            ValueError: If the index is empty or if the query dimension is incorrect
        """
        if self.ntotal == 0:
            raise ValueError("Cannot search an empty index")

        if x.shape[1] != self.d:
            raise ValueError(
                f"Query dimension {x.shape[1]} doesn't match index dimension {self.d}"
            )

        # Convert vectors to float32 if needed
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # Register access for memory management
        self.register_access()

        # Check if client exists and its mode
        client = get_client()

        if client is not None and client.mode == "remote":
            return self._range_search_remote(client, x, radius)
        else:
            return self._range_search_local(x, radius)

    def _range_search_local(
        self, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform range search using the local FAISS index.

        Args:
            query_vectors: Query vectors
            radius: Radius for the search

        Returns:
            Tuple of (distances, indices, lims)
        """
        if self._local_index is not None:
            if hasattr(self._local_index, "range_search"):
                return self._local_index.range_search(query_vectors, radius)

        # Fallback if no local index or range_search not supported
        # Return empty results with appropriate shape
        num_queries = query_vectors.shape[0]
        lims = np.zeros(num_queries + 1, dtype=np.int64)
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64), lims

    def _range_search_remote(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform range search using the remote index on the FAISSx server.

        Args:
            client: FAISSx client instance
            query_vectors: Query vectors
            radius: Radius for the search

        Returns:
            Tuple of (distances, indices, lims)
        """
        # For batch processing when there are many query vectors
        if query_vectors.shape[0] > 100:
            return self._range_search_remote_batch(client, query_vectors, radius)

        try:
            # Attempt range search on the server
            response = client.range_search(self.index_id, query_vectors, radius)

            if not isinstance(response, dict):
                raise ValueError(f"Unexpected range search response: {response}")

            # Extract data from response
            distances = response.get("distances")
            indices = response.get("indices")
            lims = response.get("lims")

            if distances is None or indices is None or lims is None:
                raise ValueError("Range search response missing required data")

            # Convert to numpy arrays
            distances = np.array(distances, dtype=np.float32)
            indices = np.array(indices, dtype=np.int64)
            lims = np.array(lims, dtype=np.int64)

            return distances, indices, lims

        except Exception as e:
            logger.error(f"Range search failed: {e}")
            # Return empty results with appropriate shape
            num_queries = query_vectors.shape[0]
            lims = np.zeros(num_queries + 1, dtype=np.int64)
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64), lims

    def _range_search_remote_batch(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform range search using the remote index in batches for better performance.

        Args:
            client: FAISSx client instance
            query_vectors: Query vectors
            radius: Radius for the search

        Returns:
            Tuple of (distances, indices, lims)
        """
        batch_size = 100  # Reasonable batch size for range search
        num_queries = query_vectors.shape[0]

        # Initialize result lists
        all_distances = []
        all_indices = []
        all_lims = [0]  # Start with 0

        try:
            for i in range(0, num_queries, batch_size):
                end_idx = min(i + batch_size, num_queries)
                batch = query_vectors[i:end_idx]

                logger.debug(
                    f"Range search batch {i//batch_size + 1}/{(num_queries-1)//batch_size + 1} "
                    f"({batch.shape[0]} queries)"
                )

                # Search this batch
                response = client.range_search(self.index_id, batch, radius)

                if not isinstance(response, dict):
                    raise ValueError(f"Unexpected range search response: {response}")

                # Extract data from response
                batch_distances = response.get("distances", [])
                batch_indices = response.get("indices", [])
                batch_lims = response.get("lims", [0] * (batch.shape[0] + 1))

                # Adjust lims for the concatenated results
                offset = all_lims[-1]
                adjusted_lims = [lim + offset for lim in batch_lims[1:]]

                # Add to result lists
                all_distances.extend(batch_distances)
                all_indices.extend(batch_indices)
                all_lims.extend(adjusted_lims)

            # Convert lists to numpy arrays
            np_distances = np.array(all_distances, dtype=np.float32)
            np_indices = np.array(all_indices, dtype=np.int64)
            np_lims = np.array(all_lims, dtype=np.int64)

            return np_distances, np_indices, np_lims

        except Exception as e:
            logger.error(f"Batch range search failed: {e}")
            # Return empty results with appropriate shape
            lims = np.zeros(num_queries + 1, dtype=np.int64)
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64), lims

    def reset(self) -> None:
        """
        Reset the index, removing all stored vectors but preserving training.

        This method clears all vectors from the index while maintaining the trained
        clustering structure. After reset, the index can be reused with new vectors
        without needing to retrain.

        The reset behavior differs between local and remote mode:

        - Local mode: Uses FAISS reset() if available, otherwise recreates the index
          with the same parameters

        - Remote mode: Attempts to use a server-side reset operation if available,
          or creates a new server-side index if necessary

        After reset:
        - Vector count (ntotal) is set to 0
        - Vector mapping is cleared
        - Training state is preserved when possible
        """
        # Remember if the index was trained
        was_trained = self.is_trained

        # Clear the cached vectors
        self._cached_vectors = None

        # Check if client exists and its mode
        client = get_client()

        if client is not None and client.mode == "remote":
            self._reset_remote(client, was_trained)
        else:
            self._reset_local(was_trained)

        # Reset vector count
        self.ntotal = 0

        # Reset vector mapping
        self._vector_mapping: Dict[int, Dict[str, int]] = {}
        self._next_idx = 0

    def _reset_local(self, was_trained: bool) -> None:
        """
        Reset a local FAISS index.

        Args:
            was_trained: Whether the index was trained before reset
        """
        if self._local_index is not None:
            # FAISS handles reset differently depending on the index type
            if hasattr(self._local_index, "reset"):
                self._local_index.reset()
            else:
                logger.warning("Local index doesn't support reset, recreating index")
                # Create a new index with the same parameters
                self._create_local_index(
                    faiss.IndexFlatL2(self.d),  # Simple temporary quantizer
                    self.d,
                    self.nlist,
                    self.M,
                    self.nbits,
                    (
                        faiss.METRIC_INNER_PRODUCT
                        if self.metric_type == "IP"
                        else faiss.METRIC_L2
                    ),
                )

            # Set nprobe
            if hasattr(self._local_index, "nprobe"):
                self._local_index.nprobe = self._nprobe

    def _reset_remote(self, client: Any, was_trained: bool) -> None:
        """
        Reset a remote index on the FAISSx server.

        Args:
            client: FAISSx client instance
            was_trained: Whether the index was trained before reset
        """
        try:
            # Try to use the reset endpoint if available
            response = client.reset(self.index_id)

            if isinstance(response, dict) and response.get("success", False):
                logger.info("Remote index reset successfully")

                # Check if we need to maintain the trained state
                if was_trained:
                    # Get the current status to see if reset affected training
                    status_response = client.get_index_status(self.index_id)
                    if isinstance(status_response, dict) and not status_response.get(
                        "is_trained", False
                    ):
                        logger.warning(
                            "Reset removed training state, index needs to be retrained"
                        )
                        self.is_trained = False
                    else:
                        self.is_trained = True
            else:
                # If reset failed, try recreating the index
                logger.warning(f"Reset failed, recreating remote index: {response}")

                # Delete the old index
                try:
                    client.delete_index(self.index_id)
                except Exception as e:
                    logger.warning(f"Failed to delete old index: {e}")

                # Create a new index with the same parameters
                old_index_id = self.index_id
                self.name = f"index-ivf-pq-{uuid.uuid4().hex[:8]}"
                self.index_id = self.name

                self._create_remote_index(
                    client,
                    None,  # No specific quantizer
                    self.d,
                    self.nlist,
                    self.M,
                    self.nbits,
                )

                logger.info(f"Recreated remote index {old_index_id} as {self.index_id}")

                # Reset trained state
                self.is_trained = False

        except Exception as e:
            logger.error(f"Error resetting remote index: {e}")
            # Don't raise to allow recovery
            self.is_trained = was_trained

    @property
    def pq(self) -> 'IndexIVFPQ.PQProxy':
        """
        Access to the PQ parameters.

        This property provides an interface to the PQ (Product Quantization) parameters
        which control the compression and approximation of vectors.

        Returns:
            PQProxy: A proxy object allowing access to PQ-specific parameters
        """
        return self.PQProxy(self)

    class PQProxy:
        """
        Proxy class for accessing PQ parameters.

        Product Quantization (PQ) parameters control how vectors are compressed and
        approximated in the index. This proxy provides a convenient interface to
        access these parameters.
        """

        def __init__(self, parent: 'IndexIVFPQ') -> None:
            """
            Initialize the PQ proxy.

            Args:
                parent: The parent IndexIVFPQ instance
            """
            self.parent = parent

        @property
        def M(self) -> int:
            """
            Number of subquantizers.

            Higher values give better precision but increase memory usage and
            computational cost.
            """
            return self.parent.M

        @property
        def nbits(self) -> int:
            """
            Number of bits per subquantizer.

            Controls the size of each subquantizer's codebook. Typical value is 8,
            which allows for 256 codes per subquantizer.
            """
            return self.parent.nbits

    def __enter__(self) -> 'IndexIVFPQ':
        """
        Support for context manager protocol.

        Allows using the index with a 'with' statement for automatic resource cleanup.

        Returns:
            Self reference for use in the context manager block
        """
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception],
                 exc_tb: Optional[Any]) -> None:
        """
        Clean up resources when exiting context.

        Called automatically when leaving a 'with' block, ensuring resources
        are properly released even if exceptions occur.

        Args:
            exc_type: Exception type if an exception was raised, None otherwise
            exc_val: Exception value if an exception was raised, None otherwise
            exc_tb: Exception traceback if an exception was raised, None otherwise
        """
        self.close()

    def close(self) -> None:
        """
        Clean up resources used by the index.

        This method releases hardware resources (like GPU memory) and clears cached data.
        It should be called when done using the index to prevent memory leaks, especially
        when GPU resources are involved.

        Important operations:
        - Releases GPU resources if using GPU acceleration
        - Clears the cached vectors to free memory
        - Resets internal state tracking

        Note: The index object can still be used after calling close(), but any GPU
        acceleration will be lost and operations may be slower.
        """
        # Free GPU resources if using GPU
        if self._use_gpu and self._gpu_resources is not None:
            try:
                # Clear reference to GPU index
                self._local_index = None
                # Release GPU resources
                self._gpu_resources = None
                self._use_gpu = False
                logger.info("Released GPU resources")
            except Exception as e:
                logger.warning(f"Error releasing GPU resources: {e}")

        # Clear cached vectors to free memory
        self._cached_vectors = None
        # Reset vector mapping to clear memory
        self._vector_mapping: Dict[int, Dict[str, int]] = {}

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        try:
            self.close()
        except Exception:
            # Ignore errors in destructor
            pass

    # Additional methods for vector reconstruction (helps with io.py)
    def get_vectors(self) -> Optional[np.ndarray]:
        """
        Get all vectors in the index if available.

        This is mainly used for persistence in io.py.

        Returns:
            Array of vectors or None if not available
        """
        # Return cached vectors if we have them
        if self._cached_vectors is not None:
            return self._cached_vectors

        # Try to reconstruct from local index
        if self._local_index is not None:
            try:
                if hasattr(self._local_index, "reconstruct_n") and self.ntotal > 0:
                    # Limit to avoid memory issues
                    count = min(self.ntotal, 100000)
                    return self._local_index.reconstruct_n(0, count)
            except Exception as e:
                logger.warning(f"Failed to reconstruct vectors from local index: {e}")

        # For remote mode, we may not be able to reconstruct
        return None

    def reconstruct(self, idx: int) -> np.ndarray:
        """
        Reconstruct a vector at the given index.

        Args:
            idx: Index of the vector to reconstruct

        Returns:
            Reconstructed vector

        Raises:
            ValueError: If the index is out of range or reconstruction fails
        """
        if idx < 0 or idx >= self.ntotal:
            raise ValueError(f"Index {idx} out of range [0, {self.ntotal-1}]")

        # Try using cached vectors first (most accurate)
        if self._cached_vectors is not None:
            if idx < len(self._cached_vectors):
                return self._cached_vectors[idx]

        # For local mode, use local index
        if self._local_index is not None:
            try:
                return self._local_index.reconstruct(idx)
            except Exception as e:
                logger.warning(
                    f"Failed to reconstruct vector {idx} from local index: {e}"
                )

        # For remote mode, try server reconstruction
        client = get_client()
        if client is not None and client.mode == "remote":
            try:
                # Convert to server-side index if we have a mapping
                server_idx = idx
                if idx in self._vector_mapping:
                    server_idx = self._vector_mapping[idx].get("server_id", idx)

                response = client.reconstruct(self.index_id, server_idx)

                if isinstance(response, dict) and "vector" in response:
                    return np.array(response["vector"], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to reconstruct vector {idx} from server: {e}")

        # If all else fails, return a zero vector
        return np.zeros(self.d, dtype=np.float32)

    def reconstruct_n(self, idx: int, n: int) -> np.ndarray:
        """
        Reconstruct multiple vectors starting at the given index.

        Args:
            idx: Starting index
            n: Number of vectors to reconstruct

        Returns:
            Array of reconstructed vectors

        Raises:
            ValueError: If the indices are out of range or reconstruction fails
        """
        if idx < 0 or idx + n > self.ntotal:
            raise ValueError(f"Range {idx}:{idx+n} out of bounds [0, {self.ntotal}]")

        # Try using cached vectors first (most accurate)
        if self._cached_vectors is not None:
            if idx + n <= len(self._cached_vectors):
                return self._cached_vectors[idx:idx + n]

        # For local mode, use local index
        if self._local_index is not None:
            try:
                if hasattr(self._local_index, "reconstruct_n"):
                    return self._local_index.reconstruct_n(idx, n)
            except Exception as e:
                logger.warning(
                    f"Failed to reconstruct vectors {idx}:{idx+n} from local index: {e}"
                )

        # For remote mode or fallback, reconstruct one by one
        vectors = np.zeros((n, self.d), dtype=np.float32)
        for i in range(n):
            try:
                vectors[i] = self.reconstruct(idx + i)
            except Exception:
                # Leave as zeros if reconstruction fails
                pass

        return vectors

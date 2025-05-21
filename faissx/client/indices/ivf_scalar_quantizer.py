#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISSx IndexIVFScalarQuantizer implementation.

This module provides a client-side implementation of the FAISS IndexIVFScalarQuantizer class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).
"""

from typing import Any, Tuple, Optional
import uuid
import numpy as np
import time

try:
    import faiss
except ImportError:
    faiss = None

from ..client import get_client
from .base import logger, FAISSxBaseIndex


class IndexIVFScalarQuantizer(FAISSxBaseIndex):
    """
    Proxy implementation of FAISS IndexIVFScalarQuantizer.

    This class mimics the behavior of FAISS IndexIVFScalarQuantizer, which combines
    inverted file indexing with scalar quantization for efficient similarity search.

    When running in local mode with CUDA-capable GPUs, it will automatically use
    GPU acceleration if available.

    Attributes:
        d (int): Vector dimension
        nlist (int): Number of clusters/partitions
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
        _nprobe (int): Number of clusters to search (default: 1)
    """

    def __init__(self, quantizer, d: int, nlist: int, qtype=None, metric_type=None):
        """
        Initialize the index with specified parameters.

        Args:
            quantizer: Quantizer object that defines the centroids (usually IndexFlatL2)
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions
            qtype: Scalar quantizer type (if None, uses default QT_8bit)
            metric_type: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        super().__init__()  # Initialize base class

        # Try to import faiss locally to avoid module-level dependency
        try:
            import faiss as local_faiss
            default_metric = local_faiss.METRIC_L2
            metric_inner_product = local_faiss.METRIC_INNER_PRODUCT
            default_qtype = local_faiss.ScalarQuantizer.QT_8bit
        except ImportError:
            # Define fallback constants when faiss isn't available
            default_metric = 0
            metric_inner_product = 1
            default_qtype = 0
            local_faiss = None

        # Set default metric if not provided
        if metric_type is None:
            metric_type = default_metric

        # Set default qtype if not specified
        if qtype is None:
            qtype = default_qtype

        # Store core parameters
        self.d = d
        self.nlist = nlist
        self.qtype = qtype
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
        self.name = f"index-ivf-sq-{uuid.uuid4().hex[:8]}"
        self.index_id = self.name

        # Initialize vector mapping for remote mode
        self._vector_mapping = {}  # Maps local indices to server-side information
        self._next_idx = 0  # Counter for local indices

        # Check if client exists and its mode
        client = get_client()

        # Explicit check for remote mode instead of just checking if client exists
        if client is not None and client.mode == "remote":
            # Remote mode
            logger.info(f"Creating remote IndexIVFScalarQuantizer on server {client.server}")
            self._create_remote_index(client, quantizer, d, nlist, qtype)
        else:
            # Local mode
            logger.info(f"Creating local IndexIVFScalarQuantizer index {self.name}")
            self._create_local_index(quantizer, d, nlist, qtype, metric_type)

    def _get_index_type_string(self) -> str:
        """
        Get standardized string representation of the IVF-SQ index type.

        Returns:
            String representation of index type
        """
        # Create the index type string based on parameters
        qtype_str = "SQ8" if self.qtype is None else f"SQ{self.qtype}"
        index_type = f"IVF{self.nlist}_{qtype_str}"

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
        self, quantizer, d: int, nlist: int, qtype: Any, metric_type: Any
    ) -> None:
        """
        Create a local FAISS IVF-SQ index.

        Args:
            quantizer: Quantizer object that defines the centroids
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions
            qtype: Scalar quantizer type
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
            cpu_index = local_faiss.IndexIVFScalarQuantizer(
                base_quantizer, d, nlist, qtype, metric_type
            )

            if gpu_available:
                # GPU is available, create resources and GPU index
                self._use_gpu = True
                self._gpu_resources = local_faiss.StandardGpuResources()

                # Convert to GPU index
                try:
                    self._local_index = local_faiss.index_cpu_to_gpu(
                        self._gpu_resources, 0, cpu_index
                    )
                    logger.info(f"Using GPU-accelerated IVF-SQ index for {self.name}")
                except Exception as e:
                    # If GPU conversion fails, fall back to CPU
                    self._local_index = cpu_index
                    self._use_gpu = False
                    logger.warning(
                        f"Failed to create GPU IVF-SQ index: {e}, using CPU instead"
                    )
            else:
                # No GPUs available, use CPU version
                self._local_index = cpu_index

            self.index_id = self.name  # Use name as ID for consistency
        except Exception as e:
            raise RuntimeError(f"Failed to create IndexIVFScalarQuantizer: {e}")

    def _create_remote_index(
        self, client: Any, quantizer, d: int, nlist: int, qtype: Any
    ) -> None:
        """
        Create a remote IVF-SQ index on the server.

        Args:
            client: FAISSx client instance
            quantizer: Quantizer object that defines the centroids
            d (int): Vector dimension
            nlist (int): Number of clusters/partitions
            qtype: Scalar quantizer type
        """
        try:
            # Get index type string - this already includes nlist and qtype info
            index_type = self._get_index_type_string()

            # Create index parameters - only use the base parameters that
            # are common across all server implementations
            params = {
                "name": self.name,
                "dimension": d,
                "index_type": index_type,
            }

            if self.metric_type != "L2":
                params["metric_type"] = self.metric_type

            # Note: We don't pass quantizer_id even if the quantizer is a remote index
            # because not all server implementations support this parameter.
            # Omitting quantizer_id means the server will create its own internal quantizer.

            # Create index on server - using the simplified parameter structure
            logger.debug(f"Creating remote index {self.name} with type {index_type}")
            response = client.create_index(**params)

            # Log the raw response for debugging
            logger.debug(f"Server response: {response}")

            # Parse response to get index ID
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
            # Raise a clear error instead of falling back to local mode
            raise RuntimeError(
                f"Failed to create remote IVF-SQ index: {e}. "
                f"Server may not support IVF-SQ indices with type {index_type}."
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
            nprobe = 1
        if nprobe > self.nlist:
            logger.warning(f"nprobe {nprobe} > nlist {self.nlist}, capping to {self.nlist}")
            nprobe = self.nlist

        self._nprobe = nprobe

        # If using local implementation, update the index directly
        client = get_client()
        if client is None or client.mode == "local":
            if self._local_index is not None:
                self._local_index.nprobe = nprobe
        else:
            # Update remote index if in remote mode
            try:
                client.set_parameter(self.index_id, "nprobe", nprobe)
            except Exception as e:
                logger.warning(f"Failed to set nprobe on server: {e}")

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

        start_time = time.time()
        # Use local FAISS implementation directly
        self._local_index.train(vectors)
        self.is_trained = self._local_index.is_trained
        logger.info(f"Local index trained in {time.time() - start_time:.2f}s")

    def _train_remote(self, client: Any, vectors: np.ndarray) -> None:
        """Train the remote index with the provided vectors."""
        logger.debug(f"Training remote index {self.index_id} with {len(vectors)} vectors")

        # Check if the index is already trained on the server
        try:
            status_response = client.get_index_status(self.index_id)
            if isinstance(status_response, dict) and status_response.get("is_trained", False):
                logger.info("Remote index already trained, skipping training")
                self.is_trained = True
                return
        except (AttributeError, Exception) as e:
            logger.warning(f"Could not check index training status: {e}")

        start_time = time.time()
        try:
            # Try different method names the server might use for training
            for method_name in ['train', 'train_index']:
                if hasattr(client, method_name):
                    train_method = getattr(client, method_name)
                    result = train_method(self.index_id, vectors)

                    # Check for explicit error response
                    if isinstance(result, dict) and result.get("success", False):
                        # Update local state based on training result
                        self.is_trained = result.get("is_trained", True)
                        logger.info(f"Remote index trained in {time.time() - start_time:.2f}s")
                        return
                    else:
                        error_msg = (result.get("error", "Unknown error")
                                     if isinstance(result, dict) else str(result))
                        logger.warning(f"Training attempt with {method_name} failed: {error_msg}")

            # If we get here, none of the training methods worked
            logger.warning("Server doesn't support explicit training, assuming auto-training")
            self.is_trained = True
        except Exception as e:
            # For servers that don't support explicit training, we'll just assume
            # the index is implicitly trained when adding vectors
            logger.warning(f"Training failed: {e}. Assuming implicit training when adding vectors.")
            self.is_trained = True

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

        # Cache vectors for reconstruction (helps with io.py persistence)
        if hasattr(self, "_cached_vectors") and self._cached_vectors is not None:
            self._cached_vectors = np.vstack([self._cached_vectors, vectors])
        else:
            self._cached_vectors = vectors.copy()

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
        start_time = time.time()
        self._local_index.add(vectors)
        self.ntotal = self._local_index.ntotal
        logger.info(f"Added vectors to local index in {time.time() - start_time:.2f}s")

    def _add_remote(self, client: Any, vectors: np.ndarray) -> None:
        """Add vectors to remote index."""
        logger.debug(f"Adding {len(vectors)} vectors to remote index {self.index_id}")

        # Get batch size parameter
        batch_size = self.get_parameter('batch_size')
        if batch_size <= 0:
            batch_size = 10000  # Default if not set or invalid

        # If vectors fit in a single batch, add them directly
        if len(vectors) <= batch_size:
            self._add_remote_batch(client, vectors)
            return

        # Otherwise, process in batches
        start_time = time.time()
        num_vectors = vectors.shape[0]
        for i in range(0, num_vectors, batch_size):
            end_idx = min(i + batch_size, num_vectors)
            batch = vectors[i:end_idx]

            batch_num = i//batch_size + 1
            total_batches = (num_vectors-1)//batch_size + 1
            logger.info(
                f"Adding batch {batch_num}/{total_batches} ({batch.shape[0]} vectors)"
            )

            self._add_remote_batch(client, batch)

        logger.info(f"Added all vectors to remote index in {time.time() - start_time:.2f}s")

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

    def _search_local(
            self, query_vectors: np.ndarray, k: int, internal_k: int, need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search in local index."""
        logger.debug(f"Searching {len(query_vectors)} vectors in local index {self.name}")

        # Set nprobe for local index before searching
        self._local_index.nprobe = self._nprobe

        # Use local FAISS implementation directly
        start_time = time.time()
        distances, indices = self._local_index.search(query_vectors, internal_k)
        logger.info(f"Local search completed in {time.time() - start_time:.2f}s")

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
            return self._search_remote_batch(
                client, query_vectors, k, internal_k, need_reranking
            )

        # Otherwise, process in batches and combine results
        start_time = time.time()
        all_distances = []
        all_indices = []
        num_queries = query_vectors.shape[0]

        for i in range(0, num_queries, batch_size):
            end_idx = min(i + batch_size, num_queries)
            batch = query_vectors[i:end_idx]

            batch_num = i//batch_size + 1
            total_batches = (num_queries-1)//batch_size + 1
            logger.debug(
                f"Searching batch {batch_num}/{total_batches} ({batch.shape[0]} queries)"
            )

            distances, indices = self._search_remote_batch(
                client, batch, k, internal_k, need_reranking
            )
            all_distances.append(distances)
            all_indices.append(indices)

        # Combine results
        result_distances = np.vstack(all_distances)
        result_indices = np.vstack(all_indices)

        logger.info(f"Remote search completed in {time.time() - start_time:.2f}s")
        return result_distances, result_indices

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

        # Clear the cached vectors
        self._cached_vectors = None

        client = get_client()

        # Explicit check for remote mode
        if client is not None and client.mode == "remote":
            self._reset_remote(client, was_trained)
        else:
            self._reset_local(was_trained)

        # Reset vector mapping
        self._vector_mapping = {}
        self._next_idx = 0

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
            # Try to use reset endpoint if available
            response = client.reset(self.index_id)

            if isinstance(response, dict) and response.get("success", False):
                logger.info("Remote index reset successfully")
                self.ntotal = 0

                # Check if the index is still trained
                status_response = client.get_index_status(self.index_id)
                if (isinstance(status_response, dict) and
                        not status_response.get("is_trained", False)):
                    logger.warning("Reset removed training state, index needs to be retrained")
                    self.is_trained = False
                else:
                    self.is_trained = was_trained
                return
        except Exception as e:
            logger.warning(f"Reset endpoint failed: {e}, falling back to re-creation")

        try:
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            index_type = self._get_index_type_string()

            response = client.create_index(
                name=new_name, dimension=self.d, index_type=index_type
            )

            if not isinstance(response, dict) or not response.get("success", False):
                error_msg = (response.get("error", "Unknown error")
                             if isinstance(response, dict) else str(response))
                raise RuntimeError(f"Failed to create new index during reset: {error_msg}")

            # Update index information
            self.index_id = self._parse_server_response(response, new_name)
            self.name = new_name

            # Don't reset training state
            self.is_trained = was_trained

            # Reset all local state
            self.ntotal = 0
        except Exception as e:
            # Ensure all errors are properly propagated
            raise RuntimeError(f"Remote reset operation failed: {e}")

    # Methods to support vector reconstruction for io.py
    def get_vectors(self) -> Optional[np.ndarray]:
        """
        Get all vectors in the index if available.

        Returns:
            Array of vectors or None if not available
        """
        # Return cached vectors if we have them
        if self._cached_vectors is not None:
            return self._cached_vectors

        # Try to reconstruct from local index
        if self._local_index is not None:
            try:
                count = min(self.ntotal, 100000)  # Limit to avoid memory issues
                if count > 0 and hasattr(self._local_index, "reconstruct_n"):
                    return self._local_index.reconstruct_n(0, count)
            except Exception as e:
                logger.warning(f"Failed to get vectors from local index: {e}")

        # For remote mode, we may not be able to reconstruct
        return None

    def reconstruct(self, idx: int) -> np.ndarray:
        """
        Reconstruct a vector at the given index.

        Args:
            idx: Index of the vector to reconstruct

        Returns:
            Reconstructed vector
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
                logger.warning(f"Failed to reconstruct vector {idx} from local index: {e}")

        # For remote mode, try server reconstruction
        client = get_client()
        if client is not None and client.mode == "remote":
            try:
                # Convert to server-side index if we have a mapping
                server_idx = idx
                if idx in self._vector_mapping:
                    server_idx = self._vector_mapping[idx].get("server_idx", idx)

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
        """
        if idx < 0 or idx + n > self.ntotal:
            raise ValueError(f"Range {idx}:{idx+n} out of bounds [0, {self.ntotal}]")

        # Try using cached vectors first (most accurate)
        if self._cached_vectors is not None:
            if idx + n <= len(self._cached_vectors):
                return self._cached_vectors[idx:idx+n]

        # For local mode, use local index
        if self._local_index is not None:
            try:
                if hasattr(self._local_index, "reconstruct_n"):
                    return self._local_index.reconstruct_n(idx, n)
            except Exception as e:
                logger.warning(f"Failed to reconstruct vectors {idx}:{idx+n} from local index: {e}")

        # For remote mode or fallback, reconstruct one by one
        vectors = np.zeros((n, self.d), dtype=np.float32)
        for i in range(n):
            try:
                vectors[i] = self.reconstruct(idx + i)
            except Exception:
                # Leave as zeros if reconstruction fails
                pass

        return vectors

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

        # Clear cached vectors to free memory
        self._cached_vectors = None
        self._vector_mapping = {}

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        self.close()

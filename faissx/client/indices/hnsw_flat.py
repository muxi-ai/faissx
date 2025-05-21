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
FAISSx IndexHNSWFlat implementation.

This module provides a client-side implementation of the FAISS IndexHNSWFlat class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).

The Hierarchical Navigable Small World (HNSW) algorithm:
- Creates a multi-layer graph structure where similar vectors are connected
- Provides logarithmic search complexity O(log N) through navigable small world graphs
- Maintains multiple layers with decreasing density for efficient approximate search
- Balances search accuracy and speed through configurable parameters (efSearch, M)
- Excels at high-dimensional similarity search with better performance than many alternatives

Key advantages:
- Excellent query performance even with millions of vectors
- Good accuracy vs. speed trade-off (configurable via parameters)
- Scales well for high-dimensional data
- Supports incremental updates (vectors can be added after index creation)
- Low memory overhead compared to some other graph-based methods

Module features:
- Support for both local and remote execution modes
- GPU acceleration when available in local mode
- Batch processing for efficient vector operations
- Compatible parameters with standard FAISS HNSW implementation
"""

import uuid
import numpy as np
from typing import Tuple, Any, Optional, List, Dict, TypeVar, cast

from ..client import get_client
from .base import logger, FAISSxBaseIndex


# Default values for common parameters
DEFAULT_BATCH_SIZE = 1000  # Default batch size for adding vectors
DEFAULT_SEARCH_BATCH_SIZE = 100  # Default batch size for search operations
DEFAULT_SEARCH_FACTOR = 1.0  # Multiplier for internal_k vs requested k
DEFAULT_EF_SEARCH = 16  # Default HNSW exploration factor during search (higher = more accurate)
DEFAULT_EF_CONSTRUCTION = 40  # Default HNSW exploration during construction (higher = better index)
DEFAULT_M = 32  # Default connections per node (higher = better accuracy, more memory)

# Type for self reference in HNSWParameters
T = TypeVar('T', bound='IndexHNSWFlat')


class HNSWParameters:
    """
    Proxy for FAISS HNSW parameters.

    This class provides a consistent interface for accessing and modifying HNSW-specific parameters,
    whether using local or remote mode. It acts as a bridge between the user-facing API and
    the underlying FAISS implementation or remote server communication.

    The HNSW algorithm has several critical parameters that affect its performance and accuracy:

    - efSearch: Controls the trade-off between search accuracy and speed. Higher values increase
      search accuracy but also increase search time. This parameter determines how extensively
      the HNSW graph is explored during search.

    - efConstruction: Affects the quality of the HNSW graph built during index construction.
      Higher values create better quality graphs (potentially yielding more accurate search results)
      but slow down the construction process.

    - M: Number of connections per node in the HNSW graph. Higher values give better accuracy
      but increase memory usage and construction time. This parameter must be set at index
      creation time and cannot be changed afterwards.

    Typical values for efSearch range from 16-128 depending on accuracy requirements.
    Typical values for efConstruction range from 40-200 for balancing build time and quality.
    """
    def __init__(self, parent_index: 'IndexHNSWFlat') -> None:
        """
        Initialize the HNSW parameters proxy.

        Sets up default parameter values based on established best practices for HNSW indices.
        These values provide a good starting point for most applications but can be tuned
        for specific use cases.

        Args:
            parent_index: The parent IndexHNSWFlat instance these parameters belong to
        """
        self.parent_index: 'IndexHNSWFlat' = parent_index
        # Default efSearch value - affects search accuracy vs speed
        # Higher values = more accurate results, slower search
        self._efSearch: int = DEFAULT_EF_SEARCH  # Default in FAISS

        # Default efConstruction value - affects index build quality
        # Higher values = better index quality, slower construction
        self._efConstruction: int = DEFAULT_EF_CONSTRUCTION  # Default in FAISS

        # M parameter - number of connections per node
        # This should match parent's M, as it was set during index creation
        self._M: int = parent_index.M if hasattr(parent_index, 'M') else DEFAULT_M

    @property
    def efSearch(self) -> int:
        """
        Get the efSearch parameter.

        The efSearch parameter controls how extensively the HNSW graph is explored during search.
        It directly affects the trade-off between:
        - Search accuracy: Higher values increase the chance of finding the true nearest neighbors
        - Search speed: Higher values increase the time spent exploring the graph

        In local mode, this property reads directly from the underlying FAISS index to ensure
        consistency. In remote mode, it returns the cached value.

        Returns:
            int: Current efSearch value
        """
        if not self.parent_index._using_remote and self.parent_index._local_index is not None:
            # Pass through to the real FAISS index if in local mode
            # This ensures we're always in sync with the actual FAISS index
            return cast(int, self.parent_index._local_index.hnsw.efSearch)
        return self._efSearch

    @efSearch.setter
    def efSearch(self, value: int) -> None:
        """
        Set the efSearch parameter.

        Updates the exploration factor used during search operations. This parameter can be
        adjusted at any time, even after vectors have been added to the index, making it
        useful for runtime performance tuning.

        A common approach is to use a lower value for efSearch when speed is critical and
        increase it when accuracy is more important. The optimal value depends on your data
        distribution and accuracy requirements.

        Args:
            value: New efSearch value (higher = more accurate search, slower)
        """
        self._efSearch = value
        if not self.parent_index._using_remote and self.parent_index._local_index is not None:
            # Update the underlying FAISS index if in local mode
            self.parent_index._local_index.hnsw.efSearch = value

    @property
    def efConstruction(self) -> int:
        """
        Get the efConstruction parameter.

        The efConstruction parameter controls how extensively the HNSW graph is explored during
        index construction. It affects:
        - Index quality: Higher values create better connected graphs with more optimal paths
        - Build speed: Higher values increase the time required to build the index

        A well-constructed graph directly impacts search performance and accuracy. In local mode,
        this property reads directly from the underlying FAISS index.

        Returns:
            int: Current efConstruction value
        """
        if not self.parent_index._using_remote and self.parent_index._local_index is not None:
            # Pass through to the real FAISS index if in local mode
            return cast(int, self.parent_index._local_index.hnsw.efConstruction)
        return self._efConstruction

    @efConstruction.setter
    def efConstruction(self, value: int) -> None:
        """
        Set the efConstruction parameter.

        Updates the exploration factor used during index construction. This parameter primarily
        affects new vectors being added to the index rather than existing ones. Setting a higher
        value will create a higher quality HNSW graph but will slow down vector additions.

        Note that changing this value is most effective before adding vectors to the index,
        as it primarily influences the graph construction process during vector addition.

        Args:
            value: New efConstruction value (higher = better index, slower build)
        """
        self._efConstruction = value
        if not self.parent_index._using_remote and self.parent_index._local_index is not None:
            # Update the underlying FAISS index if in local mode
            self.parent_index._local_index.hnsw.efConstruction = value

    @property
    def M(self) -> int:
        """
        Get the M parameter (read-only).

        M determines the number of connections per node in the HNSW graph, which is a
        fundamental parameter in the HNSW algorithm. It directly affects:
        - Search accuracy: Higher values generally yield more accurate results
        - Memory usage: Higher values increase the memory footprint of the index
        - Construction time: Higher values slow down index building

        This parameter is read-only because it must be set at index creation time and
        cannot be modified afterward due to its fundamental role in the graph structure.

        Returns:
            int: Current M value
        """
        # Simply return the M value from the parent index
        # This is more reliable than trying to access it through the local index
        # since M is a structural parameter that can't change after creation
        return self.parent_index.M


class IndexHNSWFlat(FAISSxBaseIndex):
    """
    Proxy implementation of FAISS IndexHNSWFlat.

    This class mimics the behavior of FAISS IndexHNSWFlat, which uses Hierarchical
    Navigable Small World graphs for efficient approximate similarity search. It offers
    excellent search performance with good accuracy, particularly for high-dimensional
    data.

    HNSW works by building a multi-layer graph structure where each vector is a node.
    The graph has connections between similar vectors, allowing for efficient navigation
    during search. The algorithm provides a good balance between search speed and accuracy.

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
        hnsw: Access to HNSW-specific parameters
    """

    def __init__(self, d: int, M: int = DEFAULT_M, metric: Any = None) -> None:
        """
        Initialize the HNSW index with specified parameters.

        This constructor sets up either a local FAISS index or connects to a remote FAISSx server
        based on the client configuration. For local indices, it attempts to use GPU acceleration
        if available.

        The HNSW algorithm requires the M parameter (number of connections per node) to be set
        at index creation time and cannot be changed afterwards. Higher M values generally
        result in better search accuracy at the cost of increased memory usage and longer
        construction time.

        Args:
            d: Vector dimension
            M: Number of connections per node (higher = better accuracy, more memory)
            metric: Distance metric, either faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT

        Raises:
            RuntimeError: If index initialization fails
        """
        # Initialize base class
        super().__init__()

        # Try to import faiss locally to avoid module-level dependency
        try:
            import faiss as local_faiss
            METRIC_L2 = local_faiss.METRIC_L2
            METRIC_INNER_PRODUCT = local_faiss.METRIC_INNER_PRODUCT
        except ImportError:
            # Define fallback constants when faiss isn't available
            # This allows the code to still run in remote mode even without FAISS installed
            METRIC_L2 = 0
            METRIC_INNER_PRODUCT = 1
            local_faiss = None

        # Set default metric if not provided
        if metric is None:
            metric = METRIC_L2

        # Store core parameters
        self.d: int = d
        self.M: int = M
        # Convert metric type to string representation for remote mode
        # This helps with remote server communication
        self.metric_type: str = "IP" if metric == METRIC_INNER_PRODUCT else "L2"

        # Initialize state variables
        self.is_trained: bool = True  # HNSW doesn't need training
        self.ntotal: int = 0

        # Initialize GPU-related attributes
        self._use_gpu: bool = False
        self._gpu_resources: Optional[Any] = None
        self._local_index: Optional[Any] = None
        self._using_remote: bool = False  # Will be set based on client mode

        # Generate unique name for the index
        # This ensures each index has a different identifier even if multiple
        # indices are created with the same parameters
        self.name: str = f"index-hnsw-flat-{uuid.uuid4().hex[:8]}"
        self.index_id: str = self.name

        # Initialize vector mapping for remote mode
        # This is crucial for translating between server indices and client indices
        self._vector_mapping: Dict[int, Dict[str, int]] = {}  # Maps local to server indices
        self._next_idx: int = 0  # Counter for local indices

        # Check if client exists and its mode
        client = get_client()

        if client is not None and client.mode == "remote":
            # Remote mode - we'll create the index on the FAISSx server
            logger.info(f"Creating remote IndexHNSWFlat on server {client.server}")
            self._using_remote = True
            self._create_remote_index(client, d, M)
        else:
            # Local mode - we'll create a FAISS index locally
            logger.info(f"Creating local IndexHNSWFlat index {self.name}")
            self._using_remote = False
            self._create_local_index(d, M, metric)

        # Initialize hnsw property for parameter access
        self.hnsw = HNSWParameters(self)

    def _get_index_type_string(self, M: Optional[int] = None) -> str:
        """
        Get standardized string representation of index type.

        Builds a string identifier for the index type that includes the HNSW parameter M
        and the metric type. This is used for communicating with the server in remote mode.

        Args:
            M: Connections per layer parameter to use instead of self.M

        Returns:
            String representation of index type (e.g. "HNSW32" or "HNSW64_IP")
        """
        # Use internal M if none provided
        if M is None:
            M = self.M

        # Create base type string
        index_type = f"HNSW{M}"

        # Add metric type suffix if needed
        if self.metric_type == "IP":
            index_type = f"{index_type}_IP"

        return index_type

    def _create_local_index(self, d: int, M: int, metric: Any) -> None:
        """
        Create a local FAISS HNSW index.

        Initializes a local FAISS index, attempting to use GPU acceleration if available.
        For HNSW, many operations will still run on CPU even with GPU acceleration.

        Args:
            d (int): Vector dimension
            M (int): Connections per layer parameter
            metric: Distance metric type

        Raises:
            RuntimeError: If FAISS index initialization fails
        """
        try:
            import faiss

            # Try to use GPU if available
            gpu_available = False
            try:
                import faiss.contrib.gpu  # type: ignore

                ngpus = faiss.get_num_gpus()
                gpu_available = ngpus > 0
                if gpu_available:
                    logger.info(f"Found {ngpus} GPU(s) available for HNSW index")
            except (ImportError, AttributeError) as e:
                logger.warning(f"GPU support not available: {e}")
                gpu_available = False

            if gpu_available:
                # GPU is available, create resources and GPU index
                self._use_gpu = True
                self._gpu_resources = faiss.StandardGpuResources()

                # Create CPU index first (HNSW has limited GPU operations)
                cpu_index = faiss.IndexHNSWFlat(d, M, metric)

                # Convert to GPU index - note that many HNSW operations still run on CPU
                try:
                    self._local_index = faiss.index_cpu_to_gpu(
                        self._gpu_resources, 0, cpu_index
                    )
                    logger.info(
                        f"Using GPU-accelerated HNSW index for {self.name} (search only)"
                    )
                except Exception as e:
                    # If GPU conversion fails, fall back to CPU
                    self._local_index = cpu_index
                    self._use_gpu = False
                    logger.warning(
                        f"Failed to create GPU HNSW index: {e}, using CPU instead"
                    )
            else:
                # No GPUs available, use CPU version
                self._local_index = faiss.IndexHNSWFlat(d, M, metric)
                logger.info(f"Using CPU-only HNSW index for {self.name}")

            self.index_id = self.name  # Use name as ID for consistency
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")

    def _create_remote_index(self, client: Any, d: int, M: int) -> None:
        """
        Create a remote HNSW index on the server.

        Sends a request to the FAISSx server to create a new HNSW index with the specified
        parameters.

        Args:
            client: FAISSx client instance
            d (int): Vector dimension
            M (int): Connections per layer parameter

        Raises:
            RuntimeError: If the server fails to create the index
        """
        try:
            # Get index type string
            index_type = self._get_index_type_string(M)

            # Create index on server
            logger.debug(f"Creating remote index {self.name} with type {index_type}")
            response = client.create_index(self.name, d, index_type)

            # Parse response
            if isinstance(response, dict):
                self.index_id = response.get("index_id", self.name)
            else:
                logger.warning(f"Unexpected server response format: {response}")
                self.index_id = self.name

        except Exception as e:
            raise RuntimeError(
                f"Failed to create remote HNSW index: {e}. "
                f"Server may not support HNSW indices with type {self._get_index_type_string(M)}."
            )

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Prepare vectors for indexing or search.

        Converts input vectors to the format required by FAISS (numpy array with float32 dtype).
        This ensures consistent handling of vector data regardless of input format and avoids
        potential issues with data type mismatches between client and server.

        FAISS requires all vectors to be float32 for optimal performance and compatibility.
        This method handles the conversion automatically to provide a smoother user experience.

        Args:
            vectors: Input vectors as numpy array or convertible type

        Returns:
            Normalized array with proper dtype (float32)
        """
        # Convert non-numpy inputs to numpy arrays
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        # Convert to float32 if needed (FAISS requirement)
        # If vectors are already float32, this is a no-op (no copy is made)
        return vectors.astype(np.float32) if vectors.dtype != np.float32 else vectors

    def _map_server_to_local_indices(self, server_indices: List[int]) -> np.ndarray:
        """
        Convert server-side indices to local indices.

        In remote mode, indices returned by the server need to be mapped back to the local
        indices used by the client application. This function performs that bidirectional mapping
        to ensure consistent index handling across local and remote operations.

        The mapping is necessary because the client and server may have different indexing schemes:
        - Server indices are sequential IDs in the server's vector store
        - Local indices are sequential IDs assigned by the client when vectors were added

        Args:
            server_indices: List of server-side indices to map to local indices

        Returns:
            Array of corresponding local indices, with -1 for indices not found in the mapping
        """
        # Initialize array with -1 (not found) values
        # Using -1 makes it easy to identify unmatched indices later
        local_indices = np.full(len(server_indices), -1, dtype=np.int64)

        # Create a reverse mapping for faster lookup
        # This inverts our existing mapping for O(1) lookups instead of O(n)
        server_to_local: Dict[int, int] = {}
        for local_idx, info in self._vector_mapping.items():
            server_idx = info.get("server_idx")
            if server_idx is not None:
                server_to_local[server_idx] = local_idx

        # Map each server index to its corresponding local index
        # If not found in mapping, the default -1 value will remain
        for i, server_idx in enumerate(server_indices):
            local_indices[i] = server_to_local.get(server_idx, -1)

        return local_indices

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        This method routes the add operation to either local or remote implementation
        based on the client configuration. Vectors will be properly prepared and validated
        before being added to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension
            RuntimeError: If adding vectors fails
        """
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Prepare vectors
        vectors = self._prepare_vectors(x)

        client = get_client()

        if client is not None and client.mode == "remote":
            self._add_remote(client, vectors)
        else:
            self._add_local(vectors)

    def _add_local(self, vectors: np.ndarray) -> None:
        """
        Add vectors to local index.

        Direct pass-through to the local FAISS index's add method.

        Args:
            vectors: Vectors to add, already prepared and validated

        Note:
            Updates ntotal after adding vectors to maintain consistency
        """
        logger.debug(f"Adding {len(vectors)} vectors to local index {self.name}")
        self._local_index.add(vectors)
        self.ntotal = self._local_index.ntotal

    def _add_remote(self, client: Any, vectors: np.ndarray) -> None:
        """
        Add vectors to remote index with batch processing.

        Handles batching of large vector sets to avoid overwhelming the server with a single
        large request. Uses the batch_size parameter to determine the optimal batch size.

        Args:
            client: The FAISSx client
            vectors: Vectors to add, already prepared and validated

        Raises:
            RuntimeError: If adding vectors to the remote index fails
        """
        logger.debug(f"Adding {len(vectors)} vectors to remote index {self.index_id}")

        # Get the batch size for adding vectors
        batch_size = self._get_batch_size('default')

        # If vectors fit in a single batch, add directly
        if len(vectors) <= batch_size:
            self._add_remote_batch(client, vectors)
            return

        # Process in larger batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:min(i + batch_size, len(vectors))]
            self._add_remote_batch(client, batch)

    def _add_remote_batch(self, client: Any, vectors: np.ndarray) -> None:
        """
        Add a batch of vectors to remote index.

        Sends a single batch of vectors to the server and updates local tracking information
        for successful additions.

        Args:
            client: The FAISSx client
            vectors: Batch of vectors to add

        Raises:
            RuntimeError: If the server returns an error or the response is invalid
        """
        try:
            # Send request to server
            response = client.add_vectors(self.index_id, vectors)

            # Validate response and update tracking
            if isinstance(response, dict) and response.get("success", False):
                added_count = response.get("count", 0)

                # Create mapping for each added vector
                start_idx = self.ntotal
                for i in range(added_count):
                    server_idx = start_idx + i
                    local_idx = self._next_idx

                    self._vector_mapping[local_idx] = {
                        "local_idx": local_idx,
                        "server_idx": server_idx,
                    }
                    self._next_idx += 1

                # Update total vector count
                self.ntotal += added_count
                logger.debug(f"Added {added_count} vectors to remote index {self.index_id}")
            else:
                # Handle error or unexpected response
                error_msg = (response.get("error", "Unknown server error")
                             if isinstance(response, dict) else "Invalid response format")
                logger.warning(f"Error adding vectors: {error_msg}")
                raise RuntimeError(f"Failed to add vectors: {error_msg}")

        except Exception as e:
            logger.error(f"Error adding vectors to remote index: {e}")
            raise RuntimeError(f"Failed to add vectors to remote index: {e}")

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.

        Performs approximate nearest neighbor search using the HNSW algorithm.
        Returns the k closest vectors to each query vector according to the metric type.

        Args:
            x: Query vectors, shape (n, d)
            k: Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Distances array of shape (n, k)
                - Indices array of shape (n, k)

        Raises:
            ValueError: If query vector shape doesn't match index dimension
            RuntimeError: If search operation fails
        """
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Prepare vectors
        query_vectors = self._prepare_vectors(x)

        # For HNSW, we can sometimes get better results by requesting more neighbors
        # than we actually need, then re-ranking them
        try:
            need_reranking = bool(self.get_parameter('rerank_results'))
        except ValueError:
            need_reranking = False

        try:
            search_factor = float(self.get_parameter('search_factor'))
        except ValueError:
            search_factor = DEFAULT_SEARCH_FACTOR

        # If reranking is enabled, request more neighbors than needed and filter later
        internal_k = int(k * search_factor) if need_reranking else k

        # Route to appropriate implementation based on client mode
        client = get_client()
        if client is not None and client.mode == "remote":
            return self._search_remote(client, query_vectors, k, internal_k, need_reranking)
        else:
            return self._search_local(query_vectors, k, internal_k, need_reranking)

    def _search_local(
        self, query_vectors: np.ndarray, k: int, internal_k: int, need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using local FAISS index.

        Direct pass-through to the local FAISS index's search method.

        Args:
            query_vectors: Prepared query vectors
            k: Number of results requested
            internal_k: Internal search width (may be larger than k)
            need_reranking: Whether to rerank results

        Returns:
            Tuple of (distances, indices)
        """
        logger.debug(f"Searching local index {self.name} for {len(query_vectors)} queries, k={k}")

        # Use the actual k value - reranking would be done by FAISS internally
        distances, indices = self._local_index.search(query_vectors, k)
        return distances, indices

    def _search_remote(
        self, client: Any, query_vectors: np.ndarray, k: int,
        internal_k: int, need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using remote index with batch processing.

        Handles batching of large query sets to avoid overwhelming the server with a single
        large request. Uses the search_batch_size parameter to determine the optimal batch size.

        Args:
            client: FAISSx client
            query_vectors: Prepared query vectors
            k: Number of results requested by user
            internal_k: Number of results to request from server (may be larger than k)
            need_reranking: Whether to rerank results

        Returns:
            Tuple of (distances, indices) arrays shaped (n, k)

        Raises:
            RuntimeError: If search operation fails
        """
        logger.debug(
            f"Searching remote index {self.index_id} for {len(query_vectors)} queries, k={k}"
        )

        # Get batch size for search operations
        batch_size = self._get_batch_size('search')

        # If queries fit in a single batch, search directly
        if len(query_vectors) <= batch_size:
            return self._search_remote_batch(
                client, query_vectors, k, internal_k, need_reranking
            )

        # Process in batches
        n_queries = len(query_vectors)
        # Initialize output arrays with default values
        all_distances = np.full((n_queries, k), float("inf"), dtype=np.float32)
        all_indices = np.full((n_queries, k), -1, dtype=np.int64)

        # Process each batch and collect results
        for i in range(0, n_queries, batch_size):
            batch = query_vectors[i:min(i + batch_size, n_queries)]
            batch_distances, batch_indices = self._search_remote_batch(
                client, batch, k, internal_k, need_reranking
            )

            # Copy batch results to output arrays
            batch_size_actual = len(batch)
            all_distances[i:i+batch_size_actual] = batch_distances
            all_indices[i:i+batch_size_actual] = batch_indices

        return all_distances, all_indices

    def _search_remote_batch(
        self,
        client: Any,
        query_vectors: np.ndarray,
        k: int,
        internal_k: int,
        need_reranking: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search a batch of queries using remote index.

        Performs the actual search request to the server for a batch of query vectors,
        then processes the results to match the expected output format.

        Args:
            client: FAISSx client
            query_vectors: Batch of query vectors
            k: Number of results requested by user
            internal_k: Number of results to request from server
            need_reranking: Whether to rerank results

        Returns:
            Tuple of (distances, indices) arrays shaped (batch_size, k)

        Raises:
            RuntimeError: If search operation fails or server returns an error
        """
        try:
            # Request more results if reranking is needed
            actual_k = internal_k if need_reranking else k

            # Perform search
            result = client.search(
                self.index_id,
                query_vectors=query_vectors,
                k=actual_k,
                # Uncomment to pass parameters if needed
                # params={"efSearch": self.hnsw.efSearch}
            )

            # Validate response
            if not isinstance(result, dict):
                raise RuntimeError(f"Invalid response format: {type(result)}")

            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                raise RuntimeError(f"Search failed: {error}")

            # Process search results
            n_queries = len(query_vectors)  # Number of query vectors
            search_results = result.get("results", [])

            if not search_results:
                logger.warning("Empty search results returned from server")

            # Initialize output arrays
            distances = np.full((n_queries, k), float("inf"), dtype=np.float32)
            indices = np.full((n_queries, k), -1, dtype=np.int64)

            # Process results for each query vector
            for i in range(min(n_queries, len(search_results))):
                result_data = search_results[i]
                if not isinstance(result_data, dict):
                    logger.warning(f"Invalid result format for query {i}: {result_data}")
                    continue

                # Extract distances and indices from result
                result_distances = result_data.get("distances", [])
                result_indices = result_data.get("indices", [])

                # Apply reranking if needed (truncate to k results)
                if need_reranking and len(result_distances) > k:
                    # Simple reranking by distance (server should return sorted results)
                    result_distances = result_distances[:k]
                    result_indices = result_indices[:k]

                # Fill in results for this query vector
                for j in range(min(k, len(result_distances))):
                    distances[i, j] = result_distances[j]

                    # Map server index back to local index
                    server_idx = result_indices[j]
                    for local_idx, info in self._vector_mapping.items():
                        if info.get("server_idx") == server_idx:
                            indices[i, j] = local_idx
                            break

            return distances, indices

        except Exception as e:
            logger.error(f"Error during remote search: {e}")
            raise RuntimeError(f"Search failed: {e}")

    def range_search(
        self, x: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for all vectors within the specified radius.

        Performs radius-based search, returning all vectors within a specified distance
        threshold from each query vector. This differs from k-NN search in that the number
        of results can vary depending on the data distribution.

        While k-NN search always returns exactly k results per query (regardless of actual
        similarity), range search returns only vectors that are truly within the specified
        radius threshold. This can result in varying numbers of matches per query, including
        zero matches if no vectors are within the radius.

        Args:
            x: Query vectors, shape (n, d)
            radius: Maximum distance threshold

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    - lims: array of shape (n+1) giving the boundaries of results for each query
                    - distances: array of shape (sum_of_results) containing all distances
                    - indices: array of shape (sum_of_results) containing all indices

                    The lims array is structured so that results for query i are located at
                    positions lims[i]:lims[i+1] in both the distances and indices arrays.

        Raises:
            ValueError: If query vector shape doesn't match index dimension
            RuntimeError: If range search fails or isn't supported by the index type
        """
        # Register access for memory management
        self.register_access()

        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Prepare vectors
        query_vectors = self._prepare_vectors(x)

        # Route to appropriate implementation based on client mode
        client = get_client()
        if client is not None and client.mode == "remote":
            return self._range_search_remote(client, query_vectors, radius)
        else:
            return self._range_search_local(query_vectors, radius)

    def _range_search_local(
        self, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Range search using local FAISS index.

        Performs a radius-based search using the local FAISS HNSW index. This returns all vectors
        within the specified radius of each query vector. The HNSW index can perform efficient
        approximate range searches though it may not find all exact matches due to its
        graph-based approximation nature.

        Args:
            query_vectors: Prepared query vectors, shape (n, d), normalized to float32
            radius: Maximum distance threshold for inclusion in results

        Returns:
            Tuple of (lims, distances, indices):
                - lims: array marking result boundaries for each query
                - distances: flat array of distances for all results
                - indices: flat array of indices for all results

        Raises:
            RuntimeError: If the local index doesn't support range_search
        """
        logger.debug(f"Range searching local index {self.name} with radius={radius}")

        if hasattr(self._local_index, "range_search"):
            return self._local_index.range_search(query_vectors, radius)
        else:
            raise RuntimeError("Local FAISS index does not support range_search")

    def _range_search_remote(
        self,
        client: Any,
        query_vectors: np.ndarray,
        radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Range search using remote index with batch processing.

        Handles batching of large query sets to avoid overwhelming the server with a single
        large request. Uses the search_batch_size parameter to determine the optimal batch size.

        Args:
            client: FAISSx client
            query_vectors: Prepared query vectors
            radius: Distance threshold

        Returns:
            Tuple of (lims, distances, indices)

        Raises:
            RuntimeError: If range search operation fails
        """
        logger.debug(f"Range searching remote index {self.index_id} with radius={radius}")

        # Get batch size for range search operations
        batch_size = self._get_batch_size('search')

        # Perform search using batch approach if necessary
        if len(query_vectors) <= batch_size:
            return self._range_search_remote_batch(client, query_vectors, radius)

        # Process in batches when we have many queries
        n_queries = len(query_vectors)
        all_results = []

        # Collect results from each batch
        for i in range(0, n_queries, batch_size):
            batch = query_vectors[i:min(i + batch_size, n_queries)]
            try:
                batch_result = client.range_search(self.index_id, batch, radius)

                if not batch_result.get("success", False):
                    error = batch_result.get("error", "Unknown error")
                    raise RuntimeError(f"Range search failed: {error}")

                all_results.extend(batch_result.get("results", []))
            except Exception as e:
                logger.error(f"Range search batch {i//batch_size} failed: {e}")
                raise

        # Calculate total number of results across all queries
        total_results = sum(res.get("count", 0) for res in all_results)

        # Initialize arrays
        lims = np.zeros(n_queries + 1, dtype=np.int64)
        distances = np.zeros(total_results, dtype=np.float32)
        indices = np.zeros(total_results, dtype=np.int64)

        # Fill arrays with results
        offset = 0
        for i, res in enumerate(all_results):
            # Set limit boundary for this query
            lims[i] = offset

            # Get results for this query
            result_distances = res.get("distances", [])
            result_indices = res.get("indices", [])
            count = len(result_distances)

            # Copy data to output arrays
            if count > 0:
                distances[offset:offset + count] = np.array(result_distances, dtype=np.float32)

                # Map server indices back to local indices
                mapped_indices = self._map_server_to_local_indices(result_indices)
                indices[offset:offset + count] = mapped_indices
                offset += count

        # Set final boundary
        lims[n_queries] = offset

        return lims, distances, indices

    def _range_search_remote_batch(
        self, client: Any, query_vectors: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Range search a batch of queries using remote index.

        Performs a single batch range search against the remote server. This function handles
        the actual communication with the server and processes the results into the
        FAISS-compatible format of (lims, distances, indices).

        The server returns results as a list of dictionaries, which needs to be transformed
        into the compact array representation used by FAISS. This three-array format is more
        efficient than a list of variable-length results, especially for large result sets.

        Args:
            client: FAISSx client instance for server communication
            query_vectors: Batch of query vectors to search
            radius: Distance threshold for inclusion in results

        Returns:
            Tuple of (lims, distances, indices) in FAISS range_search format:
                    - lims: Array of length n+1 with result boundaries
                    - distances: Flat array of all distances
                    - indices: Flat array of corresponding indices

        Raises:
            RuntimeError: If range search operation fails or server returns an error
        """
        try:
            # Set up search parameters
            # Uncomment and use if needed for specific HNSW parameters
            # params = {
            #     "efSearch": self.hnsw.efSearch,
            # }

            # Perform range search
            result = client.range_search(self.index_id, query_vectors, radius)

            # Check for success status in response
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                raise RuntimeError(f"Range search failed: {error}")

            # Process results
            search_results = result.get("results", [])
            n_queries = len(search_results)

            # Calculate total number of results - we need this to allocate properly sized arrays
            total_results = sum(
                res.get("count", 0) for res in search_results
            )

            # Initialize arrays - FAISS range_search returns 3 arrays:
            # 1. lims - boundaries of results for each query (n+1 length)
            # 2. distances - flat array of all distances
            # 3. indices - flat array of corresponding indices
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
                    distances[offset:offset + count] = np.array(result_distances, dtype=np.float32)

                    # Map server indices back to local indices
                    mapped_indices = self._map_server_to_local_indices(result_indices)
                    indices[offset:offset + count] = mapped_indices
                    offset += count

            # Set final boundary (n+1 element of lims)
            # This is crucial for consistent array indexing when processing results
            lims[n_queries] = offset

            return lims, distances, indices

        except Exception as e:
            logger.error(f"Error during remote range search: {e}")
            raise RuntimeError(f"Range search failed: {e}")

    def reset(self) -> None:
        """
        Reset the index to its initial state.

        Clears all vectors from the index while maintaining the same configuration.
        This is useful when you want to reuse an index for a different set of vectors
        without creating a new index object.

        The reset operation differs between local and remote mode:

        - In local mode: Uses the built-in FAISS reset method if available, otherwise
          recreates the index with the same parameters

        - In remote mode: Creates a new index on the server with a new unique ID, since
          remote servers may not support direct reset operations

        In both cases, the index's dimensional and metric configuration are preserved,
        only the stored vectors are removed. For HNSW indices, the graph structure is
        completely rebuilt when new vectors are added after reset.
        """
        # Register access for memory management
        self.register_access()

        client = get_client()

        if client is not None and client.mode == "remote":
            self._reset_remote(client)
        else:
            self._reset_local()

    def _reset_local(self) -> None:
        """
        Reset local index.

        This method handles resetting a local FAISS index, with two possible approaches:

        1. Use the built-in reset() method if available - this is more efficient as it
           avoids reallocating memory for the index structure

        2. If reset() is not available, recreate the index with the same parameters

        For HNSW indices, this means rebuilding the entire graph structure from scratch.
        After reset, the index will have zero vectors but maintain all its parameters
        including dimension, metric type, and HNSW-specific configurations.

        Raises:
            RuntimeError: If index reset fails and recreation also fails
        """
        logger.debug(f"Resetting local index {self.name}")

        if hasattr(self._local_index, "reset"):
            # Use built-in reset method if available
            # This is more efficient as it avoids reallocating memory
            self._local_index.reset()
            self.ntotal = 0
        else:
            # Recreate the index if reset is not supported
            try:
                import faiss
                # Determine metric type from string representation
                metric = faiss.METRIC_INNER_PRODUCT if self.metric_type == "IP" else faiss.METRIC_L2

                # Create a new index with the same parameters
                # This will re-initialize GPU resources if we were using them
                self._create_local_index(self.d, self.M, metric)
                self.ntotal = 0

                logger.debug(f"Recreated local index {self.name} with same parameters")
            except Exception as e:
                logger.error(f"Error resetting local index: {e}")
                raise RuntimeError(f"Failed to reset index: {e}")

    def _reset_remote(self, client: Any) -> None:
        """
        Reset remote index by creating a new one.

        For remote indices, we cannot directly reset an existing index. Instead, we create
        a new index with the same parameters but a different unique ID. This approach is necessary
        because:

        1. The server may not expose a reset operation
        2. Server-side reset could affect other clients using the same index
        3. Creating a new index is more reliable for ensuring a clean state

        This method implements a two-attempt strategy for reliability:

        1. First attempt: Create a new index with a name derived from the original
        2. Second attempt: If first fails, try with a completely random UUID-based name

        After successfully creating the new index, all local references are updated to use
        the new index, and local state (vector mappings, counters) is reset.

        Args:
            client: FAISSx client for server communication

        Raises:
            RuntimeError: If all attempts to create a new index fail
        """
        logger.debug(f"Resetting remote index {self.index_id}")

        try:
            # First attempt: Create new index with modified name
            # We keep part of the original name for easier tracking
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            index_type = self._get_index_type_string()

            # Request new index creation on server
            logger.debug(f"Creating new index {new_name} with type {index_type}")
            response = client.create_index(
                name=new_name, dimension=self.d, index_type=index_type
            )

            # Handle different response formats
            if isinstance(response, dict):
                self.index_id = response.get("index_id", new_name)
            else:
                # For string responses, use the name directly
                logger.debug(f"Got string response: {response}")
                self.index_id = new_name

            self.name = new_name
            logger.debug(f"Successfully created new index: {self.index_id}")

        except Exception as e:
            logger.warning(
                f"Failed to create new index during reset: {e}. Trying alternative method."
            )

            # Second attempt: Try a different approach with totally unique name
            try:
                # Generate a completely unique name that doesn't depend on the original
                unique_name = f"index-hnsw-{uuid.uuid4().hex[:12]}"
                index_type = self._get_index_type_string()

                logger.debug(f"Attempting to create index with unique name: {unique_name}")
                response = client.create_index(
                    name=unique_name, dimension=self.d, index_type=index_type
                )

                if isinstance(response, dict):
                    self.index_id = response.get("index_id", unique_name)
                else:
                    # For string responses, use the name directly
                    self.index_id = unique_name

                self.name = unique_name
                logger.debug(f"Successfully created alternative index: {self.index_id}")

            except Exception as e2:
                logger.error(f"Failed all reset attempts: {e2}")
                raise RuntimeError(f"Failed to reset index: {e2}")

        # Reset all local state
        # This ensures we start fresh with the new index
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def close(self) -> None:
        """
        Clean up resources.

        Releases GPU resources if used. The local index itself will be cleaned up by
        the Python garbage collector when the object is deleted.

        This method should be called when the index is no longer needed to
        ensure proper resource cleanup, especially if using GPU acceleration.
        You can also use the index as a context manager with 'with' statement
        for automatic cleanup.
        """
        logger.debug(f"Cleaning up resources for index {self.name}")

        # Clean up GPU resources if used
        if self._use_gpu and self._gpu_resources is not None:
            self._gpu_resources = None
            logger.debug("Released GPU resources")

        # Local index will be cleaned up by garbage collector
        # But we can help by explicitly removing the reference
        if self._local_index is not None:
            self._local_index = None
            logger.debug("Released local index reference")

    def __del__(self) -> None:
        """
        Clean up when the object is deleted.

        Ensures resources are properly released when the index is garbage collected.
        This is a backup to the explicit close() method which should be called
        when done with the index.
        """
        try:
            self.close()
        except Exception as e:
            # Avoid exceptions during garbage collection
            # as they can be swallowed and cause confusion
            logger.warning(f"Error during cleanup in __del__: {e}")
            pass

    def __enter__(self) -> 'IndexHNSWFlat':
        """
        Context manager entry.

        Allows using the index with a 'with' statement for automatic cleanup.

        Returns:
            Self reference for use in the with block
        """
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception],
                 exc_tb: Optional[Any]) -> None:
        """
        Context manager exit.

        Automatically cleans up resources when exiting a 'with' block.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.close()

    def _get_batch_size(self, operation_type: str = 'default') -> int:
        """
        Get batch size for different operations with fallback defaults.

        This helper method retrieves appropriate batch sizes for different operations,
        using user-configured values when available and falling back to defaults otherwise.
        Batching is important for efficient processing of large vector sets, especially
        in remote mode where it helps manage network traffic and server load.

        Args:
            operation_type: Type of operation ('default', 'search', etc.)

        Returns:
            Batch size value appropriate for the specified operation
        """
        try:
            # For search operations, use the dedicated search batch size parameter
            if operation_type == 'search':
                return int(self.get_parameter('search_batch_size') or DEFAULT_SEARCH_BATCH_SIZE)
            else:
                # For other operations (like add), use the general batch size parameter
                return int(self.get_parameter('batch_size') or DEFAULT_BATCH_SIZE)
        except ValueError:
            # If parameters are invalid or missing, use appropriate defaults
            return DEFAULT_BATCH_SIZE if operation_type != 'search' else DEFAULT_SEARCH_BATCH_SIZE

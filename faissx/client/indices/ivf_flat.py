"""
FAISSx IndexIVFFlat implementation.

This module provides a client-side implementation of the FAISS IndexIVFFlat class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).
"""

from .base import uuid, np, Tuple, faiss, logging, get_client

# Import needed for type hints
from .flat import IndexFlatL2


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
            logging.warning(f"Error initializing remote mode: {e}, falling back to local mode")

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
                    if isinstance(quantizer, IndexFlatL2) and quantizer._use_gpu:
                        # If the quantizer is already on GPU, get the CPU version
                        cpu_quantizer = faiss.index_gpu_to_cpu(quantizer._local_index)
                    else:
                        # Otherwise, use the provided quantizer directly
                        cpu_quantizer = (
                            quantizer._local_index
                            if hasattr(quantizer, '_local_index')
                            else quantizer
                        )

                    cpu_index = faiss.IndexIVFFlat(cpu_quantizer, d, nlist, metric_type)

                    # Convert to GPU index
                    self._local_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)

                    logging.info(f"Using GPU-accelerated IVF index for {self.name}")
                else:
                    # No GPUs available, use CPU version
                    self._local_index = faiss.IndexIVFFlat(
                        quantizer._local_index, d, nlist, metric_type
                    )
            except (ImportError, AttributeError):
                # GPU support not available in this FAISS build
                self._local_index = faiss.IndexIVFFlat(
                    quantizer._local_index, d, nlist, metric_type
                )

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

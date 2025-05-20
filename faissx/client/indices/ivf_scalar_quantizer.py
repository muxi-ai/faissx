#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISSx IndexIVFScalarQuantizer implementation.

This module provides a client-side implementation of the FAISS IndexIVFScalarQuantizer class.
It can operate in either local mode (using FAISS directly) or remote mode
(using the FAISSx server).
"""

from .base import uuid, np, Tuple, faiss, logging, get_client

# Import needed for type hints
from .flat import IndexFlatL2


class IndexIVFScalarQuantizer:
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
        _local_index: Local FAISS index (local mode only)
        _using_remote (bool): Whether we're using remote or local implementation
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
        # Import the actual faiss module
        import faiss as faiss_local

        # Set default metric if not provided
        if metric_type is None:
            metric_type = faiss_local.METRIC_L2

        # Set default qtype if not specified
        if qtype is None:
            qtype = faiss_local.ScalarQuantizer.QT_8bit

        # Store core parameters
        self.d = d
        self.nlist = nlist
        self.qtype = qtype
        # Convert metric type to string representation for remote mode
        self.metric_type = (
            "IP" if metric_type == faiss_local.METRIC_INNER_PRODUCT else "L2"
        )

        # Initialize state variables
        self.is_trained = False
        self.ntotal = 0
        self._nprobe = 1  # Default number of probes

        # Default to local mode
        self._using_remote = False

        # Generate unique name for the index
        self.name = f"index-ivf-sq-{uuid.uuid4().hex[:8]}"

        # Check if client exists (remote mode)
        client = get_client()
        if client is not None:
            try:
                # Remote mode is active
                self._using_remote = True
                self.client = client

                # Determine index type identifier for remote server
                qtype_str = "SQ8" if qtype is None else f"SQ{qtype}"
                index_type = f"IVF{nlist}_{qtype_str}"
                if self.metric_type == "IP":
                    index_type = f"{index_type}_IP"

                # Create index on server
                try:
                    response = self.client.create_index(
                        name=self.name, dimension=self.d, index_type=index_type
                    )

                    self.index_id = response.get("index_id", self.name)
                    self.is_trained = response.get("is_trained", False)
                except Exception as e:
                    # Raise a clear error instead of falling back to local mode
                    raise RuntimeError(
                        f"Failed to create remote IVF SQ index: {e}. "
                        f"Server may not support IVF SQ indices with type {index_type}."
                    )

                return
            except RuntimeError:
                # Re-raise runtime errors without fallback
                raise
            except Exception as e:
                # Any other exception should result in local mode
                logging.warning(f"Using local mode for IVF SQ index due to error: {e}")
                self._using_remote = False

        # If we get here, we're in local mode
        self._local_index = None

        # Import local FAISS here to avoid module-level dependency
        try:
            # Get local index from wrapped quantizer if available
            base_quantizer = quantizer._local_index if hasattr(quantizer, "_local_index") else quantizer

            # Create the local index
            self._local_index = faiss_local.IndexIVFScalarQuantizer(
                base_quantizer, d, nlist, qtype, metric_type
            )

            self.index_id = self.name  # Use name as ID for consistency
        except Exception as e:
            raise RuntimeError(f"Failed to create IndexIVFScalarQuantizer: {e}")

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
        if nprobe < 1:
            raise ValueError(f"nprobe must be at least 1, got {nprobe}")
        if nprobe > self.nlist:
            raise ValueError(
                f"nprobe must not exceed nlist ({self.nlist}), got {nprobe}"
            )

        self._nprobe = nprobe

        # If using local implementation, update the index directly
        if not self._using_remote and self._local_index is not None:
            self._local_index.nprobe = nprobe

    def train(self, x: np.ndarray) -> None:
        """
        Train the index with the provided vectors.

        Args:
            x (np.ndarray): Training vectors, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or already trained
            RuntimeError: If remote training operation fails
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        # Convert to float32 if needed (FAISS requirement)
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Use local FAISS implementation directly
            self._local_index.train(vectors)
            self.is_trained = self._local_index.is_trained
            return

        # Train the remote index
        try:
            result = self.client.train_index(self.index_id, vectors)

            # Check for explicit error response
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                raise RuntimeError(f"Remote training failed: {error_msg}")

            # Update local state based on training result
            self.is_trained = result.get("is_trained", True)
        except Exception as e:
            # Ensure all errors are properly propagated, never fall back to local mode
            raise RuntimeError(f"Remote training operation failed: {e}")

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)

        Raises:
            ValueError: If vector shape doesn't match index dimension or index not trained
            RuntimeError: If remote add operation fails
        """
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

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
        try:
            result = self.client.add_vectors(self.index_id, vectors)

            # Check for explicit error response
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                raise RuntimeError(f"Remote add operation failed: {error_msg}")

            # Update total count
            added_count = result.get("count", 0)
            self.ntotal += added_count
        except Exception as e:
            # Ensure all errors are properly propagated, never fall back to local mode
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
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"Invalid vector shape: expected (n, {self.d}), got {x.shape}"
            )

        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")

        # Convert query vectors to float32
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        if not self._using_remote:
            # Set nprobe for local index before searching
            self._local_index.nprobe = self._nprobe
            # Use local FAISS implementation directly
            return self._local_index.search(query_vectors, k)

        # Perform search on remote index (remote mode)
        # Include nprobe parameter in the search request
        try:
            result = self.client.search(
                self.index_id,
                query_vectors=query_vectors,
                k=k,
                params={"nprobe": self._nprobe}  # Send nprobe parameter to server
            )

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                raise RuntimeError(f"Remote search failed: {error_msg}")

            # Extract results
            n = x.shape[0]  # Number of query vectors
            search_results = result.get("results", [])

            # Initialize output arrays with default values
            distances = np.full((n, k), float("inf"), dtype=np.float32)
            indices = np.full((n, k), -1, dtype=np.int64)

            # Process results for each query vector
            for i in range(min(n, len(search_results))):
                result_data = search_results[i]
                result_distances = result_data.get("distances", [])
                result_indices = result_data.get("indices", [])

                # Fill in results for this query vector
                for j in range(min(k, len(result_distances))):
                    distances[i, j] = result_distances[j]
                    indices[i, j] = result_indices[j]

            return distances, indices
        except Exception as e:
            # Ensure all errors are properly propagated, never fall back to local mode
            raise RuntimeError(f"Remote search operation failed: {e}")

    def reset(self) -> None:
        """
        Reset the index to its initial state, removing all vectors but keeping training.

        This method removes all vectors from the index but preserves the training state.
        After calling reset(), you don't need to retrain the index.
        """
        # Remember if the index was trained before reset
        was_trained = self.is_trained

        if not self._using_remote:
            # Reset local FAISS index
            self._local_index.reset()
            self.ntotal = 0
            # Restore the trained state
            self.is_trained = was_trained
            return

        # Remote mode reset
        try:
            # Create new index with modified name
            new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"

            # Determine index type identifier
            qtype_str = "SQ8" if self.qtype is None else f"SQ{self.qtype}"
            index_type = f"IVF{self.nlist}_{qtype_str}"
            if self.metric_type == "IP":
                index_type = f"{index_type}_IP"

            response = self.client.create_index(
                name=new_name, dimension=self.d, index_type=index_type
            )

            if not response.get("success", False):
                error_msg = response.get("error", "Unknown error")
                raise RuntimeError(f"Failed to create new index during reset: {error_msg}")

            self.index_id = response.get("index_id", new_name)
            self.name = new_name
            # Don't reset training state
            self.is_trained = was_trained

            # Reset all local state
            self.ntotal = 0
        except Exception as e:
            # Ensure all errors are properly propagated, never fall back to local mode
            raise RuntimeError(f"Remote reset operation failed: {e}")

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        # Local index will be cleaned up by garbage collector
        pass

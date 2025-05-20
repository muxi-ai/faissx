"""
FAISSx IndexIDMap implementation.

This module provides a client-side implementation of the FAISS IndexIDMap class.
It serves as a wrapper around another index type that maps external IDs to vectors.
"""

from .base import np, Tuple


class IndexIDMap:
    """
    Proxy implementation of FAISS IndexIDMap.

    This class wraps another index and adds support for mapping between user-provided IDs
    and the internal indices used by the wrapped index. This enables operations using
    custom identifiers instead of sequential indices.

    When adding vectors, the caller can provide explicit IDs. When searching, the results
    will contain these user-provided IDs instead of internal indices.

    Attributes:
        index: The wrapped index object
        is_trained (bool): Whether the underlying index is trained
        ntotal (int): Total number of vectors in the index
        d (int): Vector dimension (same as wrapped index)
        _id_map (dict): Maps internal indices to user-provided IDs
        _rev_id_map (dict): Maps user-provided IDs to internal indices
    """

    def __init__(self, index):
        """
        Initialize the IndexIDMap with the given underlying index.

        Args:
            index: The FAISS index to wrap (e.g., IndexFlatL2, IndexIVFFlat)
        """
        self.index = index
        self.is_trained = index.is_trained
        self.ntotal = 0
        self.d = index.d
        self._id_map = {}  # Maps internal indices to user IDs
        self._rev_id_map = {}  # Maps user IDs to internal indices

    def add_with_ids(self, x: np.ndarray, ids: np.ndarray) -> None:
        """
        Add vectors with explicit IDs.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)
            ids (np.ndarray): IDs to associate with vectors, shape (n,)

        Raises:
            ValueError: If shapes don't match or if duplicate IDs are provided
        """
        if len(x) != len(ids):
            raise ValueError(
                f"Number of vectors ({len(x)}) does not match number of IDs ({len(ids)})"
            )

        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Check for duplicate IDs
        new_ids = set(ids)
        existing_ids = set(self._rev_id_map.keys())
        duplicates = new_ids.intersection(existing_ids)
        if duplicates:
            raise ValueError(f"Duplicate IDs provided: {duplicates}")

        # Add vectors to the underlying index
        self.index.add(x)

        # Keep track of ID mappings
        for i, id_val in enumerate(ids):
            internal_idx = self.ntotal + i
            self._id_map[internal_idx] = id_val
            self._rev_id_map[id_val] = internal_idx

        # Update total count
        self.ntotal += len(x)

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors with automatically generated IDs.

        This will use the vector's position in the index as its ID.

        Args:
            x (np.ndarray): Vectors to add, shape (n, d)
        """
        # Generate sequential IDs starting from the current ntotal
        n = x.shape[0]
        ids = np.arange(self.ntotal, self.ntotal + n, dtype=np.int64)

        # Add with these IDs
        self.add_with_ids(x, ids)

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.

        Args:
            x (np.ndarray): Query vectors, shape (n, d)
            k (int): Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Distances array of shape (n, k)
                - IDs array of shape (n, k) containing user-provided IDs

        Raises:
            ValueError: If query vector shape doesn't match index dimension
        """
        # Call the underlying index's search method
        distances, indices = self.index.search(x, k)

        # Convert internal indices to user IDs
        n = x.shape[0]
        ids = np.full((n, k), -1, dtype=np.int64)  # Default to -1 for not found

        for i in range(n):
            for j in range(k):
                internal_idx = indices[i, j]
                if internal_idx != -1 and internal_idx in self._id_map:
                    ids[i, j] = self._id_map[internal_idx]

        return distances, ids

    def remove_ids(self, ids: np.ndarray) -> None:
        """
        Remove vectors with the specified IDs.

        Note: This is a simulated removal as the underlying FAISS indices may not
        support direct removal. Searches will still respect the removals.

        Args:
            ids (np.ndarray): IDs of vectors to remove

        Raises:
            ValueError: If any ID is not found
        """
        # Check if all IDs exist
        missing = []
        for id_val in ids:
            if id_val not in self._rev_id_map:
                missing.append(id_val)

        if missing:
            raise ValueError(f"IDs not found: {missing}")

        # For each ID to remove
        for id_val in ids:
            # Get the internal index for this ID
            internal_idx = self._rev_id_map[id_val]

            # Remove from both mappings
            del self._id_map[internal_idx]
            del self._rev_id_map[id_val]

        # Update total count
        self.ntotal -= len(ids)

        # Note: The vectors are still in the underlying index, but now
        # they'll be inaccessible through our ID mapping

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
                - ids: array of shape (sum_of_results) containing user-provided IDs

        Raises:
            ValueError: If query vector shape doesn't match index dimension
            RuntimeError: If range search isn't supported by the underlying index
        """
        # Call the underlying index's range_search method
        lims, distances, indices = self.index.range_search(x, radius)

        # Convert internal indices to user IDs
        ids = np.full_like(indices, -1, dtype=np.int64)  # Default to -1 for not found

        for i in range(len(indices)):
            internal_idx = indices[i]
            if internal_idx != -1 and internal_idx in self._id_map:
                ids[i] = self._id_map[internal_idx]

        return lims, distances, ids

    def reconstruct(self, id_val: int) -> np.ndarray:
        """
        Reconstruct a vector from its ID.

        Args:
            id_val: The ID of the vector to reconstruct

        Returns:
            The reconstructed vector

        Raises:
            ValueError: If the ID is not found
            RuntimeError: If the underlying index doesn't support reconstruction
        """
        if id_val not in self._rev_id_map:
            raise ValueError(f"ID {id_val} not found")

        # Get the internal index
        internal_idx = self._rev_id_map[id_val]

        # Check if the underlying index supports reconstruction
        if not hasattr(self.index, 'reconstruct'):
            raise RuntimeError("Underlying index doesn't support vector reconstruction")

        # Call the underlying index's reconstruct method
        return self.index.reconstruct(internal_idx)

    def reconstruct_n(self, ids: np.ndarray) -> np.ndarray:
        """
        Reconstruct multiple vectors from their IDs.

        Args:
            ids: Array of IDs to reconstruct

        Returns:
            Array of reconstructed vectors

        Raises:
            ValueError: If any ID is not found
            RuntimeError: If the underlying index doesn't support reconstruction
        """
        # Check if all IDs exist
        missing = []
        for id_val in ids:
            if id_val not in self._rev_id_map:
                missing.append(id_val)

        if missing:
            raise ValueError(f"IDs not found: {missing}")

        # Check if the underlying index supports reconstruction
        if not hasattr(self.index, 'reconstruct_n'):
            # Fall back to individual reconstructions if reconstruct is available
            if hasattr(self.index, 'reconstruct'):
                vectors = np.zeros((len(ids), self.d), dtype=np.float32)
                for i, id_val in enumerate(ids):
                    vectors[i] = self.reconstruct(id_val)
                return vectors
            else:
                raise RuntimeError("Underlying index doesn't support vector reconstruction")

        # Map user IDs to internal indices
        internal_indices = np.array([self._rev_id_map[id_val] for id_val in ids], dtype=np.int64)

        # Call the underlying index's reconstruct_n method
        return self.index.reconstruct_n(internal_indices)

    def train(self, x: np.ndarray) -> None:
        """
        Train the underlying index if it requires training.

        Args:
            x (np.ndarray): Training vectors, shape (n, d)
        """
        self.index.train(x)
        self.is_trained = self.index.is_trained

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        self.index.reset()
        self.ntotal = 0
        self._id_map = {}
        self._rev_id_map = {}
        self.is_trained = self.index.is_trained


class IndexIDMap2(IndexIDMap):
    """
    Proxy implementation of FAISS IndexIDMap2.

    IndexIDMap2 is an extension of IndexIDMap that allows replacing
    vector content while keeping the same IDs. This is useful when
    vectors need to be updated or when the indexed vectors are based on
    objects (like images or documents) that change over time.

    This class inherits most functionality from IndexIDMap but adds
    methods to replace vectors without changing their IDs.
    """

    def replace_vector(self, id_val, vector: np.ndarray) -> None:
        """
        Replace a vector with a new one, preserving its ID.

        This creates a new index containing all vectors except the one to replace,
        then rebuilds the index with the updated vector. This is inefficient for
        frequent updates, but necessary since most FAISS indices don't support
        direct vector replacement.

        Args:
            id_val: ID of the vector to replace
            vector: New vector data, shape (d,)

        Raises:
            ValueError: If the ID is not found or the vector has wrong dimensionality
        """
        if id_val not in self._rev_id_map:
            raise ValueError(f"ID {id_val} not found")

        # Ensure the vector has the right dimension
        if isinstance(vector, np.ndarray):
            if len(vector.shape) == 1:
                # Convert 1D to 2D
                vector = vector.reshape(1, -1)

            if vector.shape[1] != self.d:
                msg = f"Vector has wrong dimension: {vector.shape[1]}, expected {self.d}"
                raise ValueError(msg)
        else:
            raise ValueError("Vector must be a numpy array")

        # For simplicity, rebuild the index:
        # 1. Get all vectors and IDs
        # 2. Update the vector with the given ID
        # 3. Reset the index and re-add all vectors

        # Get all vectors through reconstruction
        all_ids = np.array(list(self._rev_id_map.keys()))
        if hasattr(self.index, 'reconstruct_n'):
            all_vectors = []
            # Process in smaller batches to avoid memory issues with large indices
            batch_size = 1000
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i+batch_size]
                batch_internal_indices = np.array([
                    self._rev_id_map[id_val] for id_val in batch_ids
                ])
                batch_vectors = self.index.reconstruct_n(batch_internal_indices)
                all_vectors.append(batch_vectors)
            all_vectors = np.vstack(all_vectors)
        else:
            # Fall back to single reconstruction if reconstruct_n not available
            all_vectors = np.zeros((len(all_ids), self.d), dtype=np.float32)
            for i, id_val in enumerate(all_ids):
                all_vectors[i] = self.reconstruct(id_val)

        # Replace the vector for the specified ID
        idx_to_replace = np.where(all_ids == id_val)[0][0]
        all_vectors[idx_to_replace] = vector

        # Reset and rebuild the index
        old_index = self.index
        self.reset()
        self.index = old_index
        self.index.reset()

        # Re-add all vectors with their IDs
        self.add_with_ids(all_vectors, all_ids)

    def update_vectors(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """
        Update multiple vectors at once, preserving their IDs.

        Args:
            ids: Array of IDs to update
            vectors: New vector data, shape (n, d)

        Raises:
            ValueError: If any ID is not found or vectors have wrong shape
        """
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of IDs ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
            )

        # Check if all IDs exist
        missing = []
        for id_val in ids:
            if id_val not in self._rev_id_map:
                missing.append(id_val)

        if missing:
            raise ValueError(f"IDs not found: {missing}")

        # Rebuild the index
        all_known_ids = np.array(list(self._rev_id_map.keys()))

        # First, collect all vectors to keep (excluding those to be updated)
        keep_mask = np.ones(len(all_known_ids), dtype=bool)

        # Find indices of IDs to update
        for id_val in ids:
            idx = np.where(all_known_ids == id_val)[0][0]
            keep_mask[idx] = False

        # Get IDs and vectors to keep
        keep_ids = all_known_ids[keep_mask]

        # Get all vectors through reconstruction
        if len(keep_ids) > 0:
            if hasattr(self.index, 'reconstruct_n'):
                keep_internal_indices = np.array([
                    self._rev_id_map[id_val] for id_val in keep_ids
                ])
                keep_vectors = self.index.reconstruct_n(keep_internal_indices)
            else:
                # Fall back to single reconstruction
                keep_vectors = np.zeros((len(keep_ids), self.d), dtype=np.float32)
                for i, id_val in enumerate(keep_ids):
                    keep_vectors[i] = self.reconstruct(id_val)

            # Combine kept vectors with new vectors
            combined_vectors = np.vstack([keep_vectors, vectors])
            combined_ids = np.concatenate([keep_ids, ids])
        else:
            # Only update vectors
            combined_vectors = vectors
            combined_ids = ids

        # Reset and rebuild the index
        old_index = self.index
        self.reset()
        self.index = old_index
        self.index.reset()

        # Re-add all vectors
        self.add_with_ids(combined_vectors, combined_ids)

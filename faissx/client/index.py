"""
FAISS index classes implementation for ZeroMQ-based FAISSx
"""

import uuid
import numpy as np
from typing import Dict, Tuple, Any

from .client import get_client


class IndexFlatL2:
    """
    Proxy implementation of FAISS IndexFlatL2

    This class mimics the behavior of FAISS IndexFlatL2 but uses the
    remote FAISSx service for all operations via ZeroMQ.
    """

    def __init__(self, d: int):
        """
        Initialize the index.

        Args:
            d: Vector dimension
        """
        self.d = d
        self.is_trained = True
        self.ntotal = 0

        # Create a name for the index
        self.name = f"index-flat-l2-{uuid.uuid4().hex[:8]}"

        # Create the remote index
        self.client = get_client()
        self.index_id = self.client.create_index(
            name=self.name,
            dimension=self.d,
            index_type="L2"
        )

        # For tracking vectors locally
        self._vector_mapping = {}  # Maps internal indices to server-side information
        self._next_idx = 0

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x: Vectors to add (n, d)
        """
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Ensure we have float32 numpy array
        vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Add vectors to the index
        result = self.client.add_vectors(self.index_id, vectors)

        # Update count and mappings
        if result.get("success", False):
            added_count = result.get("count", 0)
            for i in range(added_count):
                self._vector_mapping[self._next_idx] = {
                    "local_idx": self._next_idx,
                    "server_idx": self.ntotal + i
                }
                self._next_idx += 1

            self.ntotal += added_count

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.

        Args:
            x: Query vectors (n, d)
            k: Number of results to return

        Returns:
            D: Distances (n, k)
            indices: Indices (n, k)
        """
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Ensure we have float32 numpy array for the query
        query_vectors = x.astype(np.float32) if x.dtype != np.float32 else x

        # Search for similar vectors
        result = self.client.search(
            self.index_id,
            query_vectors=query_vectors,
            k=k
        )

        n = x.shape[0]
        search_results = result.get("results", [])

        # Initialize output arrays
        distances = np.full((n, k), float('inf'), dtype=np.float32)
        idx = np.full((n, k), -1, dtype=np.int64)

        # Process results for each query vector
        for i in range(min(n, len(search_results))):
            result_data = search_results[i]
            result_distances = result_data.get("distances", [])
            result_indices = result_data.get("indices", [])

            # Copy results to output arrays
            for j in range(min(k, len(result_distances))):
                distances[i, j] = result_distances[j]

                # Map server-side index to client-side index
                server_idx = result_indices[j]
                # Find the local index corresponding to this server index
                found = False
                for local_idx, info in self._vector_mapping.items():
                    if info["server_idx"] == server_idx:
                        idx[i, j] = local_idx
                        found = True
                        break

                # If not found, keep -1
                if not found:
                    idx[i, j] = -1

        return distances, idx

    def reset(self) -> None:
        """
        Reset the index.
        """
        # Get index stats first to see if it exists
        try:
            stats = self.client.get_index_stats(self.index_id)
            if stats.get("success", False):
                # Create a new index (there's no explicit delete in the server API)
                new_name = f"{self.name}-{uuid.uuid4().hex[:8]}"
                self.index_id = self.client.create_index(
                    name=new_name,
                    dimension=self.d,
                    index_type="L2"
                )
                self.name = new_name
        except Exception:  # Catch specific exceptions if possible
            # If there's an error, just recreate with the same name
            self.index_id = self.client.create_index(
                name=self.name,
                dimension=self.d,
                index_type="L2"
            )

        # Reset local state
        self.ntotal = 0
        self._vector_mapping = {}
        self._next_idx = 0

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        # No explicit delete index operation in the server API
        pass

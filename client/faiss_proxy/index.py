"""
FAISS index classes implementation
"""

import uuid
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from .client import get_client


class IndexFlatL2:
    """
    Proxy implementation of FAISS IndexFlatL2

    This class mimics the behavior of FAISS IndexFlatL2 but uses the
    remote FAISS Proxy service for all operations.
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
            index_type="IndexFlatL2"
        )

        # For tracking vectors locally
        self._next_id = 0
        self._vector_ids = []

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            x: Vectors to add (n, d)
        """
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        # Convert vectors to list format for the API
        vectors = []

        for i in range(x.shape[0]):
            vector_id = f"vec-{self._next_id}"
            self._next_id += 1
            self._vector_ids.append(vector_id)

            vectors.append({
                "id": vector_id,
                "values": x[i].tolist(),
                "metadata": {}
            })

        # Add vectors to the index
        result = self.client.add_vectors(self.index_id, {"vectors": vectors})

        # Update count
        if result.get("success", False):
            self.ntotal += result.get("added_count", 0)

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.

        Args:
            x: Query vectors (n, d)
            k: Number of results to return

        Returns:
            D: Distances (n, k)
            I: Indices (n, k)
        """
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Invalid vector shape: expected (n, {self.d}), got {x.shape}")

        n = x.shape[0]
        D = np.empty((n, k), dtype=np.float32)
        I = np.empty((n, k), dtype=np.int64)

        for i in range(n):
            query_vector = x[i].tolist()

            # Search for similar vectors
            result = self.client.search(
                self.index_id,
                vector=query_vector,
                k=k
            )

            # Convert results to FAISS format
            results = result.get("results", [])

            for j in range(min(k, len(results))):
                # Map score to distance (higher score -> lower distance)
                score = results[j].get("score", 0)
                distance = 1.0 / score - 1.0 if score > 0 else float('inf')

                # Get vector ID and find its index
                vector_id = results[j].get("id")
                try:
                    idx = self._vector_ids.index(vector_id)
                except ValueError:
                    # If the vector ID is not found, use -1
                    idx = -1

                D[i, j] = distance
                I[i, j] = idx

            # Fill remaining slots with -1 and infinity
            for j in range(len(results), k):
                D[i, j] = float('inf')
                I[i, j] = -1

        return D, I

    def reset(self) -> None:
        """
        Reset the index.
        """
        # Delete the current index
        self.client.delete_index(self.index_id)

        # Create a new index
        self.index_id = self.client.create_index(
            name=self.name,
            dimension=self.d,
            index_type="IndexFlatL2"
        )

        # Reset local state
        self.ntotal = 0
        self._next_id = 0
        self._vector_ids = []

    def __del__(self) -> None:
        """
        Clean up resources when the index is deleted.
        """
        try:
            # Delete the remote index when the object is garbage collected
            self.client.delete_index(self.index_id)
        except:
            # Ignore errors during cleanup
            pass

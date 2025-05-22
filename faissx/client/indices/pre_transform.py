# Copyright 2024 Muxi, Inc. All Rights Reserved.
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
IndexPreTransform implementation for FAISSx.

This module provides a wrapper index that applies transformations to vectors
before indexing or searching them, compatible with the FAISS IndexPreTransform.
"""

import uuid
import numpy as np
from typing import Any, List, Tuple, Union

try:
    import faiss
except ImportError:
    faiss = None

from ..client import get_client
from ..transforms import VectorTransform
from .base import logger, FAISSxBaseIndex


class IndexPreTransform(FAISSxBaseIndex):
    """
    Proxy implementation of FAISS IndexPreTransform.

    This class wraps another index and applies vector transformations before
    indexing or searching vectors. Transformations can include PCA dimensionality
    reduction, L2 normalization, or custom transformations.

    The transform pipeline is applied in order, and the reverse pipeline is applied
    in reverse order during vector reconstruction.

    Attributes:
        d_in (int): Input dimension of vectors
        index: The base index being wrapped
        transform_chain (List[VectorTransform]): Chain of transformations to apply
        is_trained (bool): Whether the index and transforms are trained
        ntotal (int): Total number of vectors in the index
        name (str): Unique identifier for the index
        index_id (str): Server-side index identifier
    """

    def __init__(self, index: Any, transform: Union[VectorTransform, List[VectorTransform]]):
        """
        Initialize the IndexPreTransform with a base index and transformations.

        Args:
            index: The base index to wrap (must implement FAISS-compatible interface)
            transform: A single transform or list of transforms to apply before indexing
        """
        super().__init__()

        # Store the base index
        self.index = index

        # Ensure transform is a list
        if isinstance(transform, VectorTransform):
            self.transform_chain = [transform]
        else:
            self.transform_chain = transform

        # Validate transform chain dimensions match
        self._validate_transform_chain()

        # Set input dimension from first transform
        self.d_in = self.transform_chain[0].d_in

        # Set the output dimension to the last transform's output
        self.d_out = self.transform_chain[-1].d_out

        # Check that the output dimension matches the base index dimension
        if hasattr(self.index, "d") and self.d_out != self.index.d:
            raise ValueError(f"Output dimension of transform chain ({self.d_out}) "
                             f"does not match index dimension ({self.index.d})")

        # Initialize state variables
        self.is_trained = all(t.is_trained for t in self.transform_chain) and getattr(index, "is_trained", True)
        self.ntotal = getattr(index, "ntotal", 0)

        # Generate unique name for the index
        self.name = f"index-pre-transform-{uuid.uuid4().hex[:8]}"
        self.index_id = self.name

        # Local storage for original vectors (needed for reconstruct)
        self._original_vectors = None

        # Vector mapping for tracking indices
        self._vector_mapping = {}

        # Client connection for remote mode
        client = get_client()
        if client and client.is_remote_mode():
            logger.info(f"Creating remote IndexPreTransform on server {client.server}")
            self._create_remote_index(client)
        else:
            logger.info(f"Creating local IndexPreTransform {self.name}")
            self._create_local_index()

    def _validate_transform_chain(self) -> None:
        """
        Validate that the transform chain dimensions match up.

        Ensures that the output dimension of each transform matches the input
        dimension of the next transform in the chain.
        """
        if not self.transform_chain:
            raise ValueError("Transform chain cannot be empty")

        for i in range(len(self.transform_chain) - 1):
            if self.transform_chain[i].d_out != self.transform_chain[i + 1].d_in:
                raise ValueError(
                    f"Output dimension of transform {i} ({self.transform_chain[i].d_out}) "
                    f"does not match input dimension of transform {i+1} "
                    f"({self.transform_chain[i+1].d_in})"
                )

    def _create_local_index(self) -> None:
        """
        Create a local FAISS IndexPreTransform if FAISS is available.
        """
        if faiss is not None:
            try:
                # Check if the base index has a _local_index attribute
                base_index = getattr(self.index, "_local_index", self.index)

                # Create FAISS transforms from our transform chain
                faiss_transforms = []
                for transform in self.transform_chain:
                    # We don't create FAISS transforms directly - we'll handle transforms ourselves
                    # This is just to create a compatible local index structure
                    # Depending on the transform type, we would create different FAISS transforms
                    pass

                # For now, we'll just use our transform_chain directly
                # In a full implementation, you would convert to FAISS transform types
                self._local_index = base_index  # Just use base index for now
            except Exception as e:
                logger.warning(f"Error creating local FAISS IndexPreTransform: {e}")
                self._local_index = None
        else:
            logger.warning("FAISS not available, using fallback implementation")
            self._local_index = None

    def _create_remote_index(self, client: Any) -> None:
        """
        Create a remote IndexPreTransform on the server.

        Args:
            client: The FAISSx client connection
        """
        # Currently server-side pre-transform is not implemented
        # We'll use local implementation with remote base index
        logger.warning("Using local IndexPreTransform with remote base index - "
                      "server-side transforms not yet supported")
        self._create_local_index()

    def train(self, x: np.ndarray) -> None:
        """
        Train both the transforms and the base index.

        Args:
            x: Training vectors with shape (n, d_in)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(
                f"Training data has dimension {x.shape[1]}, but index expects {self.d_in}"
            )

        # Train all transforms in the chain
        vectors = x.copy()
        for i, transform in enumerate(self.transform_chain):
            if not transform.is_trained:
                logger.info(f"Training transform {i}")
                transform.train(vectors)

            # Apply this transform to get input for next transform
            vectors = transform.apply(vectors)

        # Train the base index with the transformed vectors
        if hasattr(self.index, "train"):
            logger.info("Training base index")
            self.index.train(vectors)

        # Update trained state
        self.is_trained = all(t.is_trained for t in self.transform_chain) and getattr(self.index, "is_trained", True)
        logger.info(f"IndexPreTransform training complete, is_trained={self.is_trained}")

    def add(self, x: np.ndarray) -> None:
        """
        Add vectors to the index after applying transformations.

        Args:
            x: Vectors to add with shape (n, d_in)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(
                f"Input vectors have dimension {x.shape[1]}, but index expects {self.d_in}"
            )

        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        # Store original vectors for reconstruction
        if self._original_vectors is None:
            self._original_vectors = x.copy()
        else:
            self._original_vectors = np.vstack([self._original_vectors, x])

        # Set up vector mapping
        start_idx = self.ntotal
        for i in range(x.shape[0]):
            self._vector_mapping[start_idx + i] = {
                "original_idx": len(self._original_vectors) - x.shape[0] + i
            }

        # Apply transforms in sequence
        vectors = x.copy()
        for transform in self.transform_chain:
            vectors = transform.apply(vectors)

        # Add transformed vectors to base index
        self.index.add(vectors)

        # Update total count
        self.ntotal = getattr(self.index, "ntotal", self.ntotal + x.shape[0])

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors after applying transformations.

        Args:
            x: Query vectors with shape (n, d_in)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices) arrays
        """
        if x.shape[1] != self.d_in:
            raise ValueError(
                f"Query vectors have dimension {x.shape[1]}, but index expects {self.d_in}"
            )

        # Apply transforms in sequence
        vectors = x.copy()
        for transform in self.transform_chain:
            vectors = transform.apply(vectors)

        # Search in the base index
        distances, indices = self.index.search(vectors, k)

        # Return the results (indices already map to original vectors)
        return distances, indices

    def reconstruct(self, idx: int) -> np.ndarray:
        """
        Reconstruct the vector at the given index.

        If the original vector was cached during add(), returns that directly.
        Otherwise, reconstructs from the base index and applies reverse transforms.

        Args:
            idx: Index of the vector to reconstruct

        Returns:
            Reconstructed vector with shape (d_in,)
        """
        if idx < 0 or idx >= self.ntotal:
            raise IndexError(f"Index {idx} out of bounds for index with {self.ntotal} vectors")

        # Try to get from original vectors cache first
        mapping = self._vector_mapping.get(idx)
        if mapping and "original_idx" in mapping and self._original_vectors is not None:
            original_idx = mapping["original_idx"]
            if original_idx < len(self._original_vectors):
                return self._original_vectors[original_idx]

        # If not cached, reconstruct from base index and apply reverse transforms
        vector = self.index.reconstruct(idx)

        # Apply reverse transforms in reverse order
        for transform in reversed(self.transform_chain):
            # Check if the transform supports reverse transformation
            if hasattr(transform, "reverse_transform"):
                vector = transform.reverse_transform(vector.reshape(1, -1))
                vector = vector.reshape(-1)  # Back to 1D
            else:
                logger.warning(f"Transform {transform.__class__.__name__} does not support "
                              "reverse transformation, result may be incomplete")

        return vector

    def reconstruct_n(self, n0: int, ni: int) -> np.ndarray:
        """
        Reconstruct a range of vectors.

        Args:
            n0: Start index
            ni: Number of vectors to reconstruct

        Returns:
            Reconstructed vectors with shape (ni, d_in)
        """
        if n0 < 0 or n0 + ni > self.ntotal:
            raise IndexError(f"Range {n0}:{n0+ni} out of bounds for index with {self.ntotal} vectors")

        # Try to get from original vectors cache first
        if self._original_vectors is not None:
            cached = []
            missing_indices = []

            for i in range(ni):
                idx = n0 + i
                mapping = self._vector_mapping.get(idx)
                if mapping and "original_idx" in mapping:
                    original_idx = mapping["original_idx"]
                    if original_idx < len(self._original_vectors):
                        cached.append((i, self._original_vectors[original_idx]))
                    else:
                        missing_indices.append(i)
                else:
                    missing_indices.append(i)

            # If all vectors are cached, return them directly
            if len(cached) == ni:
                result = np.zeros((ni, self.d_in), dtype=np.float32)
                for i, vec in cached:
                    result[i] = vec
                return result

        # Otherwise, reconstruct from base index
        if hasattr(self.index, "reconstruct_n"):
            vectors = self.index.reconstruct_n(n0, ni)
        else:
            # Fall back to individual reconstructs if base index doesn't support reconstruct_n
            vectors = np.zeros((ni, self.d_out), dtype=np.float32)
            for i in range(ni):
                vectors[i] = self.index.reconstruct(n0 + i)

        # Apply reverse transforms in reverse order
        for transform in reversed(self.transform_chain):
            if hasattr(transform, "reverse_transform"):
                vectors = transform.reverse_transform(vectors)
            else:
                logger.warning(f"Transform {transform.__class__.__name__} does not support "
                              "reverse transformation, result may be incomplete")

        return vectors

    def reset(self) -> None:
        """
        Reset the index, removing all vectors but preserving training.
        """
        # Reset the base index if it supports it
        if hasattr(self.index, "reset"):
            self.index.reset()

        # Clear our local storage
        self._original_vectors = None
        self._vector_mapping = {}
        self.ntotal = getattr(self.index, "ntotal", 0)

    def get_transform(self, i: int) -> VectorTransform:
        """
        Get a specific transform from the chain.

        Args:
            i: Index of the transform to retrieve

        Returns:
            The requested transform
        """
        if i < 0 or i >= len(self.transform_chain):
            raise IndexError(f"Transform index {i} out of bounds for chain with {len(self.transform_chain)} transforms")

        return self.transform_chain[i]

    def prepend_transform(self, transform: VectorTransform) -> None:
        """
        Add a transform at the beginning of the chain.

        Args:
            transform: Transform to prepend
        """
        # Ensure dimensions match
        if transform.d_out != self.transform_chain[0].d_in:
            raise ValueError(
                f"Output dimension of new transform ({transform.d_out}) does not match "
                f"input dimension of first existing transform ({self.transform_chain[0].d_in})"
            )

        # Update input dimension
        self.d_in = transform.d_in

        # Add transform to the beginning of the chain
        self.transform_chain.insert(0, transform)

        # Update trained state
        self.is_trained = all(t.is_trained for t in self.transform_chain) and getattr(self.index, "is_trained", True)

        # Clear stored vectors since they no longer match the new input dimension
        self._original_vectors = None
        self._vector_mapping = {}

    def append_transform(self, transform: VectorTransform) -> None:
        """
        Add a transform at the end of the chain.

        Note: This is not a standard FAISS operation but can be useful.

        Args:
            transform: Transform to append
        """
        raise NotImplementedError("Adding transforms after index creation is not supported")

    def __str__(self) -> str:
        """
        Get string representation of the index.

        Returns:
            String description of the index
        """
        transforms_str = ", ".join(t.__class__.__name__ for t in self.transform_chain)
        return f"IndexPreTransform(transforms=[{transforms_str}], index={self.index})"

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
Vector transformation utilities for FAISSx.

This module provides implementations of various vector transformations that can be
used with IndexPreTransform for preprocessing vectors before indexing or searching.
"""

import numpy as np
from typing import List, Optional


class VectorTransform:
    """
    Base class for vector transformations.

    All transform classes should inherit from this base class and implement
    the required methods for applying transformations to vectors.
    """

    def __init__(self, d_in: int, d_out: int):
        """
        Initialize the transform with input and output dimensions.

        Args:
            d_in: Input vector dimension
            d_out: Output vector dimension after transformation
        """
        self.d_in = d_in
        self.d_out = d_out
        self.is_trained = False

    def train(self, vectors: np.ndarray) -> None:
        """
        Train the transform using sample vectors.

        Args:
            vectors: Training vectors with shape (n, d_in)
        """
        self.is_trained = True

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply the transform to input vectors.

        Args:
            vectors: Input vectors with shape (n, d_in)

        Returns:
            Transformed vectors with shape (n, d_out)
        """
        raise NotImplementedError("Subclasses must implement this method")

    def reverse_transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply the reverse transform (if possible).

        Args:
            vectors: Transformed vectors with shape (n, d_out)

        Returns:
            Original vectors with shape (n, d_in)
        """
        raise NotImplementedError("This transform does not support reverse transformation")


class L2NormTransform(VectorTransform):
    """
    Transform that normalizes vectors to unit L2 norm.
    """

    def __init__(self, d: int):
        """
        Initialize the L2 normalization transform.

        Args:
            d: Vector dimension (same for input and output)
        """
        super().__init__(d, d)
        self.is_trained = True  # No training needed

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit L2 norm.

        Args:
            vectors: Input vectors with shape (n, d)

        Returns:
            L2-normalized vectors with shape (n, d)
        """
        if vectors.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {vectors.shape[1]} does not match expected {self.d_in}")

        # Calculate L2 norms for each vector
        norms = np.linalg.norm(vectors, axis=1)

        # Handle zero vectors to avoid division by zero
        norms[norms == 0] = 1.0

        # Normalize by dividing by norms (using broadcasting)
        return vectors / norms[:, np.newaxis]

    def reverse_transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        The reverse of L2 normalization isn't well-defined without stored norms.

        Args:
            vectors: Normalized vectors with shape (n, d)

        Returns:
            The input vectors (since we can't restore original magnitudes)
        """
        if vectors.shape[1] != self.d_out:
            raise ValueError(f"Input vectors dimension {vectors.shape[1]} does not match expected {self.d_out}")

        return vectors  # Cannot reverse without original norms


class PCATransform(VectorTransform):
    """
    Transform that applies PCA dimensionality reduction.
    """

    def __init__(self, d_in: int, d_out: int, eigen_power: float = 0.0):
        """
        Initialize the PCA transform.

        Args:
            d_in: Input vector dimension
            d_out: Output vector dimension after PCA
            eigen_power: Power to which eigenvalues are raised when scaling (default: 0.0)
        """
        super().__init__(d_in, d_out)

        if d_out > d_in:
            raise ValueError(f"Output dimension ({d_out}) cannot be greater than input dimension ({d_in})")

        self.mean = None  # Mean of training vectors
        self.components = None  # Principal components
        self.eigen_power = eigen_power
        self.eigenvalues = None  # Eigenvalues of covariance matrix
        self.is_trained = False

    def train(self, vectors: np.ndarray) -> None:
        """
        Train the PCA transform using sample vectors.

        Args:
            vectors: Training vectors with shape (n, d_in)
        """
        if vectors.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {vectors.shape[1]} does not match expected {self.d_in}")

        # Compute the mean vector and center data
        self.mean = np.mean(vectors, axis=0)
        centered_data = vectors - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(centered_data, rowvar=False)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep only top d_out components
        self.eigenvalues = eigenvalues[:self.d_out]
        self.components = eigenvectors[:, :self.d_out]

        # Apply eigenvalue scaling if specified
        if self.eigen_power != 0:
            # Convert eigenvalues to the specified power
            scaling = np.power(self.eigenvalues, self.eigen_power / 2)
            # Apply scaling to components
            self.components = self.components * scaling

        self.is_trained = True

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply PCA transform to input vectors.

        Args:
            vectors: Input vectors with shape (n, d_in)

        Returns:
            PCA-transformed vectors with shape (n, d_out)
        """
        if not self.is_trained:
            raise RuntimeError("PCATransform must be trained before apply")

        if vectors.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {vectors.shape[1]} does not match expected {self.d_in}")

        # Center the data
        centered_data = vectors - self.mean

        # Project onto principal components
        return np.dot(centered_data, self.components)

    def reverse_transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply the reverse PCA transform.

        Note: This is lossy if d_out < d_in as some information is discarded.

        Args:
            vectors: PCA-transformed vectors with shape (n, d_out)

        Returns:
            Reconstructed original vectors with shape (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("PCATransform must be trained before reverse_transform")

        if vectors.shape[1] != self.d_out:
            raise ValueError(f"Input vectors dimension {vectors.shape[1]} does not match expected {self.d_out}")

        # Project back to original space
        reconstructed = np.dot(vectors, self.components.T)

        # Add the mean back
        return reconstructed + self.mean


class RemapDimensionsTransform(VectorTransform):
    """
    Transform that selects or rearranges specific dimensions of input vectors.
    """

    def __init__(self, d_in: int, d_out: int, map_indices: Optional[List[int]] = None):
        """
        Initialize dimension remapping transform.

        Args:
            d_in: Input vector dimension
            d_out: Output vector dimension after remapping
            map_indices: List of indices from input vectors to select (default: first d_out dimensions)
        """
        super().__init__(d_in, d_out)

        # If map_indices not provided, default to first d_out dimensions
        if map_indices is None:
            if d_out > d_in:
                raise ValueError(f"Output dimension ({d_out}) cannot be greater than input dimension ({d_in}) without explicit mapping")
            self.map_indices = list(range(d_out))
        else:
            if len(map_indices) != d_out:
                raise ValueError(f"Length of map_indices ({len(map_indices)}) must match d_out ({d_out})")
            if max(map_indices) >= d_in:
                raise ValueError(f"Map indices must be less than d_in ({d_in})")
            self.map_indices = map_indices

        # Compute inverse mapping if possible
        self._compute_inverse_mapping()

        self.is_trained = True  # No training needed

    def _compute_inverse_mapping(self) -> None:
        """
        Compute inverse mapping for reverse transform if possible.
        """
        # Check if mapping is invertible (all output dimensions map to distinct input dimensions)
        if len(set(self.map_indices)) == len(self.map_indices):
            self.inverse_mapping = np.zeros(self.d_in, dtype=int)
            # Fill with -1 initially
            self.inverse_mapping.fill(-1)

            # Set the mapping
            for output_idx, input_idx in enumerate(self.map_indices):
                self.inverse_mapping[input_idx] = output_idx

            self.is_invertible = True
        else:
            self.inverse_mapping = None
            self.is_invertible = False

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply dimension remapping to input vectors.

        Args:
            vectors: Input vectors with shape (n, d_in)

        Returns:
            Remapped vectors with shape (n, d_out)
        """
        if vectors.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {vectors.shape[1]} does not match expected {self.d_in}")

        # Select dimensions according to mapping
        return vectors[:, self.map_indices]

    def reverse_transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply the reverse dimension remapping if possible.

        Args:
            vectors: Remapped vectors with shape (n, d_out)

        Returns:
            Reconstructed original vectors with shape (n, d_in)
        """
        if not self.is_invertible:
            raise RuntimeError("This transform is not invertible because the mapping is not one-to-one")

        if vectors.shape[1] != self.d_out:
            raise ValueError(f"Input vectors dimension {vectors.shape[1]} does not match expected {self.d_out}")

        # Create output array with original dimensions
        result = np.zeros((vectors.shape[0], self.d_in), dtype=vectors.dtype)

        # Map dimensions back to original space
        for input_idx, output_idx in enumerate(self.map_indices):
            if output_idx >= 0:  # Check if this dimension has a mapping
                result[:, output_idx] = vectors[:, input_idx]

        return result

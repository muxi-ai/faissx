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
Tests for the IndexPreTransform implementation.
"""

import pytest
import numpy as np

from faissx.client.indices.flat import IndexFlatL2
from faissx.client.indices.pre_transform import IndexPreTransform
from faissx.client.transforms import (
    L2NormTransform, PCATransform, RemapDimensionsTransform
)


def test_pre_transform_init():
    """Test IndexPreTransform initialization."""
    # Create a base index
    d_out = 64
    base_index = IndexFlatL2(d_out)

    # Create a transform - L2 normalization (same dimensions)
    transform = L2NormTransform(d_out)

    # Create the pre-transform index
    index = IndexPreTransform(base_index, transform)

    # Check that parameters match
    assert index.d_in == d_out
    assert index.is_trained is True
    assert index.ntotal == 0
    assert len(index.transform_chain) == 1


def test_pre_transform_with_pca():
    """Test IndexPreTransform with PCA dimensionality reduction."""
    # Create a base index with reduced dimensions
    d_in = 128
    d_out = 64
    base_index = IndexFlatL2(d_out)

    # Create a PCA transform
    transform = PCATransform(d_in, d_out)

    # Create the pre-transform index
    index = IndexPreTransform(base_index, transform)

    # Check that parameters match
    assert index.d_in == d_in
    assert index.ntotal == 0
    assert len(index.transform_chain) == 1
    assert index.is_trained is False  # PCA needs training

    # Create training data
    num_train = 1000
    train_data = np.random.random((num_train, d_in)).astype(np.float32)

    # Train the index
    index.train(train_data)
    assert index.is_trained is True


def test_l2_norm_transform():
    """Test the L2 normalization transform."""
    d = 32
    transform = L2NormTransform(d)

    # Generate random vectors
    vectors = np.random.random((10, d)).astype(np.float32)

    # Apply the transform
    normalized = transform.apply(vectors)

    # Check that the vectors are normalized - their L2 norms should be 1
    for i in range(len(normalized)):
        norm = np.linalg.norm(normalized[i])
        assert np.isclose(norm, 1.0)


def test_pca_transform():
    """Test the PCA transform."""
    d_in = 64
    d_out = 32
    transform = PCATransform(d_in, d_out)

    # Generate random vectors for training
    train_data = np.random.random((100, d_in)).astype(np.float32)

    # Train the transform
    transform.train(train_data)

    # Generate test vectors
    test_data = np.random.random((10, d_in)).astype(np.float32)

    # Apply the transform
    transformed = transform.apply(test_data)

    # Check dimensions
    assert transformed.shape == (10, d_out)

    # Check reverse transform
    reconstructed = transform.reverse_transform(transformed)
    assert reconstructed.shape == (10, d_in)


def test_remap_dimensions_transform():
    """Test the dimension remapping transform."""
    d_in = 8
    d_out = 4

    # Create a transform that selects specific dimensions
    selected_dims = [0, 2, 4, 6]  # Select even-indexed dimensions
    transform = RemapDimensionsTransform(d_in, d_out, selected_dims)

    # Generate test vectors
    vectors = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [10, 11, 12, 13, 14, 15, 16, 17]
    ], dtype=np.float32)

    # Apply the transform
    transformed = transform.apply(vectors)

    # Check that the correct dimensions were selected
    assert transformed.shape == (2, d_out)
    np.testing.assert_array_equal(transformed[0], [0, 2, 4, 6])
    np.testing.assert_array_equal(transformed[1], [10, 12, 14, 16])


def test_transform_chain():
    """Test a chain of transforms."""
    d_in = 64
    d_mid = 32
    d_out = 32
    base_index = IndexFlatL2(d_out)

    # Create a chain of transforms: PCA followed by L2 normalization
    transform_chain = [
        PCATransform(d_in, d_mid),
        L2NormTransform(d_mid)
    ]

    # Create the pre-transform index
    index = IndexPreTransform(base_index, transform_chain)

    # Check parameters
    assert index.d_in == d_in
    assert index.is_trained is False  # PCA needs training
    assert len(index.transform_chain) == 2

    # Train the index
    train_data = np.random.random((100, d_in)).astype(np.float32)
    index.train(train_data)

    # Check that training worked
    assert index.is_trained is True

    # Add vectors
    vectors = np.random.random((10, d_in)).astype(np.float32)
    index.add(vectors)
    assert index.ntotal == 10

    # Test search
    query = np.random.random((1, d_in)).astype(np.float32)
    distances, indices = index.search(query, k=5)

    # Check shapes
    assert distances.shape == (1, 5)
    assert indices.shape == (1, 5)

    # Test reconstruction
    reconstructed = index.reconstruct(0)
    assert reconstructed.shape == (d_in,)


def test_vector_caching():
    """Test that original vectors are cached for reconstruction."""
    d_in = 64
    d_out = 32
    base_index = IndexFlatL2(d_out)

    # Create a PCA transform
    transform = PCATransform(d_in, d_out)

    # Create the pre-transform index
    index = IndexPreTransform(base_index, transform)

    # Train the index
    train_data = np.random.random((100, d_in)).astype(np.float32)
    index.train(train_data)

    # Add vectors
    vectors = np.random.random((10, d_in)).astype(np.float32)
    index.add(vectors)

    # Reconstruct a vector and verify it matches the original
    reconstructed = index.reconstruct(5)
    np.testing.assert_allclose(reconstructed, vectors[5], rtol=1e-5)

    # Test reconstruct_n
    reconstructed_batch = index.reconstruct_n(0, 10)
    np.testing.assert_allclose(reconstructed_batch, vectors, rtol=1e-5)


def test_prepend_transform():
    """Test prepending a transform to an existing IndexPreTransform."""
    d_in = 32
    d_out = 16
    base_index = IndexFlatL2(d_out)

    # Initial transform - PCA
    pca = PCATransform(d_in, d_out)
    index = IndexPreTransform(base_index, pca)

    # Train the initial setup
    train_data = np.random.random((100, d_in)).astype(np.float32)
    index.train(train_data)

    # Try to prepend an incompatible transform - should raise an error
    with pytest.raises(ValueError):
        index.prepend_transform(L2NormTransform(d_out))  # Wrong dimensions

    # Prepend a compatible transform
    new_d_in = 64
    new_transform = PCATransform(new_d_in, d_in)

    # Train the new transform separately
    new_train_data = np.random.random((100, new_d_in)).astype(np.float32)
    new_transform.train(new_train_data)

    # Now prepend the trained transform
    index.prepend_transform(new_transform)

    # Check the updated dimensions
    assert index.d_in == new_d_in
    assert len(index.transform_chain) == 2

    # Add and search should work with the new dimensions
    vectors = np.random.random((10, new_d_in)).astype(np.float32)
    index.add(vectors)

    query = np.random.random((1, new_d_in)).astype(np.float32)
    distances, indices = index.search(query, k=5)
    assert distances.shape == (1, 5)

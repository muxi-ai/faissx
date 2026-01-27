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
Tests for the IndexBinaryFlat index implementation.
"""

import faiss as original_faiss  # Original FAISS for comparison
import numpy as np
import pytest

# Import our implementation
from faissx.client.indices.binary_flat import IndexBinaryFlat


def create_random_binary_vectors(n: int, d: int) -> np.ndarray:
    """
    Create random binary vectors.

    Args:
        n (int): Number of vectors
        d (int): Dimension in bytes

    Returns:
        np.ndarray: Binary vectors with shape (n, d)
    """
    # Create random vectors as uint8 arrays
    return np.random.randint(0, 256, size=(n, d), dtype=np.uint8)


def compute_hamming_distances_reference(xq: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """
    Compute Hamming distances for reference testing.

    Args:
        xq (np.ndarray): Query vectors, shape (nq, d)
        xb (np.ndarray): Database vectors, shape (nb, d)

    Returns:
        np.ndarray: Distances, shape (nq, nb)
    """
    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2, "Dimensions must match"

    distances = np.zeros((nq, nb), dtype=np.float32)

    for i in range(nq):
        for j in range(nb):
            # XOR the vectors and count the number of 1 bits
            xor_result = np.bitwise_xor(xq[i], xb[j])
            # Count bits set in each byte
            hamming_dist = sum(bin(byte).count("1") for byte in xor_result)
            distances[i, j] = hamming_dist

    return distances


def test_binary_flat_init():
    """Test IndexBinaryFlat initialization."""
    d = 16  # 16 bytes = 128 bits
    index = IndexBinaryFlat(d)

    assert index.d == d
    assert index.code_size == d
    assert index.is_trained is True
    assert index.ntotal == 0


def test_binary_vector_add():
    """Test adding binary vectors."""
    d = 8  # 8 bytes = 64 bits
    n = 10
    index = IndexBinaryFlat(d)

    # Create some random binary vectors
    vectors = create_random_binary_vectors(n, d)

    # Add to index
    index.add(vectors)

    assert index.ntotal == n
    # Check that the stored vectors match the input
    np.testing.assert_array_equal(index._vectors, vectors)


def test_binary_vector_search():
    """Test searching binary vectors."""
    d = 8  # 8 bytes = 64 bits
    nb = 100  # database size
    nq = 10  # query size
    k = 5  # number of nearest neighbors

    # Create index
    index = IndexBinaryFlat(d)

    # Create database and query vectors
    db_vectors = create_random_binary_vectors(nb, d)
    query_vectors = create_random_binary_vectors(nq, d)

    # Add to index
    index.add(db_vectors)

    # Search
    distances, indices = index.search(query_vectors, k)

    # Check results
    assert distances.shape == (nq, k)
    assert indices.shape == (nq, k)

    # Compute reference distances
    ref_distances = compute_hamming_distances_reference(query_vectors, db_vectors)

    # For each query, check that the returned indices correspond to the actual smallest distances
    for i in range(nq):
        # Get indices of top k smallest distances (using reference calculation)
        ref_indices = np.argsort(ref_distances[i])[:k]
        ref_top_distances = ref_distances[i][ref_indices]

        # Check that the returned distances match (within numerical tolerance)
        np.testing.assert_allclose(distances[i], ref_top_distances, rtol=1e-5)

        # Check that the returned indices yield the same vectors
        # This is a bit trickier as there might be ties in Hamming distance
        for j, idx in enumerate(indices[i]):
            # For each returned index, the distance should match the reference
            actual_dist = ref_distances[i][idx]
            assert np.isclose(actual_dist, distances[i][j], rtol=1e-5)


def test_binary_reconstruction():
    """Test vector reconstruction."""
    d = 8
    n = 10
    index = IndexBinaryFlat(d)

    # Create some random binary vectors
    vectors = create_random_binary_vectors(n, d)

    # Add to index
    index.add(vectors)

    # Test reconstruct for each vector
    for i in range(n):
        reconstructed = index.reconstruct(i)
        np.testing.assert_array_equal(reconstructed, vectors[i])

    # Test reconstruct_n
    indices = [3, 5, 7]
    reconstructed = index.reconstruct_n(indices)
    expected = vectors[indices]
    np.testing.assert_array_equal(reconstructed, expected)


def test_binary_reset():
    """Test resetting the index."""
    d = 8
    n = 10
    index = IndexBinaryFlat(d)

    # Create some random binary vectors
    vectors = create_random_binary_vectors(n, d)

    # Add to index
    index.add(vectors)
    assert index.ntotal == n

    # Reset
    index.reset()
    assert index.ntotal == 0
    assert index._vectors is None


@pytest.mark.parametrize("d", [4, 8, 16])
def test_comparison_with_faiss(d):
    """Compare our implementation with the original FAISS."""
    try:
        # Try to use the original FAISS implementation for comparison
        original_index = original_faiss.IndexBinaryFlat(d * 8)  # FAISS uses bit dimensions

        # Create our index
        our_index = IndexBinaryFlat(d)

        # Create data
        nb = 50
        nq = 10
        k = 5

        db_vectors = create_random_binary_vectors(nb, d)
        query_vectors = create_random_binary_vectors(nq, d)

        # Add to both indices
        original_index.add(db_vectors)
        our_index.add(db_vectors)

        # Search with both
        faiss_distances, faiss_indices = original_index.search(query_vectors, k)
        our_distances, our_indices = our_index.search(query_vectors, k)

        # Convert FAISS distances to float32 for comparison
        faiss_distances = faiss_distances.astype(np.float32)

        # Check that the distances match
        np.testing.assert_allclose(our_distances, faiss_distances, rtol=1e-5)

                # The indices might differ if there are ties, but the distance values should match
        # For each query result, we check that the returned distances are the same
        for i in range(nq):
            for j in range(k):
                # Check that the distances match
                our_dist = our_distances[i, j]
                faiss_dist = faiss_distances[i, j]

                assert np.isclose(our_dist, faiss_dist, rtol=1e-5)

    except (ImportError, AttributeError):
        # Skip if FAISS is not available or doesn't have IndexBinaryFlat
        pytest.skip("Original FAISS not available or no IndexBinaryFlat support")


def test_error_handling():
    """Test that appropriate errors are raised for invalid inputs."""
    d = 8
    index = IndexBinaryFlat(d)

    # Test adding wrong dtype
    with pytest.raises(TypeError):
        index.add(np.random.rand(10, d).astype(np.float32))

    # Test adding wrong dimension
    with pytest.raises(ValueError):
        index.add(np.random.randint(0, 256, size=(10, d+1), dtype=np.uint8))

    # Test searching wrong dtype
    with pytest.raises(TypeError):
        index.search(np.random.rand(10, d).astype(np.float32), 5)

    # Test searching wrong dimension
    with pytest.raises(ValueError):
        index.search(np.random.randint(0, 256, size=(10, d+1), dtype=np.uint8), 5)

    # Test reconstructing invalid index
    with pytest.raises(IndexError):
        index.reconstruct(10)  # Index out of bounds

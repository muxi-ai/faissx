#!/usr/bin/env python3
"""
FAISSx IndexIVFFlat Example

This example demonstrates how to use the IndexIVFFlat implementation
with FAISSx. IndexIVFFlat provides improved search performance for large
datasets compared to flat indices by using an inverted file structure.

Key features demonstrated:
1. Creation of IVF index
2. Training the index (required for IVF indices)
3. Adding vectors
4. Searching for nearest neighbors
5. Performance comparison with flat index
"""

import numpy as np
import time
import faissx

# Configure to use remote server (optional)
# faissx.configure(url="tcp://localhost:45678")

# Generate random data for testing (10,000 128-dimensional vectors)
dimension = 128
num_vectors = 10000
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# Create some query vectors
num_queries = 10
query_vectors = np.random.random((num_queries, dimension)).astype('float32')


def test_flat_index():
    """Test with a flat index (baseline)"""
    start_time = time.time()

    # Create a flat L2 index
    flat_index = faissx.IndexFlatL2(dimension)

    # Add vectors
    flat_index.add(vectors)

    # Time for adding vectors
    add_time = time.time() - start_time
    print(f"Flat index add time: {add_time:.4f} seconds")

    # Search for nearest neighbors
    start_time = time.time()
    k = 5  # number of nearest neighbors
    distances, indices = flat_index.search(query_vectors, k)
    search_time = time.time() - start_time

    print(f"Flat index search time: {search_time:.4f} seconds")
    print(f"Flat index first result distances: {distances[0]}")
    return search_time


def test_ivf_index():
    """Test with an IVF index"""
    start_time = time.time()

    # Create a flat index for the quantizer
    quantizer = faissx.IndexFlatL2(dimension)

    # Create IVF index with 100 centroids
    nlist = 100  # number of clusters/cells
    ivf_index = faissx.IndexIVFFlat(quantizer, dimension, nlist)

    # Train the index (required for IVF indices)
    # We can use a subset of vectors for training
    train_size = min(num_vectors, 5000)
    train_vectors = vectors[:train_size]
    ivf_index.train(train_vectors)
    train_time = time.time() - start_time

    print(f"IVF index train time: {train_time:.4f} seconds")

    # Add vectors
    start_time = time.time()
    ivf_index.add(vectors)
    add_time = time.time() - start_time

    print(f"IVF index add time: {add_time:.4f} seconds")

    # Search for nearest neighbors
    start_time = time.time()
    k = 5  # number of nearest neighbors
    distances, indices = ivf_index.search(query_vectors, k)
    search_time = time.time() - start_time

    print(f"IVF index search time: {search_time:.4f} seconds")
    print(f"IVF index first result distances: {distances[0]}")
    return search_time


if __name__ == "__main__":
    print("Testing with flat index...")
    flat_time = test_flat_index()

    print("\nTesting with IVF index...")
    ivf_time = test_ivf_index()

    # Compare performance
    speedup = flat_time / ivf_time if ivf_time > 0 else float('inf')
    print("\nPerformance comparison:")
    print(f"IVF search is {speedup:.2f}x faster than flat search")
    print("\nNote: For small datasets, flat indices may be faster. The IVF advantage")
    print("increases with dataset size, often showing 10-100x speedups for millions of vectors.")

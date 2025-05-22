#!/usr/bin/env python3
"""
FAISSx IndexIVFPQ Example

This example demonstrates how to use the IndexIVFPQ class in FAISSx, which combines
Inverted File indexing with Product Quantization for highly efficient vector similarity
search. This approach is particularly well-suited for large datasets, offering excellent
memory efficiency and search performance.

The example covers:
1. Creating and configuring an IVFPQ index
2. Training the index (required for IVF and PQ indices)
3. Adding vectors to the index
4. Searching for similar vectors
5. Using both local and remote modes

IndexIVFPQ requires that the vector dimension is divisible by the number of subquantizers (M).
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Import FAISSx components
from faissx import configure
from faissx.client.index import IndexFlatL2, IndexIVFPQ

# Set random seed for reproducibility
np.random.seed(42)


def plot_results(data, query_point, indices, title):
    """Plot the data points, query point, and search results."""
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], s=8, alpha=0.5, label="Database vectors")
    plt.scatter(
        query_point[0],
        query_point[1],
        color="red",
        s=100,
        marker="*",
        label="Query vector",
    )

    # Mark search results
    results = data[indices]
    plt.scatter(
        results[:, 0],
        results[:, 1],
        color="green",
        s=50,
        marker="o",
        facecolors="none",
        label="Search results",
    )

    plt.title(title)
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()


def generate_data(n_samples=10000, n_features=128, n_clusters=10):
    """Generate synthetic data with cluster structure."""
    print(f"Generating {n_samples} vectors with {n_features} dimensions...")

    # Create clustered data for more realistic search scenarios
    centers = n_clusters
    X, _ = make_blobs(
        n_samples=n_samples, centers=centers, n_features=n_features, random_state=42
    )

    # Scale to unit norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms

    print(f"Data shape: {X.shape}, with {centers} natural clusters")
    return X


def benchmark_ivfpq_index(X, query, k=10):
    """
    Benchmark the IndexIVFPQ index in both local and remote modes.

    Args:
        X: Dataset vectors
        query: Query vector
        k: Number of neighbors to retrieve
    """
    dim = X.shape[1]

    # Determine appropriate PQ parameters
    # M (num subquantizers) must be a divisor of dimension
    # Typical values are 8, 16, 32, etc.
    M = 16  # Number of subquantizers
    if dim % M != 0:
        # Find the closest divisor of dim
        divisors = [i for i in range(1, min(64, dim + 1)) if dim % i == 0]
        M = min(divisors, key=lambda x: abs(x - 16))
        print(f"Adjusted M to {M} to be a divisor of dimension {dim}")

    nbits = 8  # Bits per subquantizer (typically 8)
    nlist = 100  # Number of IVF clusters/partitions

    # Local mode (directly using FAISS)
    print("\n===== Local Mode: IndexIVFPQ =====")

    # Create the index
    print(
        f"Creating IVFPQ index with {nlist} clusters, {M} subquantizers, {nbits} bits"
    )
    # Create a quantizer for clustering
    quantizer = IndexFlatL2(dim)

    # Create the IVFPQ index
    index = IndexIVFPQ(quantizer, dim, M, nbits, nlist)

    # Train the index (required for both IVF and PQ components)
    print(f"Training the index on {len(X)} vectors...")
    t0 = time.time()
    index.train(X)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.4f} seconds")

    # Add vectors to the index
    print(f"Adding {len(X)} vectors to the index...")
    t0 = time.time()
    index.add(X)
    add_time = time.time() - t0
    print(f"Added {index.ntotal} vectors in {add_time:.4f} seconds")

    # Search for similar vectors
    print(f"Searching for {k} nearest neighbors...")
    t0 = time.time()
    distances, indices = index.search(query, k)
    search_time = time.time() - t0
    print(f"Search completed in {search_time:.4f} seconds")

    # Display results
    print("\nSearch Results (Local Mode):")
    print(f"{'Index':<8} {'Distance':<12}")
    for idx, (i, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"{i:<8} {dist:<12.6f}")

    # Keep results for comparison with remote mode
    local_results = indices[0]

    # If the dataset is 2D, visualize the results
    if X.shape[1] == 2:
        plot_results(X, query, indices[0], "IndexIVFPQ Local Mode Results")

    # Remote mode (using FAISSx server)
    print("\n===== Remote Mode: IndexIVFPQ =====")

    # Configure to use remote server (assuming server is already running)
    # Normally you would set the actual server URL and API key
    configure(url="tcp://localhost:45678")

    # Create the remote index
    print(
        f"Creating remote IVFPQ index with {nlist} clusters, {M} subquantizers, {nbits} bits"
    )
    quantizer = IndexFlatL2(dim)
    index = IndexIVFPQ(quantizer, dim, M, nbits, nlist)

    # Train the index (required for IVF and PQ)
    print(f"Training the remote index on {len(X)} vectors...")
    t0 = time.time()
    index.train(X)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.4f} seconds")

    # Add vectors to the index
    print(f"Adding {len(X)} vectors to the index...")
    t0 = time.time()
    index.add(X)
    add_time = time.time() - t0
    print(f"Added {index.ntotal} vectors in {add_time:.4f} seconds")

    # Search for similar vectors
    print(f"Searching for {k} nearest neighbors...")
    t0 = time.time()
    distances, indices = index.search(query, k)
    search_time = time.time() - t0
    print(f"Search completed in {search_time:.4f} seconds")

    # Display results
    print("\nSearch Results (Remote Mode):")
    print(f"{'Index':<8} {'Distance':<12}")
    for idx, (i, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"{i:<8} {dist:<12.6f}")

    # If the dataset is 2D, visualize the results
    if X.shape[1] == 2:
        plot_results(X, query, indices[0], "IndexIVFPQ Remote Mode Results")

    # Compare local and remote results for consistency
    remote_results = indices[0]
    common_results = np.intersect1d(local_results, remote_results)

    print("\n===== Results Comparison =====")
    print(f"Local mode found indices: {local_results}")
    print(f"Remote mode found indices: {remote_results}")
    print(f"Number of common results: {len(common_results)} out of {k}")
    print(f"Common indices: {common_results}")


if __name__ == "__main__":
    # For visualization, use a 2D dataset
    X_2d = generate_data(n_samples=5000, n_features=2, n_clusters=10)
    query_2d = np.array([X_2d[42]], dtype=np.float32)  # Use a sample as query

    # Run benchmark with 2D data
    benchmark_ivfpq_index(X_2d, query_2d, k=10)

    # For a more realistic example, use higher-dimensional data
    # Make sure dimension is divisible by typical M values like 16
    X_high = generate_data(n_samples=10000, n_features=128, n_clusters=20)
    query_high = np.array([X_high[100]], dtype=np.float32)  # Use a sample as query

    # Run benchmark with high-dimensional data
    benchmark_ivfpq_index(X_high, query_high, k=10)

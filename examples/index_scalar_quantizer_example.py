#!/usr/bin/env python
"""
FAISSx IndexScalarQuantizer Example

This example demonstrates how to use the IndexScalarQuantizer class in FAISSx, which
uses scalar quantization for efficient memory usage while maintaining search accuracy.
It's a good compromise between the high memory usage of flat indices and the lower
precision of product quantization.

The example covers:
1. Creating and configuring a scalar quantizer index
2. Adding vectors to the index (no training needed)
3. Searching for similar vectors
4. Using both local and remote modes
5. Comparing memory usage with flat indices
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sys
import gc
import os

# Import FAISSx components
from faissx import configure
from faissx.client.index import IndexFlatL2, IndexScalarQuantizer

# Set random seed for reproducibility
np.random.seed(42)


def get_memory_usage(obj):
    """Get approximate memory usage of an object in MB"""
    # Force garbage collection to get more accurate memory usage
    gc.collect()
    # Get memory usage of the process
    start_memory = get_process_memory()

    # Store reference to object and return memory delta
    _ = sys.getsizeof(obj)
    end_memory = get_process_memory()

    return end_memory - start_memory


def get_process_memory():
    """Return the memory usage of the current process in MB"""
    import psutil

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def plot_results(X, query, indices, title):
    """
    Plot the results of a vector search, showing the query vector and its nearest neighbors.

    Args:
        X (np.ndarray): The dataset vectors
        query (np.ndarray): The query vector
        indices (np.ndarray): Indices of the nearest neighbors
        title (str): Title for the plot
    """
    # Skip plotting in dimensions higher than 2
    if X.shape[1] > 2:
        print(f"Cannot visualize results for dimension {X.shape[1]} > 2")
        return

    plt.figure(figsize=(10, 6))

    # Ensure query is properly reshaped as a single vector
    if len(query.shape) > 1 and query.shape[0] == 1:
        query_point = query[0]  # Extract the first query if it's a batch
    else:
        query_point = query  # Use as is if it's already a single vector

    # Plot all data points
    plt.scatter(X[:, 0], X[:, 1], c="lightgrey", alpha=0.5, s=10)

    # Plot query point
    plt.scatter(
        query_point[0],
        query_point[1],
        color="red",
        s=100,
        marker="*",
        label="Query vector",
    )

    # Plot nearest neighbors
    nearest = X[indices]
    plt.scatter(
        nearest[:, 0], nearest[:, 1], color="blue", s=50, label="Nearest neighbors"
    )

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_data(n_samples=10000, n_features=128, n_clusters=10):
    """Generate synthetic data with cluster structure"""
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


def benchmark_scalar_quantizer_index(X, query, k=10):
    """
    Benchmark the IndexScalarQuantizer index in both local and remote modes,
    comparing with the flat index.

    Args:
        X: Dataset vectors
        query: Query vector
        k: Number of neighbors to retrieve
    """
    dim = X.shape[1]

    # Compare with Flat index first for accuracy reference
    print("\n===== Reference: IndexFlatL2 =====")

    # Create the flat index for comparison
    flat_index = IndexFlatL2(dim)

    # Add vectors to the flat index
    print(f"Adding {len(X)} vectors to the flat index...")
    t0 = time.time()
    flat_index.add(X)
    add_time = time.time() - t0
    print(f"Added {flat_index.ntotal} vectors in {add_time:.4f} seconds")

    # Get approximate memory usage
    flat_memory = get_memory_usage(flat_index)
    print(f"Approximate memory usage: {flat_memory:.2f} MB")

    # Search for similar vectors
    print(f"Searching for {k} nearest neighbors...")
    t0 = time.time()
    flat_distances, flat_indices = flat_index.search(query, k)
    search_time = time.time() - t0
    print(f"Search completed in {search_time:.4f} seconds")

    # Display results
    print("\nReference Search Results (Flat Index):")
    print(f"{'Index':<8} {'Distance':<12}")
    for idx, (i, dist) in enumerate(zip(flat_indices[0], flat_distances[0])):
        print(f"{i:<8} {dist:<12.6f}")

    if X.shape[1] == 2:
        plot_results(X, query, flat_indices[0], "IndexFlatL2 Reference Results")

    # Local mode (directly using FAISS)
    print("\n===== Local Mode: IndexScalarQuantizer =====")

    # Create the index
    print(f"Creating ScalarQuantizer index with dimension {dim}")
    scalar_index = IndexScalarQuantizer(dim)

    # Add vectors to the index (no training needed)
    print(f"Adding {len(X)} vectors to the index...")
    t0 = time.time()
    scalar_index.add(X)
    add_time = time.time() - t0
    print(f"Added {scalar_index.ntotal} vectors in {add_time:.4f} seconds")

    # Get approximate memory usage
    scalar_memory = get_memory_usage(scalar_index)
    print(f"Approximate memory usage: {scalar_memory:.2f} MB")
    print(f"Memory compression ratio: {flat_memory / scalar_memory:.2f}x")

    # Search for similar vectors
    print(f"Searching for {k} nearest neighbors...")
    t0 = time.time()
    distances, indices = scalar_index.search(query, k)
    search_time = time.time() - t0
    print(f"Search completed in {search_time:.4f} seconds")

    # Display results
    print("\nSearch Results (Local Mode):")
    print(f"{'Index':<8} {'Distance':<12}")
    for idx, (i, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"{i:<8} {dist:<12.6f}")

    # Keep results for comparison
    local_results = indices[0]

    # If the dataset is 2D, visualize the results
    if X.shape[1] == 2:
        plot_results(X, query, indices[0], "IndexScalarQuantizer Local Mode Results")

    # Compare the results with flat index
    common_with_flat = np.intersect1d(flat_indices[0], local_results)
    print(f"\nAccuracy comparison: {len(common_with_flat)}/{k} matches with flat index")

    # Remote mode (using FAISSx server)
    print("\n===== Remote Mode: IndexScalarQuantizer =====")

    # Configure to use remote server (assuming server is already running)
    # Normally you would set the actual server URL and API key
    configure(url="tcp://localhost:45678")

    # Create the remote index
    print(f"Creating remote ScalarQuantizer index with dimension {dim}")
    scalar_index = IndexScalarQuantizer(dim)

    # Add vectors to the index
    print(f"Adding {len(X)} vectors to the index...")
    t0 = time.time()
    scalar_index.add(X)
    add_time = time.time() - t0
    print(f"Added {scalar_index.ntotal} vectors in {add_time:.4f} seconds")

    # Search for similar vectors
    print(f"Searching for {k} nearest neighbors...")
    t0 = time.time()
    distances, indices = scalar_index.search(query, k)
    search_time = time.time() - t0
    print(f"Search completed in {search_time:.4f} seconds")

    # Display results
    print("\nSearch Results (Remote Mode):")
    print(f"{'Index':<8} {'Distance':<12}")
    for idx, (i, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"{i:<8} {dist:<12.6f}")

    # If the dataset is 2D, visualize the results
    if X.shape[1] == 2:
        plot_results(X, query, indices[0], "IndexScalarQuantizer Remote Mode Results")

    # Compare local and remote results for consistency
    remote_results = indices[0]
    common_results = np.intersect1d(local_results, remote_results)

    print("\n===== Results Comparison =====")
    print(f"Local mode found indices: {local_results}")
    print(f"Remote mode found indices: {remote_results}")
    print(f"Number of common results: {len(common_results)} out of {k}")
    print(f"Common indices: {common_results}")

    # Compare with flat index for accuracy
    common_with_flat = np.intersect1d(flat_indices[0], remote_results)
    print(f"Accuracy comparison: {len(common_with_flat)}/{k} matches with flat index")


if __name__ == "__main__":
    # For visualization, use a 2D dataset
    X_2d = generate_data(n_samples=5000, n_features=2, n_clusters=10)
    query_2d = np.array([X_2d[42]], dtype=np.float32)  # Use a sample as query

    # Run benchmark with 2D data
    benchmark_scalar_quantizer_index(X_2d, query_2d, k=10)

    # For a more realistic example, use higher-dimensional data
    X_high = generate_data(n_samples=10000, n_features=128, n_clusters=20)
    query_high = np.array([X_high[100]], dtype=np.float32)  # Use a sample as query

    # Run benchmark with high-dimensional data
    benchmark_scalar_quantizer_index(X_high, query_high, k=10)

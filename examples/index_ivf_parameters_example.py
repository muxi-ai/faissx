#!/usr/bin/env python3
"""
FAISSx IndexIVFFlat Parameter Control Example

This example demonstrates how to use the nprobe parameter with the IndexIVFFlat class
to control the search-time accuracy vs. performance tradeoff.

The nprobe parameter specifies how many clusters (partitions) to search during query time.
Increasing nprobe results in:
- More accurate search results
- Slower search performance

This example compares search results and performance across different nprobe values.
"""

import numpy as np
import time
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Import FAISSx components
from faissx.client.index import IndexFlatL2, IndexIVFFlat

# Set random seed for reproducibility
np.random.seed(42)


def generate_data(n_samples=10000, n_features=128, n_clusters=100):
    """Generate synthetic data with cluster structure"""
    print(f"Generating {n_samples} vectors with {n_features} dimensions...")

    # Create clustered data for more realistic search scenarios
    centers = n_clusters
    X, _ = make_blobs(
        n_samples=n_samples, centers=centers, n_features=n_features, random_state=42
    )

    # Scale to unit norm (optional but helps with vector comparisons)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms

    print(f"Data shape: {X.shape}, with {centers} natural clusters")
    return X


def benchmark_nprobe_settings(data, query, k=10):
    """
    Benchmark different nprobe settings for IndexIVFFlat.

    Args:
        data (np.ndarray): Dataset vectors to index
        query (np.ndarray): Query vector
        k (int): Number of neighbors to retrieve
    """
    dim = data.shape[1]
    nlist = 100  # Number of clusters - in practice could be sqrt(n) where n is dataset size

    print("\n===== Creating Ground Truth with IndexFlatL2 =====")
    # Use flat index for reference (exact search)
    flat_index = IndexFlatL2(dim)
    flat_index.add(data)

    t0 = time.time()
    flat_distances, flat_indices = flat_index.search(query, k)
    flat_search_time = time.time() - t0

    print(f"Exact search completed in {flat_search_time:.4f} seconds")
    print(f"Reference results: {flat_indices[0]}")

    # Create and train IVF index
    print("\n===== Creating and Training IndexIVFFlat =====")
    # Create quantizer (used for clustering)
    quantizer = IndexFlatL2(dim)
    # Create IVF index with the quantizer
    ivf_index = IndexIVFFlat(quantizer, dim, nlist)

    # Train the index - this step is required for IVF indices
    print(f"Training the index on {len(data)} vectors...")
    t0 = time.time()
    ivf_index.train(data)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.4f} seconds")

    # Add vectors to the index
    print(f"Adding {len(data)} vectors to the index...")
    t0 = time.time()
    ivf_index.add(data)
    add_time = time.time() - t0
    print(f"Added {ivf_index.ntotal} vectors in {add_time:.4f} seconds")

    # Test different nprobe values
    nprobe_values = [1, 5, 10, 20, 50, 100]
    times = []
    accuracies = []

    print("\n===== Testing Different nprobe Settings =====")
    print(f"{'nprobe':<10} {'Search Time (s)':<20} {'Accuracy (%)':<15} {'Common Results'}")
    print("-" * 70)

    for nprobe in nprobe_values:
        # Set the number of probes
        ivf_index.set_nprobe(nprobe)

        # Perform search with current nprobe setting
        t0 = time.time()
        distances, indices = ivf_index.search(query, k)
        search_time = time.time() - t0
        times.append(search_time)

        # Compare with ground truth (flat index results)
        common_results = np.intersect1d(flat_indices[0], indices[0])
        accuracy = len(common_results) / k * 100
        accuracies.append(accuracy)

        print(f"{nprobe:<10} {search_time:<20.4f} {accuracy:<15.2f} {common_results}")

    # Plot the results
    plt.figure(figsize=(12, 5))

    # Plot search time
    plt.subplot(1, 2, 1)
    plt.plot(nprobe_values, times, 'o-', label='Search Time')
    plt.axhline(y=flat_search_time, color='r', linestyle='--', label='Flat Index (Exact)')
    plt.xlabel('nprobe')
    plt.ylabel('Search Time (seconds)')
    plt.title('Search Time vs nprobe')
    plt.grid(True)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(nprobe_values, accuracies, 'o-', color='green')
    plt.xlabel('nprobe')
    plt.ylabel('Accuracy (%)')
    plt.title('Search Accuracy vs nprobe')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('nprobe_benchmark.png')
    plt.show()


if __name__ == "__main__":
    # Generate dataset
    data = generate_data(n_samples=10000, n_features=128, n_clusters=100)

    # Create a query vector (use a sample from the dataset for simplicity)
    query = np.array([data[42]], dtype=np.float32)  # Use a sample as query

    # Run benchmark
    benchmark_nprobe_settings(data, query, k=10)

    print("\n===== Conclusion =====")
    print("Higher nprobe values:")
    print("  + Increase search accuracy")
    print("  - Decrease search performance (slower)")
    print("\nLower nprobe values:")
    print("  + Increase search performance (faster)")
    print("  - Decrease search accuracy")
    print("\nRecommendation: Start with nprobe=1 and increase until you get "
          "the desired accuracy/performance tradeoff.")

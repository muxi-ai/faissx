#!/usr/bin/env python
"""
FAISSx Optimization Controls Example

This example demonstrates how to use the optimization controls in FAISSx:
1. Fine-grained parameter control for different index types
2. Memory management options for efficient resource usage
3. Performance tuning for different workloads

These features allow customizing FAISSx for different use cases and hardware
configurations, balancing between search quality, speed, and memory usage.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Import FAISSx components
from faissx.client.indices import IndexFlatL2, IndexIVFFlat, IndexHNSWFlat
from faissx.client.optimization import memory_manager

# Set up logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(n_vectors=10000, n_dimensions=128, n_clusters=10):
    """Generate sample vector data for indexing and searching."""
    print(f"Generating {n_vectors} vectors with {n_dimensions} dimensions...")

    # Generate data with well-defined clusters
    vectors, _ = make_blobs(
        n_samples=n_vectors,
        n_features=n_dimensions,
        centers=n_clusters,
        cluster_std=10.0
    )

    # Normalize vectors (optional but recommended for some distance metrics)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Split into database and query vectors
    db_vectors = vectors[:n_vectors-100].astype(np.float32)
    query_vectors = vectors[n_vectors-100:].astype(np.float32)

    return db_vectors, query_vectors


def test_flat_index_parameters():
    """Test parameter settings with a flat index."""
    print("\n===== Testing Flat Index with Parameter Settings =====")

    # Generate data
    db_vectors, query_vectors = generate_data(n_vectors=5000, n_dimensions=64)
    n_dimensions = db_vectors.shape[1]

    # Create a flat index
    index = IndexFlatL2(n_dimensions)

    # Add vectors
    index.add(db_vectors)
    print(f"Added {index.ntotal} vectors to the index")

    # Set k_factor parameter (for oversampling)
    index.set_parameter('k_factor', 2.0)
    print(f"Set k_factor to {index.get_parameter('k_factor')}")

    # Set batch size parameter
    index.set_parameter('batch_size', 500)
    print(f"Set batch_size to {index.get_parameter('batch_size')}")

    # Check all parameters
    print("\nAll applicable parameters for this index:")
    params = index.get_parameters()
    for name, value in params.items():
        print(f"  {name}: {value}")

    # Perform search with default parameters
    k = 10
    print(f"\nPerforming search for top {k} neighbors...")
    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    elapsed = time.time() - start_time

    print(f"Search completed in {elapsed:.4f} seconds")
    print(f"Average distance to nearest neighbor: {np.mean(distances[:, 0]):.4f}")

    # Reset parameters and check they return to defaults
    index.reset_parameters()
    print("\nAfter resetting parameters:")
    params = index.get_parameters()
    for name, value in params.items():
        print(f"  {name}: {value}")


def compare_ivf_parameters():
    """Compare different parameter settings for IVF indices."""
    print("\n===== Comparing IVF Index with Different nprobe Settings =====")

    # Generate data
    db_vectors, query_vectors = generate_data(n_vectors=20000, n_dimensions=128, n_clusters=100)
    n_dimensions = db_vectors.shape[1]

    # Create an IVF index with 100 centroids
    nlist = 100
    index = IndexIVFFlat(n_dimensions, nlist)

    # Train and add vectors
    print("Training the index...")
    index.train(db_vectors)

    print("Adding vectors...")
    index.add(db_vectors)
    print(f"Added {index.ntotal} vectors to the index")

    # Test different nprobe values
    k = 10
    nprobe_values = [1, 5, 10, 20, 50]
    times = []
    recall_rates = []

    # Establish ground truth with flat index
    print("\nEstablishing ground truth with flat index...")
    flat_index = IndexFlatL2(n_dimensions)
    flat_index.add(db_vectors)
    gt_distances, gt_indices = flat_index.search(query_vectors, k)

    print("\nTesting different nprobe values:")
    for nprobe in nprobe_values:
        # Set nprobe parameter
        index.set_parameter('nprobe', nprobe)

        # Perform search
        start_time = time.time()
        distances, indices = index.search(query_vectors, k)
        elapsed = time.time() - start_time
        times.append(elapsed)

        # Calculate recall
        recall = 0
        for i in range(len(query_vectors)):
            gt_set = set(gt_indices[i])
            result_set = set(indices[i])
            recall += len(gt_set.intersection(result_set)) / k
        recall /= len(query_vectors)
        recall_rates.append(recall)

        print(f"  nprobe={nprobe}: Search time={elapsed:.4f}s, Recall={recall:.4f}")

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(nprobe_values, times, 'o-')
    plt.xlabel('nprobe')
    plt.ylabel('Search time (s)')
    plt.title('Search Time vs nprobe')

    plt.subplot(1, 2, 2)
    plt.plot(nprobe_values, recall_rates, 'o-')
    plt.xlabel('nprobe')
    plt.ylabel('Recall@10')
    plt.title('Recall vs nprobe')

    plt.tight_layout()
    plt.savefig('ivf_parameter_comparison.png')
    print("Saved plot to 'ivf_parameter_comparison.png'")


def test_memory_management():
    """Test memory management options."""
    print("\n===== Testing Memory Management =====")

    # Set memory management options
    print("Setting memory management options...")
    memory_manager.set_option('max_memory_usage_mb', 1024)  # 1GB limit
    memory_manager.set_option('use_memory_mapping', True)
    memory_manager.set_option('index_cache_size', 50)
    memory_manager.set_option('auto_unload_unused_indices', True)

    # Display all options
    print("\nCurrent memory management settings:")
    options = memory_manager.get_all_options()
    for name, value in options.items():
        print(f"  {name}: {value}")

    # Generate some small indices to test with
    print("\nCreating test indices...")
    indices = []
    for i in range(10):
        db_vectors, _ = generate_data(n_vectors=1000, n_dimensions=64)
        index = IndexFlatL2(64)
        index.add(db_vectors)
        indices.append(index)
        print(f"  Created index {i} with {index.ntotal} vectors")

    # Estimate memory usage
    print("\nEstimated memory usage of each index:")
    for i, index in enumerate(indices):
        size_bytes = index.estimate_memory_usage()
        size_mb = size_bytes / (1024 * 1024)
        print(f"  Index {i}: {size_mb:.2f} MB")

    # Simulate activity on some indices
    print("\nSimulating index usage...")
    query_vector = np.random.rand(1, 64).astype(np.float32)
    for i in [0, 2, 5, 7]:
        indices[i].search(query_vector, 5)
        print(f"  Used index {i}")

    # Reset options
    print("\nResetting memory options to defaults...")
    memory_manager.reset_options()

    print("\nMemory management options after reset:")
    options = memory_manager.get_all_options()
    for name, value in options.items():
        print(f"  {name}: {value}")


def test_hnsw_parameters():
    """Test HNSW-specific parameters."""
    print("\n===== Testing HNSW Parameters =====")

    # Generate data
    db_vectors, query_vectors = generate_data(n_vectors=10000, n_dimensions=64)
    n_dimensions = db_vectors.shape[1]

    # Create HNSW index
    M = 16  # Number of connections per layer
    efConstruction = 40  # Default
    index = IndexHNSWFlat(n_dimensions, M)

    # Set parameters before adding vectors
    print("Setting efConstruction parameter...")
    index.set_parameter('efConstruction', efConstruction)

    # Add vectors
    print("Adding vectors...")
    index.add(db_vectors)
    print(f"Added {index.ntotal} vectors to the index")

    # Test different efSearch values
    k = 10
    ef_search_values = [10, 20, 40, 80, 100]
    times = []
    recall_rates = []

    # Establish ground truth with flat index
    print("\nEstablishing ground truth with flat index...")
    flat_index = IndexFlatL2(n_dimensions)
    flat_index.add(db_vectors)
    gt_distances, gt_indices = flat_index.search(query_vectors, k)

    print("\nTesting different efSearch values:")
    for ef in ef_search_values:
        # Set efSearch parameter
        index.set_parameter('efSearch', ef)

        # Perform search
        start_time = time.time()
        distances, indices = index.search(query_vectors, k)
        elapsed = time.time() - start_time
        times.append(elapsed)

        # Calculate recall
        recall = 0
        for i in range(len(query_vectors)):
            gt_set = set(gt_indices[i])
            result_set = set(indices[i])
            recall += len(gt_set.intersection(result_set)) / k
        recall /= len(query_vectors)
        recall_rates.append(recall)

        print(f"  efSearch={ef}: Search time={elapsed:.4f}s, Recall={recall:.4f}")

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ef_search_values, times, 'o-')
    plt.xlabel('efSearch')
    plt.ylabel('Search time (s)')
    plt.title('Search Time vs efSearch')

    plt.subplot(1, 2, 2)
    plt.plot(ef_search_values, recall_rates, 'o-')
    plt.xlabel('efSearch')
    plt.ylabel('Recall@10')
    plt.title('Recall vs efSearch')

    plt.tight_layout()
    plt.savefig('hnsw_parameter_comparison.png')
    print("Saved plot to 'hnsw_parameter_comparison.png'")


if __name__ == "__main__":
    print("FAISSx Optimization Controls Example")
    print("===================================\n")

    # Run the tests
    test_flat_index_parameters()
    compare_ivf_parameters()
    test_memory_management()
    test_hnsw_parameters()

    print("\nExample completed successfully!")

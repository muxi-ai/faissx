#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx range search example
# https://github.com/muxi-ai/faissx
#
# Run this example to demonstrate range search functionality across
# different index types (FlatL2, IVF, HNSW, PQ)

import time
import numpy as np
import psutil
import faissx
import matplotlib.pyplot as plt

# Configuration
dimension = 128  # Vector dimension
num_vectors = 10000  # Number of vectors to add
num_queries = 10  # Number of query vectors to search for
radius = 20.0  # Radius for range search


# Function to measure memory usage
def process_memory():
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


# Generate random data
print(f"Generating {num_vectors} random vectors of dimension {dimension}...")
np.random.seed(42)  # For reproducibility
vectors = np.random.random((num_vectors, dimension)).astype("float32")
vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]  # Normalize

# Generate query vectors
query_vectors = np.random.random((num_queries, dimension)).astype("float32")
query_vectors = (
    query_vectors / np.linalg.norm(query_vectors, axis=1)[:, np.newaxis]
)  # Normalize

# Results dictionary
results = {}


def test_flat_index():
    """Test range search with a flat index"""
    print("\n----- Testing Flat L2 Index Range Search -----")

    # Record memory before
    mem_before = process_memory()

    # Create flat index
    flat_index = faissx.IndexFlatL2(dimension)

    # Add vectors
    start_time = time.time()
    flat_index.add(vectors)
    add_time = time.time() - start_time
    print(f"Add time: {add_time:.4f} seconds")

    # Record memory after adding vectors
    mem_after = process_memory()
    mem_usage = mem_after - mem_before
    print(f"Approximate memory usage: {mem_usage:.2f} MB")

    # Regular search for comparison
    start_time = time.time()
    k = 10
    distances, indices = flat_index.search(query_vectors, k)
    regular_search_time = time.time() - start_time
    print(f"Regular search time (k={k}): {regular_search_time:.4f} seconds")

    # Range search
    start_time = time.time()
    lims, distances, indices = flat_index.range_search(query_vectors, radius)
    range_search_time = time.time() - start_time

    # Calculate results per query
    results_per_query = np.diff(lims)
    avg_results = np.mean(results_per_query) if len(results_per_query) > 0 else 0

    print(f"Range search time (radius={radius}): {range_search_time:.4f} seconds")
    print(f"Average results per query: {avg_results:.1f}")
    print(f"Total results: {len(distances)}")

    return {
        "add_time": add_time,
        "regular_search_time": regular_search_time,
        "range_search_time": range_search_time,
        "memory": mem_usage,
        "results_per_query": results_per_query,
        "total_results": len(distances),
    }


def test_ivf_index():
    """Test range search with an IVF index"""
    print("\n----- Testing IVF Index Range Search -----")

    # Record memory before
    mem_before = process_memory()

    # Create IVF index
    nlist = 100  # Number of clusters
    quantizer = faissx.IndexFlatL2(dimension)
    ivf_index = faissx.IndexIVFFlat(quantizer, dimension, nlist)

    # Train the index
    print("Training IVF index...")
    start_time = time.time()
    ivf_index.train(vectors[: min(2000, len(vectors))])  # Use subset for training
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.4f} seconds")

    # Add vectors
    start_time = time.time()
    ivf_index.add(vectors)
    add_time = time.time() - start_time
    print(f"Add time: {add_time:.4f} seconds")

    # Record memory after adding vectors
    mem_after = process_memory()
    mem_usage = mem_after - mem_before
    print(f"Approximate memory usage: {mem_usage:.2f} MB")

    # Regular search for comparison
    start_time = time.time()
    k = 10
    distances, indices = ivf_index.search(query_vectors, k)
    regular_search_time = time.time() - start_time
    print(f"Regular search time (k={k}): {regular_search_time:.4f} seconds")

    # Range search
    start_time = time.time()
    lims, distances, indices = ivf_index.range_search(query_vectors, radius)
    range_search_time = time.time() - start_time

    # Calculate results per query
    results_per_query = np.diff(lims)
    avg_results = np.mean(results_per_query) if len(results_per_query) > 0 else 0

    print(f"Range search time (radius={radius}): {range_search_time:.4f} seconds")
    print(f"Average results per query: {avg_results:.1f}")
    print(f"Total results: {len(distances)}")

    return {
        "train_time": train_time,
        "add_time": add_time,
        "regular_search_time": regular_search_time,
        "range_search_time": range_search_time,
        "memory": mem_usage,
        "results_per_query": results_per_query,
        "total_results": len(distances),
    }


def test_hnsw_index():
    """Test range search with HNSW index"""
    print("\n----- Testing HNSW Index Range Search -----")

    # Record memory before
    mem_before = process_memory()

    # Create HNSW index
    M = 32  # Number of connections per node
    hnsw_index = faissx.IndexHNSWFlat(dimension, M)

    # Add vectors
    start_time = time.time()
    hnsw_index.add(vectors)
    add_time = time.time() - start_time
    print(f"Add time: {add_time:.4f} seconds")

    # Record memory after adding vectors
    mem_after = process_memory()
    mem_usage = mem_after - mem_before
    print(f"Approximate memory usage: {mem_usage:.2f} MB")

    # Regular search for comparison
    start_time = time.time()
    k = 10
    distances, indices = hnsw_index.search(query_vectors, k)
    regular_search_time = time.time() - start_time
    print(f"Regular search time (k={k}): {regular_search_time:.4f} seconds")

    # Range search
    start_time = time.time()
    lims, distances, indices = hnsw_index.range_search(query_vectors, radius)
    range_search_time = time.time() - start_time

    # Calculate results per query
    results_per_query = np.diff(lims)
    avg_results = np.mean(results_per_query) if len(results_per_query) > 0 else 0

    print(f"Range search time (radius={radius}): {range_search_time:.4f} seconds")
    print(f"Average results per query: {avg_results:.1f}")
    print(f"Total results: {len(distances)}")

    return {
        "add_time": add_time,
        "regular_search_time": regular_search_time,
        "range_search_time": range_search_time,
        "memory": mem_usage,
        "results_per_query": results_per_query,
        "total_results": len(distances),
    }


def test_pq_index():
    """Test range search with PQ index"""
    print("\n----- Testing PQ Index Range Search -----")

    # Record memory before
    mem_before = process_memory()

    # Create PQ index
    M = 8  # Number of subquantizers
    nbits = 8  # Bits per subquantizer
    pq_index = faissx.IndexPQ(dimension, M, nbits)

    # Train the index
    print("Training PQ index...")
    start_time = time.time()
    pq_index.train(vectors[: min(2000, len(vectors))])  # Use subset for training
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.4f} seconds")

    # Add vectors
    start_time = time.time()
    pq_index.add(vectors)
    add_time = time.time() - start_time
    print(f"Add time: {add_time:.4f} seconds")

    # Record memory after adding vectors
    mem_after = process_memory()
    mem_usage = mem_after - mem_before
    print(f"Approximate memory usage: {mem_usage:.2f} MB")

    # Regular search for comparison
    start_time = time.time()
    k = 10
    distances, indices = pq_index.search(query_vectors, k)
    regular_search_time = time.time() - start_time
    print(f"Regular search time (k={k}): {regular_search_time:.4f} seconds")

    # Range search
    start_time = time.time()
    lims, distances, indices = pq_index.range_search(query_vectors, radius)
    range_search_time = time.time() - start_time

    # Calculate results per query
    results_per_query = np.diff(lims)
    avg_results = np.mean(results_per_query) if len(results_per_query) > 0 else 0

    print(f"Range search time (radius={radius}): {range_search_time:.4f} seconds")
    print(f"Average results per query: {avg_results:.1f}")
    print(f"Total results: {len(distances)}")

    return {
        "train_time": train_time,
        "add_time": add_time,
        "regular_search_time": regular_search_time,
        "range_search_time": range_search_time,
        "memory": mem_usage,
        "results_per_query": results_per_query,
        "total_results": len(distances),
    }


def plot_results(results):
    """Plot performance comparison of the different index types"""
    index_types = list(results.keys())

    # Prepare data for plotting
    add_times = [results[idx].get("add_time", 0) for idx in index_types]
    range_search_times = [
        results[idx].get("range_search_time", 0) for idx in index_types
    ]
    memory_usage = [results[idx].get("memory", 0) for idx in index_types]

    # Create plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Add times
    ax1.bar(index_types, add_times)
    ax1.set_title("Add Times (s)")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_xticklabels(index_types, rotation=45)

    # Range search times
    ax2.bar(index_types, range_search_times)
    ax2.set_title("Range Search Times (s)")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_xticklabels(index_types, rotation=45)

    # Memory usage
    ax3.bar(index_types, memory_usage)
    ax3.set_title("Memory Usage (MB)")
    ax3.set_ylabel("Memory (MB)")
    ax3.set_xticklabels(index_types, rotation=45)

    plt.tight_layout()
    plt.savefig("range_search_comparison.png")
    print("\nPerformance comparison chart saved as 'range_search_comparison.png'")
    plt.close()


if __name__ == "__main__":
    print("FAISSx Range Search Example")
    print("===========================")
    print(f"Testing range search with radius={radius} across different index types")

    # Run tests
    try:
        results["FlatL2"] = test_flat_index()
        results["IVF"] = test_ivf_index()
        results["HNSW"] = test_hnsw_index()
        results["PQ"] = test_pq_index()

        # Plot results if matplotlib is available
        try:
            plot_results(results)
        except Exception as e:
            print(f"Could not generate plots: {e}")

        print("\nRange Search Results Summary:")
        print("============================")
        for idx_type, res in results.items():
            print(f"{idx_type}:")
            print(f"  Range search time: {res.get('range_search_time', 0):.4f}s")
            print(
                f"  Avg results per query: {np.mean(res.get('results_per_query', [0])):.1f}"
            )
            print(f"  Memory usage: {res.get('memory', 0):.2f} MB")

    except Exception as e:
        print(f"Error during tests: {e}")

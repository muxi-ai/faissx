#!/usr/bin/env python3
"""
FAISSx IndexHNSWFlat and IndexPQ Example

This example demonstrates how to use the advanced index types in FAISSx:
1. IndexHNSWFlat - Hierarchical Navigable Small World graphs for fast, approximate search
2. IndexPQ - Product Quantization for memory-efficient vector storage

The example shows:
- Creating different index types
- Training requirements for different indices
- Search accuracy and performance trade-offs
- Memory usage comparison
"""

import numpy as np
import time
import faissx
import sys
import psutil
import os

# Configure to use remote server (optional)
# faissx.configure(url="tcp://localhost:45678")

# Generate random data for testing
dimension = 128
num_vectors = 10000
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# Create some query vectors
num_queries = 10
query_vectors = np.random.random((num_queries, dimension)).astype('float32')


def process_memory():
    """Get the memory usage of the current process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def test_flat_index():
    """Test with a flat index (baseline)"""
    print("\n----- Testing Flat L2 Index (Baseline) -----")

    # Record memory before
    mem_before = process_memory()

    # Create a flat L2 index
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

    # Search for nearest neighbors
    start_time = time.time()
    k = 5  # number of nearest neighbors
    distances, indices = flat_index.search(query_vectors, k)
    search_time = time.time() - start_time

    print(f"Search time: {search_time:.4f} seconds")
    print(f"First result distances: {distances[0]}")

    return {
        "add_time": add_time,
        "search_time": search_time,
        "memory": mem_usage,
        "distances": distances
    }


def test_hnsw_index():
    """Test with an HNSW index"""
    print("\n----- Testing HNSW Index -----")

    # Record memory before
    mem_before = process_memory()

    # Create HNSW index
    M = 32  # Number of connections per node
    hnsw_index = faissx.IndexHNSWFlat(dimension, M)

    # HNSW doesn't require training
    print("HNSW doesn't require training")

    # Add vectors
    start_time = time.time()
    hnsw_index.add(vectors)
    add_time = time.time() - start_time
    print(f"Add time: {add_time:.4f} seconds")

    # Record memory after adding vectors
    mem_after = process_memory()
    mem_usage = mem_after - mem_before
    print(f"Approximate memory usage: {mem_usage:.2f} MB")

    # Search for nearest neighbors
    start_time = time.time()
    k = 5  # number of nearest neighbors
    distances, indices = hnsw_index.search(query_vectors, k)
    search_time = time.time() - start_time

    print(f"Search time: {search_time:.4f} seconds")
    print(f"First result distances: {distances[0]}")

    return {
        "add_time": add_time,
        "search_time": search_time,
        "memory": mem_usage,
        "distances": distances
    }


def test_pq_index():
    """Test with a Product Quantization index"""
    print("\n----- Testing PQ Index -----")

    # Record memory before
    mem_before = process_memory()

    # Create PQ index
    # We need dimension to be divisible by M (number of subquantizers)
    M = 16  # Number of subquantizers
    pq_index = faissx.IndexPQ(dimension, M, 8)  # 8 bits per subquantizer

    # Train the index (required for PQ)
    print("Training PQ index...")
    start_time = time.time()
    train_size = min(num_vectors, 5000)
    train_vectors = vectors[:train_size]
    pq_index.train(train_vectors)
    train_time = time.time() - start_time
    print(f"Train time: {train_time:.4f} seconds")

    # Add vectors
    start_time = time.time()
    pq_index.add(vectors)
    add_time = time.time() - start_time
    print(f"Add time: {add_time:.4f} seconds")

    # Record memory after adding vectors
    mem_after = process_memory()
    mem_usage = mem_after - mem_before
    print(f"Approximate memory usage: {mem_usage:.2f} MB")

    # Search for nearest neighbors
    start_time = time.time()
    k = 5  # number of nearest neighbors
    distances, indices = pq_index.search(query_vectors, k)
    search_time = time.time() - start_time

    print(f"Search time: {search_time:.4f} seconds")
    print(f"First result distances: {distances[0]}")

    return {
        "train_time": train_time,
        "add_time": add_time,
        "search_time": search_time,
        "memory": mem_usage,
        "distances": distances
    }


def print_comparison(flat_results, hnsw_results, pq_results):
    """Print a comparison of the different index types"""
    print("\n----- Performance Comparison -----")
    print(f"{'Metric':<18} {'Flat':<15} {'HNSW':<15} {'PQ':<15}")
    print("-" * 63)

    # Add time comparison
    print(f"{'Add time (s)':<18} {flat_results['add_time']:<15.4f} "
          f"{hnsw_results['add_time']:<15.4f} {pq_results['add_time']:<15.4f}")

    # Search time comparison
    print(f"{'Search time (s)':<18} {flat_results['search_time']:<15.4f} "
          f"{hnsw_results['search_time']:<15.4f} {pq_results['search_time']:<15.4f}")

    # Memory usage comparison
    print(f"{'Memory usage (MB)':<18} {flat_results['memory']:<15.2f} "
          f"{hnsw_results['memory']:<15.2f} {pq_results['memory']:<15.2f}")

    # HNSW speedup
    hnsw_speedup = flat_results['search_time'] / hnsw_results['search_time']
    print(f"\nHNSW search is {hnsw_speedup:.2f}x faster than flat search")

    # PQ memory savings
    pq_memory_ratio = flat_results['memory'] / pq_results['memory']
    print(f"PQ uses {pq_memory_ratio:.2f}x less memory than flat index")

    print("\nIndex Characteristics:")
    print("- Flat: Exact search, highest accuracy, slowest for large datasets")
    print("- HNSW: Fast approximate search, good accuracy, higher memory usage")
    print("- PQ: Low memory usage, faster than flat but less accurate")


if __name__ == "__main__":
    # Check if psutil is installed
    try:
        # psutil already imported at the top
        pass
    except ImportError:
        print("Please install psutil to run this example: pip install psutil")
        sys.exit(1)

    print(f"Testing with {num_vectors} vectors of dimension {dimension}")

    flat_results = test_flat_index()
    hnsw_results = test_hnsw_index()
    pq_results = test_pq_index()

    print_comparison(flat_results, hnsw_results, pq_results)

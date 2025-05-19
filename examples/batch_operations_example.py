#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx batch operations example
# https://github.com/muxi-ai/faissx
#
# This example demonstrates the performance benefits of batched operations
# in FAISSx by comparing normal operations vs. batched operations.

import time
import numpy as np
import matplotlib.pyplot as plt
import faissx

# Configure for remote operation (would use your own server in production)
faissx.configure(server="tcp://localhost:45678")  # Default server

# Test parameters
DIMENSION = 128
NUM_VECTORS = 100000  # 100K vectors
BATCH_SIZES = [100, 1000, 5000, 10000, 20000]  # Different batch sizes to test
K = 10  # For search operations
RADIUS = 40.0  # For range search operations

# Results storage
results = {
    "add": {"normal": [], "batched": []},
    "search": {"normal": [], "batched": []},
    "range_search": {"normal": [], "batched": []},
}


def generate_data(dim=DIMENSION, count=NUM_VECTORS):
    """Generate random vectors"""
    print(f"Generating {count} random vectors (dimension={dim})...")
    np.random.seed(42)  # For reproducibility
    vectors = np.random.random((count, dim)).astype('float32')
    # Normalize vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    return vectors


def test_batch_adds(vectors, batch_sizes):
    """Test vector addition with different batch sizes"""
    print("\n=== Testing batch add operations ===")

    for batch_size in batch_sizes:
        print(f"\nTesting with batch_size={batch_size}")

        # Create a new index for each test
        index = faissx.IndexFlatL2(DIMENSION)
        client = faissx.client.get_client()

        # Test normal add (one API call per batch)
        vectors_to_add = vectors.copy()
        start_time = time.time()

        for i in range(0, len(vectors_to_add), batch_size):
            batch = vectors_to_add[i:i+batch_size]
            client.add_vectors(index.index_id, batch)

        normal_time = time.time() - start_time
        print(f"  Normal add time: {normal_time:.2f}s")
        results["add"]["normal"].append(normal_time)

        # Create a new index for batched test
        index = faissx.IndexFlatL2(DIMENSION)

        # Test batch add (using optimized batch operation)
        start_time = time.time()
        client.batch_add_vectors(index.index_id, vectors_to_add, batch_size=batch_size)
        batched_time = time.time() - start_time
        print(f"  Batched add time: {batched_time:.2f}s")
        results["add"]["batched"].append(batched_time)

        # Calculate speedup
        speedup = normal_time / batched_time if batched_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")


def test_batch_searches(vectors, query_vectors, batch_sizes):
    """Test search operations with different batch sizes"""
    print("\n=== Testing batch search operations ===")

    # Create and fill index once
    index = faissx.IndexFlatL2(DIMENSION)
    client = faissx.client.get_client()
    client.add_vectors(index.index_id, vectors)
    print(f"Added {len(vectors)} vectors to index")

    for batch_size in batch_sizes:
        print(f"\nTesting with batch_size={batch_size}")

        # Test normal search (one API call per batch)
        queries = query_vectors.copy()
        start_time = time.time()

        all_results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            result = client.search(index.index_id, batch, k=K)
            all_results.extend(result.get("results", []))

        normal_time = time.time() - start_time
        print(f"  Normal search time: {normal_time:.2f}s")
        results["search"]["normal"].append(normal_time)

        # Test batch search (using optimized batch operation)
        start_time = time.time()
        client.batch_search(
            index.index_id, queries, k=K, batch_size=batch_size
        )
        batched_time = time.time() - start_time
        print(f"  Batched search time: {batched_time:.2f}s")
        results["search"]["batched"].append(batched_time)

        # Calculate speedup
        speedup = normal_time / batched_time if batched_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")


def test_batch_range_searches(vectors, query_vectors, batch_sizes):
    """Test range search operations with different batch sizes"""
    print("\n=== Testing batch range search operations ===")

    # Create and fill index once
    index = faissx.IndexFlatL2(DIMENSION)
    client = faissx.client.get_client()
    client.add_vectors(index.index_id, vectors)
    print(f"Added {len(vectors)} vectors to index")

    for batch_size in batch_sizes:
        print(f"\nTesting with batch_size={batch_size}")

        # Test normal range search (one API call per batch)
        queries = query_vectors.copy()
        start_time = time.time()

        all_results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            result = client.range_search(index.index_id, batch, radius=RADIUS)
            all_results.extend(result.get("results", []))

        normal_time = time.time() - start_time
        print(f"  Normal range search time: {normal_time:.2f}s")
        results["range_search"]["normal"].append(normal_time)

        # Test batch range search (using optimized batch operation)
        start_time = time.time()
        client.batch_range_search(
            index.index_id, queries, radius=RADIUS, batch_size=batch_size
        )
        batched_time = time.time() - start_time
        print(f"  Batched range search time: {batched_time:.2f}s")
        results["range_search"]["batched"].append(batched_time)

        # Calculate speedup
        speedup = normal_time / batched_time if batched_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")


def plot_results(results, batch_sizes):
    """Plot the performance results"""
    plt.figure(figsize=(15, 10))

    # Create subplots for each operation type
    operations = ["add", "search", "range_search"]
    for i, op in enumerate(operations):
        plt.subplot(1, 3, i+1)

        # Plot normal vs batched times
        plt.plot(batch_sizes, results[op]["normal"], 'o-', label="Normal")
        plt.plot(batch_sizes, results[op]["batched"], 's-', label="Batched")

        # Calculate and show speedups
        speedups = [n/b if b > 0 else 0 for n, b in
                    zip(results[op]["normal"], results[op]["batched"])]

        # Add labels
        for j, (x, y, s) in enumerate(zip(batch_sizes, results[op]["batched"], speedups)):
            plt.annotate(f"{s:.2f}x", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center')

        plt.title(f"{op.replace('_', ' ').title()} Performance")
        plt.xlabel("Batch Size")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("batch_performance.png")
    print("\nPerformance chart saved as 'batch_performance.png'")


if __name__ == "__main__":
    print("FAISSx Batch Operations Performance Example")
    print("===========================================")

    # Generate test data
    vectors = generate_data(dim=DIMENSION, count=NUM_VECTORS)

    # Generate query vectors (fewer than database vectors)
    query_vectors = generate_data(dim=DIMENSION, count=1000)

    try:
        # Run performance tests
        test_batch_adds(vectors, BATCH_SIZES)
        test_batch_searches(vectors, query_vectors, BATCH_SIZES)
        test_batch_range_searches(vectors, query_vectors, BATCH_SIZES)

        # Plot results
        try:
            plot_results(results, BATCH_SIZES)
        except Exception as e:
            print(f"Could not generate plots: {e}")

        # Print summary
        print("\nPerformance Summary:")
        print("===================")

        for op in ["add", "search", "range_search"]:
            print(f"\n{op.replace('_', ' ').title()}:")
            for i, size in enumerate(BATCH_SIZES):
                normal = results[op]["normal"][i]
                batched = results[op]["batched"][i]
                speedup = normal / batched if batched > 0 else float('inf')
                print(f"  Batch size {size}: {speedup:.2f}x speedup")

    except Exception as e:
        print(f"Error during benchmark: {e}")

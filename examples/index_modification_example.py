#!/usr/bin/env python3
"""
FAISSx Index Modification Example

This example demonstrates how to use the index modification features to:
1. Merge multiple indices into a single index
2. Split an index into multiple smaller indices

These operations are useful for:
- Combining indices from different sources
- Distributing large indices across multiple servers
- Rebalancing vector distributions
- Creating specialized indices for different data subsets
"""

import numpy as np
from sklearn.datasets import make_blobs
import os
import tempfile

# Import FAISSx components
from faissx.client.indices import (
    IndexFlatL2,
    IndexIDMap,
    IndexIDMap2,
    merge_indices,
    split_index,
    write_index,
    read_index
)

# Set random seed for reproducibility
np.random.seed(42)


def generate_data(n_samples=500, n_features=64, n_clusters=5, cluster_std=0.5):
    """Generate synthetic data with cluster structure"""
    print(f"Generating {n_samples} vectors with {n_features} dimensions...")

    # Create clustered data
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=n_features,
        random_state=42,
        cluster_std=cluster_std
    )

    # Scale to unit norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms

    return X, y


def create_sample_indices(vector_dim=64):
    """Create several sample indices with different configurations"""
    # Generate different datasets
    data1, _ = generate_data(500, vector_dim, 5, 0.5)
    data2, _ = generate_data(300, vector_dim, 3, 0.8)
    data3, _ = generate_data(200, vector_dim, 2, 1.0)

    # Create a flat index
    index1 = IndexFlatL2(vector_dim)
    index1.add(data1)
    print(f"Created index1: Flat with {index1.ntotal} vectors")

    # Create an IDMap index
    index2 = IndexIDMap(IndexFlatL2(vector_dim))
    custom_ids = np.array([10000 + i for i in range(len(data2))])
    index2.add_with_ids(data2, custom_ids)
    print(f"Created index2: IndexIDMap with {index2.ntotal} vectors and custom IDs")

    # Create an IDMap2 index
    index3 = IndexIDMap2(IndexFlatL2(vector_dim))
    custom_ids = np.array([20000 + i for i in range(len(data3))])
    index3.add_with_ids(data3, custom_ids)
    print(f"Created index3: IndexIDMap2 with {index3.ntotal} vectors and custom IDs")

    return index1, index2, index3


def demonstrate_merging():
    """Demonstrate merging multiple indices"""
    print("\n=== Demonstrating Index Merging ===\n")

    # Create sample indices
    index1, index2, index3 = create_sample_indices()

    # Record original vector counts
    original_counts = [index1.ntotal, index2.ntotal, index3.ntotal]
    original_total = sum(original_counts)
    print(f"Original indices: {original_counts} vectors (total: {original_total})")

    # Basic merge (Flat indices)
    print("\n1. Merging flat indices...")
    merged = merge_indices([index1, index2, index3])
    print(f"  Result: {type(merged).__name__} with {merged.ntotal} vectors")

    # Check if the merged index has all vectors
    assert merged.ntotal == original_total, "Merged index does not have all vectors!"

    # Merge with IDMap output
    print("\n2. Merging with IDMap output...")
    merged_idmap = merge_indices([index1, index2, index3], id_map=True)
    print(f"  Result: {type(merged_idmap).__name__} with {merged_idmap.ntotal} vectors")

    # Merge with custom output type
    print("\n3. Merging to a different index type...")
    merged_ivf = merge_indices([index1, index2, index3], output_type="IVF100,Flat")
    print(f"  Result: {type(merged_ivf).__name__} with {merged_ivf.ntotal} vectors")

    # Verify merged results with a query
    query = np.random.random(index1.d).reshape(1, -1)
    query = query / np.linalg.norm(query)

    print("\nTest query results from merged index:")
    distances, indices = merged.search(query, 5)
    print(f"  Top 5 results: distances={distances[0][:5]}")

    # Return a merged index for split testing
    return merged_idmap


def demonstrate_splitting(index=None):
    """Demonstrate splitting an index into multiple parts"""
    print("\n=== Demonstrating Index Splitting ===\n")

    # Use provided index or create a sample index
    if index is None:
        # Generate data
        data, clusters = generate_data(1000, 64, 5, 0.5)

        # Create an index with vectors
        index = IndexIDMap(IndexFlatL2(data.shape[1]))
        custom_ids = np.array([1000 + i for i in range(len(data))])
        index.add_with_ids(data, custom_ids)

    original_count = index.ntotal
    print(f"Original index: {type(index).__name__} with {original_count} vectors")

    # 1. Simple sequential split
    print("\n1. Sequential splitting into 3 parts...")
    split_indices = split_index(index, num_parts=3, split_method='sequential')

    # Check the resulting parts
    split_counts = [idx.ntotal for idx in split_indices]
    split_total = sum(split_counts)
    print(f"  Result: {len(split_indices)} parts with {split_counts} vectors each")
    print(f"  Total vectors: {split_total} (original: {original_count})")

    # All vectors should be preserved
    assert split_total == original_count, "Split indices do not have all vectors!"

    # 2. Cluster-based split
    if hasattr(index, 'is_trained') and index.is_trained:
        print("\n2. Cluster-based splitting into 4 parts...")
        cluster_indices = split_index(index, num_parts=4, split_method='cluster')

        # Check the resulting parts
        cluster_counts = [idx.ntotal for idx in cluster_indices]
        cluster_total = sum(cluster_counts)
        print(f"  Result: {len(cluster_indices)} parts with {cluster_counts} vectors each")
        print(f"  Total vectors: {cluster_total} (original: {original_count})")

        # All vectors should be preserved
        assert cluster_total == original_count, "Cluster split indices do not have all vectors!"
    else:
        print("\n2. Skipping cluster-based split (requires trained index)")

    # 3. Custom split function
    print("\n3. Custom splitting into 2 parts based on vector norm...")

    # Define a custom split function
    def norm_based_split(vectors):
        """Split vectors based on their norm (>0.5 or <=0.5)"""
        norms = np.linalg.norm(vectors, axis=1)
        return [1 if norm > 0.5 else 0 for norm in norms]

    custom_indices = split_index(
        index,
        num_parts=2,
        split_method='custom',
        custom_split_fn=norm_based_split
    )

    # Check the resulting parts
    custom_counts = [idx.ntotal for idx in custom_indices]
    custom_total = sum(custom_counts)
    print(f"  Result: {len(custom_indices)} parts with {custom_counts} vectors each")
    print(f"  Total vectors: {custom_total} (original: {original_count})")

    # All vectors should be preserved
    assert custom_total == original_count, "Custom split indices do not have all vectors!"

    # Demonstrate persistence of split indices
    print("\n4. Saving and loading split indices...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the indices
        for i, idx in enumerate(custom_indices):
            file_path = os.path.join(temp_dir, f"part_{i}.index")
            write_index(idx, file_path)
            print(f"  Saved part {i} to {file_path}")

        # Load the first part and test
        file_path = os.path.join(temp_dir, "part_0.index")
        loaded_index = read_index(file_path)
        print(f"  Loaded {type(loaded_index).__name__} with {loaded_index.ntotal} vectors")

    print("\nSplit index testing complete!")


if __name__ == "__main__":
    # Demonstrate merging
    merged_index = demonstrate_merging()

    # Demonstrate splitting using the merged index
    demonstrate_splitting(merged_index)

    print("\n=== Index Modification Use Cases ===")
    print("1. Merging indices from different data sources")
    print("2. Combining differently structured indices into a unified format")
    print("3. Splitting large indices for distributed search")
    print("4. Re-balancing vectors based on similarity for optimized retrieval")
    print("5. Creating specialized indices for different query patterns")
    print("6. Partitioning data by custom business rules")
    print("7. Preserving custom IDs when reorganizing vector data")

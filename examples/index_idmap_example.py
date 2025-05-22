#!/usr/bin/env python3
"""
FAISSx IndexIDMap Example

This example demonstrates how to use IndexIDMap and IndexIDMap2 to associate custom IDs
with vectors in your FAISS indices. Using custom IDs allows you to map between your own
identifiers (like database IDs, UUIDs, or other application-specific identifiers) and vectors.

Key features demonstrated:
1. Creating an IndexIDMap wrapping another index
2. Adding vectors with custom IDs
3. Searching and getting results with custom IDs
4. Vector reconstruction by ID
5. Removing vectors by ID
6. Using IndexIDMap2 to update vectors while preserving IDs
"""

import numpy as np
from sklearn.datasets import make_blobs
import uuid

# Import FAISSx components
from faissx.client.indices import IndexFlatL2, IndexIDMap, IndexIDMap2

# Set random seed for reproducibility
np.random.seed(42)


def generate_data(n_samples=1000, n_features=128, n_clusters=10):
    """Generate synthetic data with cluster structure"""
    print(f"Generating {n_samples} vectors with {n_features} dimensions...")

    # Create clustered data
    X, y = make_blobs(
        n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42
    )

    # Scale to unit norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms

    print(f"Data shape: {X.shape}")
    return X, y


def create_custom_ids(n, id_type="int"):
    """Create custom IDs for the vectors"""
    if id_type == "int":
        # Use non-sequential integer IDs (real-world database IDs are rarely sequential)
        return np.array([10000 + i * 5 for i in range(n)], dtype=np.int64)
    elif id_type == "uuid":
        # Use UUIDs (string identifiers)
        return np.array([str(uuid.uuid4()) for _ in range(n)])
    else:
        raise ValueError(f"Unsupported ID type: {id_type}")


def demonstrate_indexidmap():
    """Demonstrate IndexIDMap functionality"""
    # Generate sample data
    data, labels = generate_data(n_samples=1000, n_features=128, n_clusters=10)

    # Create custom IDs (non-sequential integer IDs for this example)
    custom_ids = create_custom_ids(len(data), id_type="int")

    print("\n===== Creating IndexIDMap =====")
    # First, create the base index (any FAISS index type can be used)
    base_index = IndexFlatL2(data.shape[1])

    # Then create the ID map wrapper
    index = IndexIDMap(base_index)

    print(f"Created IndexIDMap wrapping {type(base_index).__name__}")

    # Add vectors with custom IDs
    print(f"\n===== Adding {len(data)} vectors with custom IDs =====")
    print(f"Sample IDs: {custom_ids[:5]} ...")

    index.add_with_ids(data, custom_ids)
    print(f"Added {index.ntotal} vectors to the index")

    # Select a query vector
    query_idx = 42
    query_vector = data[query_idx:query_idx+1]  # Keep 2D shape

    # Search for similar vectors
    print("\n===== Searching with IndexIDMap =====")
    k = 5  # Number of results to return
    distances, result_ids = index.search(query_vector, k)

    print(f"Query vector ID: {custom_ids[query_idx]}")
    print("Search results (showing custom IDs):")
    for i in range(k):
        print(f"  Result #{i+1}: ID={result_ids[0, i]}, distance={distances[0, i]:.6f}")

    # Demonstrate vector reconstruction
    print("\n===== Vector Reconstruction by ID =====")
    # Get ID of a vector to reconstruct
    target_id = custom_ids[100]

    # Reconstruct the vector from its ID
    reconstructed = index.reconstruct(target_id)
    original = data[100]

    # Compute the reconstruction error
    error = np.linalg.norm(reconstructed - original)
    print(f"Reconstructed vector for ID {target_id}")
    print(f"Reconstruction error: {error:.8f}")

    # Demonstrate vector removal
    print("\n===== Removing Vectors by ID =====")
    # Select some IDs to remove
    ids_to_remove = custom_ids[200:205]
    print(f"Removing 5 vectors with IDs: {ids_to_remove}")

    # Count before removal
    count_before = index.ntotal

    # Remove the vectors
    index.remove_ids(ids_to_remove)

    # Count after removal
    count_after = index.ntotal
    print(f"Vectors before removal: {count_before}")
    print(f"Vectors after removal: {count_after}")

    # Verify the removal by trying to search for a removed vector
    query_vector = data[200:201]  # This vector was removed
    distances, result_ids = index.search(query_vector, 1)

    print(f"Search for removed vector returns ID: {result_ids[0, 0]}")
    print("Note: -1 indicates no match found or the vector was removed")


def demonstrate_indexidmap2():
    """Demonstrate IndexIDMap2 functionality for updating vectors"""
    print("\n\n============================================================")
    print("          DEMONSTRATING IndexIDMap2 FUNCTIONALITY")
    print("============================================================")

    # Generate sample data
    data, labels = generate_data(n_samples=500, n_features=64, n_clusters=10)

    # Create custom IDs
    custom_ids = create_custom_ids(len(data), id_type="int")

    print("\n===== Creating IndexIDMap2 =====")
    # First, create the base index
    base_index = IndexFlatL2(data.shape[1])

    # Then create the ID map wrapper
    index = IndexIDMap2(base_index)

    print(f"Created IndexIDMap2 wrapping {type(base_index).__name__}")

    # Add vectors with custom IDs
    print(f"\n===== Adding {len(data)} vectors with custom IDs =====")
    index.add_with_ids(data, custom_ids)
    print(f"Added {index.ntotal} vectors to the index")

    # Select a vector to replace
    target_id = custom_ids[50]
    original_vector = data[50]

    # Create a modified version of the vector
    modified_vector = original_vector.copy()
    # Move the vector in a different direction (normalized)
    modified_vector += np.random.normal(0, 0.5, modified_vector.shape)
    modified_vector /= np.linalg.norm(modified_vector)

    print("\n===== Replacing a Vector While Preserving ID =====")
    print(f"Target ID: {target_id}")

    # Before replacement: search for nearest neighbors to the original vector
    query = original_vector.reshape(1, -1)
    before_distances, before_ids = index.search(query, 5)

    print("\nSearch results BEFORE replacement:")
    for i in range(5):
        print(f"  Result #{i+1}: ID={before_ids[0, i]}, distance={before_distances[0, i]:.6f}")

    # Replace the vector
    print("\nReplacing vector...")
    index.replace_vector(target_id, modified_vector)

    # After replacement: search again
    after_distances, after_ids = index.search(query, 5)

    print("\nSearch results AFTER replacement:")
    for i in range(5):
        print(f"  Result #{i+1}: ID={after_ids[0, i]}, distance={after_distances[0, i]:.6f}")

    # Verify the replacement by reconstructing
    reconstructed = index.reconstruct(target_id)
    original_error = np.linalg.norm(reconstructed - original_vector)
    modified_error = np.linalg.norm(reconstructed - modified_vector)

    print("\nVerifying replacement through reconstruction:")
    print(f"  Error vs original vector: {original_error:.6f}")
    print(f"  Error vs modified vector: {modified_error:.6f}")
    print("  The smaller error with the modified vector confirms replacement succeeded")

    # Demonstrate batch updates
    print("\n===== Batch Updating Multiple Vectors =====")
    update_ids = custom_ids[100:105]
    update_vectors = np.random.randn(5, 64)  # New random vectors
    update_vectors = update_vectors / np.linalg.norm(update_vectors, axis=1, keepdims=True)

    print(f"Updating 5 vectors with IDs: {update_ids}")
    index.update_vectors(update_ids, update_vectors)

    # Verify batch update
    sample_id = update_ids[2]
    expected_vector = update_vectors[2]
    actual_vector = index.reconstruct(sample_id)
    update_error = np.linalg.norm(actual_vector - expected_vector)

    print(f"\nVerifying batch update for ID {sample_id}:")
    print(f"  Reconstruction error: {update_error:.8f}")
    print("  Small error indicates successful batch update")


if __name__ == "__main__":
    demonstrate_indexidmap()
    demonstrate_indexidmap2()

    print("\n===== IndexIDMap/IndexIDMap2 Use Cases =====")
    print("1. Maintaining custom IDs from an external database")
    print("2. Using non-integer IDs (UUIDs, hash values, etc.)")
    print("3. Maintaining stable IDs when the underlying vectors change")
    print("4. Selectively removing vectors without rebuilding the entire index")
    print("5. Mapping between application-level entities and vector representations")
    print("6. Updating vectors while preserving IDs (IndexIDMap2)")
    print("7. Batch updating multiple vectors at once (IndexIDMap2)")
    print("8. Syncing vector database with changing source data (IndexIDMap2)")

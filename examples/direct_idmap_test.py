#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct test for IndexIDMap remote mode functionality.

This script directly tests the remote mode operations for IndexIDMap,
including:
- Creating an IndexIDMap with a base index
- Adding vectors with IDs
- Searching for vectors
- Removing vectors by ID
- Reconstructing vectors by ID

Run this script after starting the FAISSx server.
"""

import numpy as np
from faissx import client as faiss
import sys

# Set higher log level for more output
import logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run IDMap remote mode tests."""
    print("\n=== Testing IndexIDMap Remote Mode ===\n")

    # Configure remote server connection
    try:
        faiss.configure(server="tcp://localhost:45678")
        print("Connected to server.")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        print("Make sure the server is running with: faissx.server run")
        return 1

    # Parameters
    dimension = 64
    num_vectors = 100

    print("\n1. Creating base flat index...")
    base_index = faiss.IndexFlatL2(dimension)

    print("\n2. Creating IDMap wrapper...")
    idmap = faiss.IndexIDMap(base_index)

    print("\n3. Generating random vectors and IDs...")
    vectors = np.random.random((num_vectors, dimension)).astype('float32')
    # Use non-consecutive IDs to test ID mapping
    ids = np.array([i * 2 + 10 for i in range(num_vectors)], dtype=np.int64)

    print("\n4. Adding vectors with IDs...")
    try:
        idmap.add_with_ids(vectors, ids)
        print(f"Successfully added {num_vectors} vectors with IDs.")
        print(f"Index now contains {idmap.ntotal} vectors.")
    except Exception as e:
        print(f"Failed to add vectors: {e}")
        return 1

    print("\n5. Testing search functionality...")
    query = np.random.random((1, dimension)).astype('float32')
    try:
        distances, found_ids = idmap.search(query, k=5)
        print("Search results:")
        for i in range(5):
            print(f"  ID: {found_ids[0, i]}, Distance: {distances[0, i]:.4f}")
    except Exception as e:
        print(f"Search failed: {e}")

    print("\n6. Testing reconstruction...")
    try:
        # Get the first ID from our ID list
        id_to_reconstruct = ids[0]
        vector = idmap.reconstruct(id_to_reconstruct)
        print(f"Reconstructed vector for ID {id_to_reconstruct}, shape: {vector.shape}")
        # Verify it matches the original
        mse = np.mean((vector - vectors[0]) ** 2)
        print(f"Mean squared error from original: {mse:.8f}")
    except Exception as e:
        print(f"Reconstruction failed: {e}")

    print("\n7. Testing remove_ids...")
    try:
        # Remove the first 10 vectors
        ids_to_remove = ids[:10]
        idmap.remove_ids(ids_to_remove)
        print(f"Removed {len(ids_to_remove)} vectors.")
        print(f"Index now contains {idmap.ntotal} vectors.")
    except Exception as e:
        print(f"Failed to remove vectors: {e}")

    print("\n8. Creating IndexIDMap2...")
    try:
        idmap2 = faiss.IndexIDMap2(faiss.IndexFlatL2(dimension))
        idmap2.add_with_ids(vectors, ids)
        print(f"Successfully created IndexIDMap2 with {idmap2.ntotal} vectors.")

        # Update a vector
        new_vector = np.random.random((1, dimension)).astype('float32')
        id_to_update = ids[20]
        idmap2.replace_vector(id_to_update, new_vector)
        print(f"Updated vector with ID {id_to_update}")

        # Verify the update
        updated = idmap2.reconstruct(id_to_update)
        mse = np.mean((updated - new_vector.reshape(-1)) ** 2)
        print(f"Mean squared error after update: {mse:.8f}")
    except Exception as e:
        print(f"IndexIDMap2 test failed: {e}")

    print("\nTest completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

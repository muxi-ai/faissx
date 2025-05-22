#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISS Remote Server Test
This test explicitly checks remote server operations
"""

import numpy as np
import logging
import sys
import time
import uuid

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Import client
from faissx import client as faiss

# Configure remote server
print("\n=== Configuring Remote Server ===")
faiss.configure(server="tcp://localhost:45678")

# Test basic flat index
def test_flat_index():
    print("\n=== Testing Flat Index ===")
    dimension = 8

    # Create index
    index = faiss.IndexFlatL2(dimension)
    print(f"Created IndexFlatL2 with dimension {dimension}")

    # Add vectors
    n_vectors = 100
    vectors = np.random.random((n_vectors, dimension)).astype('float32')
    index.add(vectors)
    print(f"Added {n_vectors} vectors, ntotal = {index.ntotal}")

    # Search
    n_queries = 5
    k = 10
    queries = np.random.random((n_queries, dimension)).astype('float32')
    distances, indices = index.search(queries, k=k)

    print(f"Search returned {indices.shape} indices and {distances.shape} distances")
    print(f"First query top result: index={indices[0][0]}, distance={distances[0][0]:.4f}")

    return True

# Test IDMap index
def test_idmap_index():
    print("\n=== Testing IDMap Index ===")
    dimension = 8

    # Create base index
    base_index = faiss.IndexFlatL2(dimension)
    print(f"Created base IndexFlatL2 with dimension {dimension}")

    # Create IDMap
    idmap = faiss.IndexIDMap(base_index)
    print(f"Created IndexIDMap wrapping the base index")

    # Add vectors with IDs
    n_vectors = 50
    vectors = np.random.random((n_vectors, dimension)).astype('float32')
    ids = np.arange(1000, 1000 + n_vectors).astype('int64')
    idmap.add_with_ids(vectors, ids)
    print(f"Added {n_vectors} vectors with IDs {ids[0]}...{ids[-1]}, ntotal = {idmap.ntotal}")

    # Search
    n_queries = 3
    k = 5
    queries = np.random.random((n_queries, dimension)).astype('float32')
    distances, result_ids = idmap.search(queries, k=k)

    print(f"Search returned {result_ids.shape} IDs and {distances.shape} distances")
    print(f"First query top result: ID={result_ids[0][0]}, distance={distances[0][0]:.4f}")

    return True

# Test IDMap2 index
def test_idmap2_index():
    print("\n=== Testing IDMap2 Index ===")
    dimension = 8

    # Create base index
    base_index = faiss.IndexFlatL2(dimension)
    print(f"Created base IndexFlatL2 with dimension {dimension}")

    # Create IDMap2
    idmap2 = faiss.IndexIDMap2(base_index)
    print(f"Created IndexIDMap2 wrapping the base index")

    # Add vectors with IDs
    n_vectors = 50
    vectors = np.random.random((n_vectors, dimension)).astype('float32')
    ids = np.arange(2000, 2000 + n_vectors).astype('int64')
    idmap2.add_with_ids(vectors, ids)
    print(f"Added {n_vectors} vectors with IDs {ids[0]}...{ids[-1]}, ntotal = {idmap2.ntotal}")

    # Search
    n_queries = 3
    k = 5
    queries = np.random.random((n_queries, dimension)).astype('float32')
    distances, result_ids = idmap2.search(queries, k=k)

    print(f"Search returned {result_ids.shape} IDs and {distances.shape} distances")
    print(f"First query top result: ID={result_ids[0][0]}, distance={distances[0][0]:.4f}")

    # Update some vectors
    update_count = 5
    update_ids = ids[:update_count]
    update_vectors = np.random.random((update_count, dimension)).astype('float32')
    idmap2.add_with_ids(update_vectors, update_ids)
    print(f"Updated {update_count} vectors with IDs {update_ids[0]}...{update_ids[-1]}")

    # Search again
    distances, result_ids = idmap2.search(queries, k=k)
    print(f"Search after updates: top result ID={result_ids[0][0]}, distance={distances[0][0]:.4f}")

    return True

# Test HNSW index
def test_hnsw_index():
    print("\n=== Testing HNSW Index ===")
    dimension = 8

    try:
        # Create HNSW index with unique name
        hnsw_index_name = f"index-hnsw-test-{uuid.uuid4().hex[:8]}"
        print(f"Creating HNSW index with name '{hnsw_index_name}'...")

        client = faiss.get_client()
        if client:
            # Use direct API to create index with 'HNSW' type
            response = client.create_index(
                name=hnsw_index_name,
                dimension=dimension,
                index_type="HNSW"
            )
            print(f"Create index response: {response}")

            # Handle both dict and string responses
            if isinstance(response, dict):
                index_id = response.get("index_id", "index-hnsw-test")
            else:
                # If response is a string, use it directly
                index_id = str(response)

            print(f"Using index ID: {index_id}")
        else:
            # Fallback to local mode
            index = faiss.IndexHNSWFlat(dimension, 32)
            index_id = None

        # If server mode and index created
        if client and index_id:
            # Add vectors
            n_vectors = 100
            vectors = np.random.random((n_vectors, dimension)).astype('float32')
            response = client.add_vectors(index_id, vectors.tolist())
            print(f"Added vectors response: {response}")

            # Search
            n_queries = 5
            k = 10
            queries = np.random.random((n_queries, dimension)).astype('float32')
            search_response = client.search(index_id, queries.tolist(), k)
            print(f"Search response: {search_response.get('success', False)}")

            if search_response.get("success", False):
                results = search_response.get("results", [])
                if results:
                    first_result = results[0]
                    distances = first_result.get("distances", [])
                    indices = first_result.get("indices", [])
                    if distances and indices:
                        print(f"First query top result: index={indices[0]}, distance={distances[0]:.4f}")
                        return True

            return False
        else:
            # Local mode or fallback
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
            print(f"Created IndexHNSWFlat with dimension {dimension}")

            # Add vectors
            n_vectors = 100
            vectors = np.random.random((n_vectors, dimension)).astype('float32')
            index.add(vectors)
            print(f"Added {n_vectors} vectors, ntotal = {index.ntotal}")

            # Search
            n_queries = 5
            k = 10
            queries = np.random.random((n_queries, dimension)).astype('float32')
            distances, indices = index.search(queries, k=k)

            print(f"Search returned {indices.shape} indices and {distances.shape} distances")
            print(f"First query top result: index={indices[0][0]}, distance={distances[0][0]:.4f}")

            return True
    except Exception as e:
        print(f"HNSW test failed: {e}")
        return False

# Run all tests
def run_all_tests():
    tests = [
        ("Flat Index", test_flat_index),
        ("IDMap Index", test_idmap_index),
        ("IDMap2 Index", test_idmap2_index),
        ("HNSW Index", test_hnsw_index),
    ]

    results = {}

    for name, test_func in tests:
        print(f"\n\n{'=' * 50}")
        print(f"Running test: {name}")
        print(f"{'=' * 50}")

        try:
            start_time = time.time()
            success = test_func()
            end_time = time.time()

            if success:
                result = "PASSED"
            else:
                result = "FAILED"

            results[name] = result
            print(f"\n{name}: {result} in {end_time - start_time:.2f} seconds")
        except Exception as e:
            results[name] = "ERROR"
            print(f"\n{name}: ERROR - {str(e)}")

    # Print summary
    print("\n\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)

    passed = 0
    for name, result in results.items():
        print(f"{name}: {result}")
        if result == "PASSED":
            passed += 1

    print(f"\n{passed}/{len(tests)} tests passed")

    return passed == len(tests)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

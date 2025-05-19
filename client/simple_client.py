#!/usr/bin/env python3
"""
Simple FAISSx Client Example

This script demonstrates how to use the FAISSx client to interact with the server.
"""

import os
import numpy as np
import time
import sys

# Add parent directory to path to import faissx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from faissx import client as faiss

def main():
    # Configure the client to connect to the server
    server = os.environ.get("FAISSX_SERVER", "tcp://localhost:45678")
    print(f"Connecting to FAISSx server at {server}")
    faiss.configure(server=server)

    # Create a new index
    dimension = 128
    print(f"Creating FAISS index with dimension {dimension}")
    index = faiss.IndexFlatL2(dimension)

    # Generate random vectors
    vector_count = 1000
    print(f"Generating {vector_count} random vectors")
    vectors = np.random.random((vector_count, dimension)).astype(np.float32)

    # Add vectors to the index
    print("Adding vectors to the index")
    start_time = time.time()
    index.add(vectors)
    add_time = time.time() - start_time
    print(f"Added {vector_count} vectors in {add_time:.4f} seconds")
    print(f"Index contains {index.ntotal} vectors")

    # Search for similar vectors
    query_count = 10
    k = 5
    print(f"Searching for {query_count} vectors, k={k}")
    query_vectors = np.random.random((query_count, dimension)).astype(np.float32)

    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - start_time

    print(f"Search completed in {search_time:.4f} seconds")
    print(f"Average search time: {(search_time / query_count) * 1000:.2f} ms per query")

    # Show sample results
    print("\nSample search results:")
    for i in range(min(3, query_count)):
        print(f"Query {i}:")
        for j in range(k):
            print(f"  Result {j}: Distance={distances[i][j]:.4f}, Index={indices[i][j]}")

    print("\nDone!")

if __name__ == "__main__":
    main()

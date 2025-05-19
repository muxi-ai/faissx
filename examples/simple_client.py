#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/faissx
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FAISSx Simple Client Example

This example demonstrates using the FAISSx client library to:
1. Configure the client to connect to a FAISSx server
2. Create a vector index with the specified dimension
3. Add random vectors to the index
4. Search for nearest neighbors to a query vector
5. Display the search results

The client behaves as a drop-in replacement for FAISS with the same API,
but operations are executed on the remote FAISSx server.
"""

import numpy as np
import time
import logging
import sys

# Import FAISSx client as 'faiss' for drop-in compatibility
from faissx import client as faiss

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration parameters
SERVER_ADDRESS = "tcp://localhost:45678"
API_KEY = "test-key-1"
TENANT_ID = "tenant-1"
DIMENSION = 128
NUM_VECTORS = 1000
K = 10  # Number of nearest neighbors to return

def main():
    """Run the FAISSx client example."""
    print(f"FAISSx Client Example - connecting to {SERVER_ADDRESS}")

    # Configure the client to connect to the FAISSx server
    try:
        faiss.configure(
            server=SERVER_ADDRESS,
            api_key=API_KEY,
            tenant_id=TENANT_ID
        )
        print("Client configured successfully")
    except Exception as e:
        print(f"Error configuring client: {e}")
        sys.exit(1)

    # Create a FAISS index (same API as standard FAISS)
    index = faiss.IndexFlatL2(DIMENSION)
    print(f"Created IndexFlatL2 with dimension {DIMENSION}")

    # Generate random vectors
    print(f"Generating {NUM_VECTORS} random vectors...")
    vectors = np.random.random((NUM_VECTORS, DIMENSION)).astype('float32')

    # Add vectors to the index
    print("Adding vectors to index...")
    start = time.time()
    index.add(vectors)
    add_time = time.time() - start
    print(f"Added {NUM_VECTORS} vectors in {add_time:.2f} seconds")

    # Generate a random query vector
    query = np.random.random((1, DIMENSION)).astype('float32')

    # Search for similar vectors
    print(f"Searching for {K} nearest neighbors...")
    start = time.time()
    distances, indices = index.search(query, K)
    search_time = time.time() - start
    print(f"Search completed in {search_time:.4f} seconds")

    # Display results
    print("\nSearch Results:")
    print("-" * 50)
    print("Query vector:", query[0][:5], "...")  # Show first 5 elements
    print("-" * 50)
    print("  Index  |  Distance")
    print("-" * 50)

    for i in range(len(indices[0])):
        if indices[0][i] >= 0:  # Valid index
            print(f"  {indices[0][i]:5d}  |  {distances[0][i]:.6f}")
        else:
            print(f"  {'N/A':5s}  |  {distances[0][i]:.6f}")

    print("\nNote: This example assumes you have a FAISSx server running at", SERVER_ADDRESS)
    print("with authentication enabled and the specified API key/tenant configured.")

if __name__ == "__main__":
    main()

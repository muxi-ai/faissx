#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple test for IndexIDMap2 in remote mode.
"""

import numpy as np
from faissx import client as faiss

# Configure to use remote server
faiss.configure(server="tcp://localhost:45678")

# Create a simple test
def test_idmap2_simple():
    # Create a base index
    dimension = 64
    base_index = faiss.IndexFlatL2(dimension)

    # Verify base index works
    print("Base index created successfully:", base_index.d)

    try:
        # Create the IDMap2 index
        print("Creating IndexIDMap2...")
        idmap2 = faiss.IndexIDMap2(base_index)
        print("IndexIDMap2 created successfully:", idmap2.d)

        # Generate some test data
        xb = np.random.random((10, dimension)).astype('float32')
        ids = np.arange(1000, 1010).astype('int64')

        # Add vectors
        print("Adding vectors...")
        idmap2.add_with_ids(xb, ids)
        print("Added vectors:", idmap2.ntotal)

        # Search
        print("Searching...")
        xq = np.random.random((1, dimension)).astype('float32')
        distances, indices = idmap2.search(xq, k=5)
        print("Search result shapes:", distances.shape, indices.shape)

        print("Success!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_idmap2_simple()

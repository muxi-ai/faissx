#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Debug connection to FAISSx server
#
# Copyright (C) 2025 Ran Aroussi

"""
Debug script to test connection to remote FAISSx server.
"""

import os
import sys
import numpy as np
import faissx
from faissx.client.client import FaissXClient

# Add parent directory to path to import faissx
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(current_dir), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def test_connection():
    """Test connection to FAISSx server."""
    print("Testing connection to FAISSx server")

    # Configure to use remote server
    print("Configuring remote server at tcp://0.0.0.0:45678")
    faissx.configure(
        url="tcp://0.0.0.0:45678",
        tenant_id=None  # No tenant ID for unauthenticated mode
    )

    # Create a client directly
    print("Importing client...")
    client = FaissXClient(server="tcp://0.0.0.0:45678")

    print(f"Created client: {client}")

    # Check internal state
    print(f"API URL: {faissx._API_URL}")

    # Try to create and use an index
    try:
        print("Creating index...")
        d = 64  # dimension
        index = faissx.client.IndexFlatL2(d)
        print(f"Created index: {index}")

        # Add some vectors
        n = 10  # number of vectors
        vectors = np.random.random((n, d)).astype('float32')
        print(f"Adding {n} vectors...")
        index.add(vectors)
        print(f"Added vectors, total: {index.ntotal}")

        # Search
        query = np.random.random((1, d)).astype('float32')
        print("Searching...")
        distances, indices = index.search(query, k=5)
        print(f"Search results: {distances.shape}, {indices.shape}")

        print("Remote connection test successful!")
    except Exception as e:
        print(f"Error testing connection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_connection()

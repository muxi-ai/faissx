#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test IVFFlat with forced remote mode
#
# Copyright (C) 2025 Ran Aroussi

"""
Test that IVFFlat works or fails appropriately when remote mode is forced.

This script demonstrates how IVFFlat behaves when the client is configured
to use remote mode exclusively with no fallback to local mode.
"""

import os
import sys
import numpy as np

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(current_dir), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add current directory to path for local imports
sys.path.insert(0, current_dir)
from force_remote import force_remote_mode

# Now import faissx
import faissx


def main():
    """Run the test for IVFFlat with forced remote mode."""
    print("Testing IVFFlat with forced remote mode...")

    # Apply the monkeypatch to force remote mode
    force_remote_mode(server_url="tcp://0.0.0.0:45678")

    # Test case 1: Create a flat index - should work with remote server
    try:
        print("\nTest 1: Creating a flat index (should succeed)...")
        d = 32  # dimension
        flat_index = faissx.client.IndexFlatL2(d)
        print(f"  Success: Created flat index with dimension {d}")

        # Add some vectors to verify it works
        vectors = np.random.random((5, d)).astype('float32')
        flat_index.add(vectors)
        print(f"  Success: Added {len(vectors)} vectors, total: {flat_index.ntotal}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test case 2: Create an IVFFlat index - should fail or work, but not fall back to local
    try:
        print("\nTest 2: Creating an IVFFlat index...")
        d = 32  # dimension
        nlist = 4  # number of clusters

        # Create a quantizer
        quantizer = faissx.client.IndexFlatL2(d)

        # Create an IVF index
        ivf_index = faissx.client.IndexIVFFlat(quantizer, d, nlist)
        print("  Success: Created IVFFlat index")

        # If we got here, try training the index
        training_vectors = np.random.random((100, d)).astype('float32')
        ivf_index.train(training_vectors)
        print(f"  Success: Trained IVFFlat index with {len(training_vectors)} vectors")

        # Add vectors
        vectors = np.random.random((20, d)).astype('float32')
        ivf_index.add(vectors)
        print(f"  Success: Added {len(vectors)} vectors, total: {ivf_index.ntotal}")

        # Search
        query = np.random.random((1, d)).astype('float32')
        distances, indices = ivf_index.search(query, k=5)
        print(f"  Success: Search returned {len(indices[0])} results")
    except Exception as e:
        print(f"  Expected error: {e}")

    print("\nTest complete!")


if __name__ == "__main__":
    main()

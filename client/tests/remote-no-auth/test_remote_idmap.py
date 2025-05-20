#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test IndexIDMap with forced remote mode
#
# Copyright (C) 2025 Ran Aroussi

"""
Test that IndexIDMap raises errors when remote mode is forced.

This script demonstrates how IndexIDMap behaves when the client is configured
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

# Import the monkeypatching function
sys.path.insert(0, current_dir)  # Add current directory to path
from force_remote import force_remote_mode

# Now import faissx
import faissx


def main():
    """Run the test for IndexIDMap with forced remote mode."""
    print("Testing IndexIDMap with forced remote mode...")

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

    # Test case 2: Create an IDMap index - should fail due to lack of remote implementation
    try:
        print("\nTest 2: Creating an IndexIDMap (should fail)...")
        flat_index = faissx.client.IndexFlatL2(d)
        id_map_index = faissx.client.IndexIDMap(flat_index)
        print(f"  Unexpected success: Created IndexIDMap")
    except Exception as e:
        print(f"  Expected error: {e}")

    # Test case 3: Try with non-existent server - should fail with connection errors
    try:
        print("\nTest 3: Connecting to non-existent server (should fail with retries)...")
        force_remote_mode(server_url="tcp://nonexistent-server:12345")
        bad_index = faissx.client.IndexFlatL2(d)
        print(f"  Unexpected success: Connected to non-existent server")
    except Exception as e:
        print(f"  Expected error: {e}")

    print("\nTest complete!")


if __name__ == "__main__":
    main()

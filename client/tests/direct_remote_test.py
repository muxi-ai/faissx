#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISS Remote Connection Test
This test checks if FAISSx is actually connecting to a remote server
"""

from faissx import client as faiss
from faissx.client.client import get_client, FaissXClient
import faissx
import numpy as np

print("=== FAISSx Remote Connection Test ===")
print(f"FAISSx Version: {faissx.__version__}")

# Directly examine the API URL
print(f"\nModule API URL: {faissx._API_URL}")

# Try to get client
client = get_client()
print(f"get_client() returns: {client}")

# Force client creation
print("\nTrying to create client directly:")
try:
    direct_client = FaissXClient(
        server="tcp://localhost:45678",
        api_key="",
        tenant_id="tenant-1"
    )
    print(f"Direct client: {direct_client}")
    print(f"Connected: {direct_client.connected}")

    # Force client into module global for operations
    from faissx.client.client import _client as global_client_var
    print(f"Original global client: {global_client_var}")
    faissx.client.client._client = direct_client
    print(f"Modified global client: {get_client()}")
except Exception as e:
    print(f"Error creating client: {e}")

# Create an index
dim = 8
print(f"\nCreating IndexFlatL2({dim})...")
index = faiss.IndexFlatL2(dim)

# Check index internals
print(f"Index type: {type(index)}")
if hasattr(index, "_using_remote"):
    print(f"Index._using_remote: {index._using_remote}")
if hasattr(index, "_local_index"):
    print(f"Has local index: {index._local_index is not None}")

# Add vectors
print("\nAdding vectors...")
vecs = np.random.random((10, dim)).astype('float32')
index.add(vecs)
print(f"Added vectors: ntotal = {index.ntotal}")

# Search
print("\nPerforming search...")
query = np.random.random((1, dim)).astype('float32')
distances, indices = index.search(query, k=3)
print(f"Search results: {indices[0]}")

print("\n=== TEST COMPLETE ===")
print("Please check:")
print("1. Did you see any logs in the server window?")
print("2. If no logs appeared, we are definitely using local mode")
print("3. Try to stop the server and run this test again; it should fail if we're in remote mode")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEAD SIMPLE FAISSx Server Connection Test

This is a bare-minimum test to confirm whether we can actually
connect to the server on port 45678.
"""

from faissx.client.client import FaissXClient
import numpy as np

print("\n==== SUPER SIMPLE SERVER CONNECTION TEST ====\n")

# Directly create a client instance (bypassing get_client)
print("1. Creating client directly...")
client = FaissXClient(
    server="tcp://localhost:45678",
    api_key="",
    tenant_id="test-tenant"
)
print(f"   Client created: {client}")
print(f"   Server address: {client.server}")
print(f"   Connected: {client.connected}")

# Try to connect explicitly
print("\n2. Explicitly connecting to server...")
try:
    connected = client.connect()
    print(f"   Connection successful: {connected}")
except Exception as e:
    print(f"   Connection failed: {e}")

# Try to send a direct request to the server
print("\n3. Sending ping request to server...")
try:
    print("   Sending request (should appear in server logs)...")
    response = client._send_request({"action": "ping", "message": "Hello Server!"})
    print(f"   Response: {response}")
except Exception as e:
    print(f"   Request failed: {e}")

# Try to create a simple index on the server
print("\n4. Creating test index on server...")
try:
    response = client.create_index(
        name="simple-test-index",
        dimension=4,
        index_type="L2"
    )
    print(f"   Create index response: {response}")

    # Add some vectors
    vectors = np.random.random((5, 4)).astype('float32')
    response = client.add_vectors("simple-test-index", vectors)
    print(f"   Add vectors response: {response}")

    # Search
    query = np.random.random((1, 4)).astype('float32')
    response = client.search("simple-test-index", query, k=2)
    print(f"   Search response: {response}")
except Exception as e:
    print(f"   Index operations failed: {e}")

print("\n==== TEST COMPLETE ====")
print("Check server logs - you should see the ping request and index operations")
print("If you don't see anything in the logs, the client is failing to connect to the server")

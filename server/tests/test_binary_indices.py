#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binary Indices Test Script

This script tests the server's support for binary indices by creating and
using different binary index types.
"""

import zmq
import numpy as np
import msgpack
import json
import sys
import traceback

# Debug level (0=minimal, 1=normal, 2=verbose)
DEBUG_LEVEL = 2

def debug_print(message, level=1):
    """Print debug messages based on verbosity level"""
    if DEBUG_LEVEL >= level:
        print(message)

# Message serialization/deserialization
def send_request(request):
    """Send a request to the server and return the response, using a fresh connection for each request"""
    debug_print(f"Sending request: {request['action']}", 1)
    if DEBUG_LEVEL >= 2:
        debug_print(f"Request details: {json.dumps(request)}", 2)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:45678")

    # Pack the request
    packed_request = msgpack.packb(request, use_bin_type=True)
    socket.send(packed_request)

    debug_print("Request sent, waiting for response...", 1)
    response_data = socket.recv()
    response = msgpack.unpackb(response_data, raw=False)
    debug_print("Response received\n", 1)

    # Close the socket and context
    socket.close()
    context.term()

    return response

def print_response(response):
    """Print response with formatting based on success/failure"""
    if response.get("success", False):
        print("✅ SUCCESS")
    else:
        print(f"❌ ERROR: {response.get('error', 'Unknown error')}")

    print(json.dumps(response, indent=2))
    print("-" * 80)

def run_binary_index_tests():
    """Run tests for binary indices"""
    print("=== Testing Binary Indices Support ===\n")

    try:
        # Test 1: Create binary flat index
        print("\n🔍 Test 1: Create binary flat index")
        response = send_request({
            "action": "create_index",
            "index_id": "binary_flat",
            "dimension": 64,
            "index_type": "BINARY_FLAT"
        })
        print_response(response)

        # Test 2: Add vectors to binary flat index
        print("\n🔍 Test 2: Add vectors to binary flat index")
        # Generate random binary vectors (as floats 0.0 or 1.0)
        vectors = np.random.randint(0, 2, size=(10, 64)).astype(np.float32).tolist()
        response = send_request({
            "action": "add_vectors",
            "index_id": "binary_flat",
            "vectors": vectors
        })
        print_response(response)

        # Test 3: Search binary flat index
        print("\n🔍 Test 3: Search binary flat index")
        query_vector = np.random.randint(0, 2, size=(1, 64)).astype(np.float32).tolist()
        response = send_request({
            "action": "search",
            "index_id": "binary_flat",
            "query_vectors": query_vector,
            "k": 5
        })
        print_response(response)

        # Test 4: Reconstruct vector from binary index
        print("\n🔍 Test 4: Reconstruct vector from binary index")
        response = send_request({
            "action": "reconstruct",
            "index_id": "binary_flat",
            "idx": 0
        })
        print_response(response)

        print("\n✅ All tests completed!")

    except Exception as e:
        print(f"\n❌ Error running tests: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_binary_index_tests()

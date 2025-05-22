#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Server Improvements Test Script

This script tests the standardized response formats and improved behavior
in the FAISSx server implementation.
"""

import zmq
import numpy as np
import msgpack
import json
import time
import traceback
import sys

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
        debug_print(f"Request details: {json.dumps(request, default=str)}", 2)

    try:
        # Create a new connection for each request
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        socket.connect("tcp://localhost:45678")

        # Send the request
        socket.send(msgpack.packb(request, use_bin_type=True))
        debug_print("Request sent, waiting for response...", 2)

        # Receive the response
        response = socket.recv()
        debug_print("Response received", 2)

        # Clean up the connection
        socket.close()
        context.term()

        unpacked = msgpack.unpackb(response, raw=False)
        return unpacked
    except zmq.error.Again:
        print("Error: Server timeout (no response received)")
        return {"success": False, "error": "Server timeout"}
    except Exception as e:
        print(f"Request error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# Pretty print the response
def print_response(response, indent=2):
    try:
        # Print success/error status prominently
        if response.get("success", False):
            print("\n✅ SUCCESS")
        else:
            print(f"\n❌ ERROR: {response.get('error', 'Unknown error')}")

        # Print the full response
        print(json.dumps(response, indent=indent, default=str))
        print("-" * 80)
    except Exception as e:
        print(f"Error printing response: {e}")
        print(f"Raw response: {response}")
        traceback.print_exc()

# Main test script
def run_tests():
    try:
        print("=== Testing Standardized Response Formats and Improved Behavior ===\n")

        # Test 1: Create index with training information
        print("\n🔍 Test 1: Create IVF index with training information")
        response = send_request({
            "action": "create_index",
            "index_id": "test_ivf",
            "dimension": 128,
            "index_type": "IVF100"
        })
        print_response(response)

        # Test 2: Get index status with training requirements
        print("\n🔍 Test 2: Get index status with training requirements")
        response = send_request({
            "action": "get_index_status",
            "index_id": "test_ivf"
        })
        print_response(response)

        # Test 3: Attempt search on untrained index
        print("\n🔍 Test 3: Attempt search on untrained index")
        query_vector = np.random.rand(1, 128).astype(np.float32).tolist()
        response = send_request({
            "action": "search",
            "index_id": "test_ivf",
            "query_vectors": query_vector,
            "k": 5
        })
        print_response(response)

        # Test 4: Train the index
        print("\n🔍 Test 4: Train the index with recommended vectors")
        training_vectors = np.random.rand(500, 128).astype(np.float32).tolist()
        response = send_request({
            "action": "train_index",
            "index_id": "test_ivf",
            "training_vectors": training_vectors
        })
        print_response(response)

        # Test 5: Add vectors to trained index
        print("\n🔍 Test 5: Add vectors to trained index")
        vectors = np.random.rand(100, 128).astype(np.float32).tolist()
        response = send_request({
            "action": "add_vectors",
            "index_id": "test_ivf",
            "vectors": vectors
        })
        print_response(response)

        # Test 6: Search trained index
        print("\n🔍 Test 6: Search trained index")
        response = send_request({
            "action": "search",
            "index_id": "test_ivf",
            "query_vectors": query_vector,
            "k": 5
        })
        print_response(response)

        # Test 7: Search and reconstruct
        print("\n🔍 Test 7: Search and reconstruct")
        response = send_request({
            "action": "search_and_reconstruct",
            "index_id": "test_ivf",
            "query_vectors": query_vector,
            "k": 3
        })
        print_response(response)

        # Test 8: Create flat index (no training required)
        print("\n🔍 Test 8: Create flat index (no training required)")
        response = send_request({
            "action": "create_index",
            "index_id": "test_flat",
            "dimension": 64,
            "index_type": "L2"
        })
        print_response(response)

        # Test 9: Attempt to train flat index
        print("\n🔍 Test 9: Attempt to train flat index")
        training_vectors = np.random.rand(10, 64).astype(np.float32).tolist()
        response = send_request({
            "action": "train_index",
            "index_id": "test_flat",
            "training_vectors": training_vectors
        })
        print_response(response)

        # Test 10: Get vectors with pagination
        print("\n🔍 Test 10: Get vectors with pagination")
        # First add some vectors
        vectors = np.random.rand(20, 64).astype(np.float32).tolist()
        response = send_request({
            "action": "add_vectors",
            "index_id": "test_flat",
            "vectors": vectors
        })
        print_response(response)

        # Now get them with pagination
        response = send_request({
            "action": "get_vectors",
            "index_id": "test_flat",
            "start_idx": 5,
            "limit": 10
        })
        print_response(response)

        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_tests()

#!/usr/bin/env python3
"""
Simple test client for FAISS Proxy Server (ZeroMQ)
"""

import zmq
import time
import numpy as np
import msgpack
import random

# Connection settings
SERVER_ADDRESS = "tcp://localhost:5555"
TIMEOUT_MS = 10000  # 10 seconds

def serialize_message(data):
    """Serialize a message to binary format"""
    return msgpack.packb(data, use_bin_type=True)

def deserialize_message(data):
    """Deserialize a binary message"""
    try:
        return msgpack.unpackb(data, raw=False)
    except Exception as e:
        return {"success": False, "error": f"Failed to deserialize message: {str(e)}"}

def send_request(socket, request):
    """Send a request to the server and get the response"""
    print(f"Sending request: {request}")
    socket.send(serialize_message(request))
    response_data = socket.recv()
    response = deserialize_message(response_data)
    return response

def main():
    """Run a test sequence against the FAISS server"""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
    socket.connect(SERVER_ADDRESS)

    print(f"Connected to FAISS proxy server at {SERVER_ADDRESS}")

    try:
        # 1. Test ping
        print("\n--- Testing Ping ---")
        response = send_request(socket, {"action": "ping"})
        print(f"Ping response: {response}")

        # 2. Create an index
        print("\n--- Creating Index ---")
        index_id = "test_index"
        dimension = 128
        response = send_request(socket, {
            "action": "create_index",
            "index_id": index_id,
            "dimension": dimension,
            "index_type": "L2"
        })
        print(f"Create index response: {response}")

        # 3. List indexes
        print("\n--- Listing Indexes ---")
        response = send_request(socket, {"action": "list_indexes"})
        print(f"List indexes response: {response}")

        # 4. Add vectors
        print("\n--- Adding Vectors ---")
        # Create 100 random vectors of dimension 128
        vectors = np.random.rand(100, dimension).astype(np.float32).tolist()
        response = send_request(socket, {
            "action": "add_vectors",
            "index_id": index_id,
            "vectors": vectors
        })
        print(f"Add vectors response: {response}")

        # 5. Get index stats
        print("\n--- Getting Index Stats ---")
        response = send_request(socket, {
            "action": "get_index_stats",
            "index_id": index_id
        })
        print(f"Index stats response: {response}")

        # 6. Search vectors
        print("\n--- Searching Vectors ---")
        # Create 5 random query vectors
        query_vectors = np.random.rand(5, dimension).astype(np.float32).tolist()
        response = send_request(socket, {
            "action": "search",
            "index_id": index_id,
            "query_vectors": query_vectors,
            "k": 5
        })
        print(f"Search response success: {response.get('success', False)}")
        if response.get("success", False):
            # Print just the first result
            first_result = response["results"][0]
            print(f"First result - indices: {first_result['indices']}, distances: {first_result['distances']}")
        else:
            print(f"Search error: {response.get('error', 'Unknown error')}")

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()

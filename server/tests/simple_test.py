#!/usr/bin/env python3
"""
Simple test script for FAISSx server
"""

import zmq
import msgpack
import numpy as np

# Connect to server
server_address = 'tcp://localhost:45678'
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(server_address)

def send_request(request):
    """Send a request and receive the response"""
    print(f"Sending request: {request['action'] if 'action' in request else request}")
    socket.send(msgpack.packb(request))
    response = socket.recv()
    return msgpack.unpackb(response, raw=False)

# Create an index
index_id = f"test-index-{int(np.random.random() * 10000)}"
create_index_request = {
    "action": "create_index",
    "index_id": index_id,
    "dimension": 128,
    "index_type": "L2"
}

response = send_request(create_index_request)
print(f"Create index response: {response}")

if response.get('success', False):
    print(f"Created index with ID: {index_id}")

    # Add vectors to the index
    vectors = np.random.random((10, 128)).astype(np.float32).tolist()

    add_vectors_request = {
        "action": "add_vectors",
        "index_id": index_id,
        "vectors": vectors
    }

    response = send_request(add_vectors_request)
    print(f"Add vectors response: {response}")

    # Search for vectors
    query_vector = np.random.random((1, 128)).astype(np.float32).tolist()
    search_request = {
        "action": "search",
        "index_id": index_id,
        "query_vectors": query_vector,
        "k": 5
    }

    response = send_request(search_request)
    print(f"Search response: {response}")

    # Get index info
    get_index_request = {
        "action": "get_index_stats",
        "index_id": index_id
    }

    response = send_request(get_index_request)
    print(f"Get index response: {response}")

else:
    print("Failed to create index")

# Close the connection
socket.close()
context.term()

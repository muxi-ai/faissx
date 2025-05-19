#!/usr/bin/env python3
"""
Simple test client for the ZeroMQ FAISS Proxy server.
This script exercises the basic operations to verify server functionality.
"""

import zmq
import time
import numpy as np
import uuid

# Import from faiss_proxy package
from faiss_proxy.server.protocol import (
    prepare_create_index_request,
    prepare_add_vectors_request,
    prepare_search_request,
    prepare_get_index_info_request,
    deserialize_message
)

# Configuration
SERVER_ADDRESS = "tcp://localhost:5555"
API_KEY = "test-key-1"
TENANT_ID = "tenant-1"
DIMENSION = 128


def connect_to_server():
    """Connect to the ZeroMQ server"""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(SERVER_ADDRESS)
    print(f"Connected to server at {SERVER_ADDRESS}")
    return context, socket


def send_request(socket, request):
    """Send a request and receive the response"""
    print("Sending request...")
    socket.send(request)

    print("Waiting for response...")
    response = socket.recv()

    header, vectors, metadata = deserialize_message(response)
    print(f"Response: {header}")
    if metadata:
        print(f"Metadata: {metadata}")
    return header, vectors, metadata


def test_create_index(socket):
    """Test creating an index"""
    print("\n=== Testing create_index ===")
    index_name = f"test-index-{uuid.uuid4()}"
    request = prepare_create_index_request(
        api_key=API_KEY,
        tenant_id=TENANT_ID,
        name=index_name,
        dimension=DIMENSION
    )

    header, _, _ = send_request(socket, request)
    if header.get("status") == "ok":
        index_id = header.get("result", {}).get("index_id")
        print(f"Successfully created index: {index_id}")
        return index_id
    else:
        print(f"Failed to create index: {header}")
        return None


def test_add_vectors(socket, index_id, count=10):
    """Test adding vectors to an index"""
    print(f"\n=== Testing add_vectors ({count} vectors) ===")

    # Generate random vectors
    vectors = np.random.random((count, DIMENSION)).astype(np.float32)

    # Generate IDs and metadata
    vector_ids = [f"vector-{i}" for i in range(count)]
    vector_metadata = [{"value": i, "timestamp": time.time()} for i in range(count)]

    request = prepare_add_vectors_request(
        api_key=API_KEY,
        tenant_id=TENANT_ID,
        index_id=index_id,
        vectors=vectors,
        vector_ids=vector_ids,
        vector_metadata=vector_metadata
    )

    header, _, _ = send_request(socket, request)
    if header.get("status") == "ok":
        result = header.get("result", {})
        print(f"Successfully added vectors: {result}")
        return vector_ids, vectors
    else:
        print(f"Failed to add vectors: {header}")
        return None, None


def test_search(socket, index_id, query_vector, k=5):
    """Test searching for similar vectors"""
    print(f"\n=== Testing search (k={k}) ===")

    request = prepare_search_request(
        api_key=API_KEY,
        tenant_id=TENANT_ID,
        index_id=index_id,
        query_vector=query_vector,
        k=k
    )

    header, _, _ = send_request(socket, request)
    if header.get("status") == "ok":
        results = header.get("result", [])
        print(f"Search results: {results}")
        return results
    else:
        print(f"Failed to search: {header}")
        return None


def test_get_index_info(socket, index_id):
    """Test getting index information"""
    print("\n=== Testing get_index_info ===")

    request = prepare_get_index_info_request(
        api_key=API_KEY,
        tenant_id=TENANT_ID,
        index_id=index_id
    )

    header, _, _ = send_request(socket, request)
    if header.get("status") == "ok":
        info = header.get("result", {})
        print(f"Index info: {info}")
        return info
    else:
        print(f"Failed to get index info: {header}")
        return None


def main():
    """Run the test sequence"""
    print("Starting ZeroMQ FAISS Proxy test client")

    # Connect to server
    context, socket = connect_to_server()

    try:
        # Create a test index
        index_id = test_create_index(socket)
        if not index_id:
            print("Test failed: Could not create index")
            return

        # Add vectors
        vector_ids, vectors = test_add_vectors(socket, index_id, count=10)
        if vectors is None:
            print("Test failed: Could not add vectors")
            return

        # Get index info
        info = test_get_index_info(socket, index_id)
        if not info:
            print("Test failed: Could not get index info")
            return

        # Search using one of the vectors
        query_vector = vectors[0]
        results = test_search(socket, index_id, query_vector, k=3)
        if not results:
            print("Test failed: Could not search")
            return

        print("\n=== All tests completed successfully ===")

    finally:
        # Clean up
        socket.close()
        context.term()
        print("Test client shutdown complete")


if __name__ == "__main__":
    main()

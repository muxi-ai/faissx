#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Debug ZeroMQ connection to FAISSx server."""

import sys
import os
import traceback
import time
import zmq

# Add parent directory to path to import faissx
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def debug_connection():
    """Debug ZeroMQ connection to server."""
    server_url = "tcp://localhost:45679"
    print(f"Testing direct ZeroMQ connection to {server_url}")

    # Create ZeroMQ context and socket
    try:
        # Set up connection with timeouts
        context = zmq.Context()
        print("Created ZeroMQ context")

        socket = context.socket(zmq.REQ)
        print("Created socket")

        # Set socket options for better debugging
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second receive timeout
        socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second send timeout
        print("Set socket options")

        # Connect to server
        print(f"Connecting to {server_url}...")
        socket.connect(server_url)
        print("Connected successfully")

        # Send ping request
        import msgpack
        request = {"action": "ping"}
        packed_request = msgpack.packb(request)
        print(f"Sending ping request: {request}")

        socket.send(packed_request)
        print("Request sent, waiting for response...")

        # Wait for response
        response = socket.recv()
        print("Received response")

        result = msgpack.unpackb(response, raw=False)
        print(f"Unpacked response: {result}")

        # Clean up
        socket.close()
        context.term()
        print("Connection test completed successfully!")
        return True
    except Exception as e:
        print(f"ZeroMQ error: {e}")
        traceback.print_exc()
        return False


def debug_faissx_client():
    """Debug FAISSx client connection."""
    import faissx
    from faissx.client.client import FaissXClient

    server_url = "tcp://localhost:45679"
    print(f"\nTesting FAISSx client connection to {server_url}")

    # Set global configuration
    faissx.configure(url=server_url)
    print(f"Configured FAISSx with URL: {faissx._API_URL}")

    try:
        # Create client directly (not using get_client())
        client = FaissXClient(server=server_url)
        print("Successfully created FaissXClient")

        # Try a simple operation
        response = client.list_indexes()
        print(f"list_indexes response: {response}")

        return True
    except Exception as e:
        print(f"FaissXClient error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # First test direct ZeroMQ connection
    zmq_success = debug_connection()

    # Then test FAISSx client
    client_success = debug_faissx_client()

    # Overall result
    print("\n===== Debug Summary =====")
    print(f"ZeroMQ connection: {'PASS' if zmq_success else 'FAIL'}")
    print(f"FAISSx client: {'PASS' if client_success else 'FAIL'}")

    sys.exit(0 if (zmq_success and client_success) else 1)

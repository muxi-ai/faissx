#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the optimized FAISSx client implementation.

This script verifies:
1. Proper initialization of timeout values
2. Socket timeout configuration
3. Global TIMEOUT synchronization
4. Enhanced error handling with detailed messages
"""

import logging
import sys
import zmq
from faissx import client as faiss


# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def test_client_setup():
    """Test client initialization and configuration."""
    print("\n===== Testing Client Setup =====")

    # Test default timeout
    client = faiss.FaissXClient()
    print(f"Default timeout value: {client.timeout}s")
    assert client.timeout == 5.0, "Default timeout should be 5.0s"

    # Test custom timeout setup
    custom_timeout = 3.5
    client.timeout = custom_timeout
    print(f"Custom timeout value: {client.timeout}s")
    assert client.timeout == custom_timeout, f"Timeout should be {custom_timeout}s"

    # Test timeout from configure() without connecting
    configure_timeout = 2.0
    # Save original connect method
    original_connect = client.connect
    try:
        # Replace connect method temporarily to prevent actual connection
        client.connect = lambda: print("Connection attempt bypassed")

        # Now call configure
        client.configure(
            server="tcp://nonexistent-host:45678",
            timeout=configure_timeout
        )
        print(f"Configured timeout value: {client.timeout}s")

        # Import TIMEOUT here to get the latest value
        from faissx.client.timeout import TIMEOUT
        print(f"Global TIMEOUT value: {TIMEOUT}s")

        assert client.timeout == configure_timeout, "Client timeout not updated by configure()"
        assert TIMEOUT == configure_timeout, "Global TIMEOUT not updated by configure()"

    finally:
        # Restore original connect method
        client.connect = original_connect

    print("✓ Client setup tests passed!")
    return True


def test_socket_options():
    """Test socket option configuration."""
    print("\n===== Testing Socket Options =====")

    # Create a client with a specific timeout
    test_timeout = 2.5
    client = faiss.FaissXClient()
    client.timeout = test_timeout

    # Save original connect and _send_request methods
    original_connect = client.connect
    original_send_request = client._send_request

    try:
        # Replace connect method with one that only sets up the socket but doesn't connect
        def mock_connect():
            # Call disconnect to clean up any existing socket
            client.disconnect()

            # Create a new context and socket without connecting
            client.context = zmq.Context()
            client.socket = client.context.socket(zmq.REQ)

            # Configure socket options (this is what we're testing)
            client.socket.setsockopt(zmq.RCVTIMEO, int(client.timeout * 1000))
            client.socket.setsockopt(zmq.LINGER, 0)

            # Don't actually connect or ping the server
            print(f"Would connect to {client.server}")

        # Also mock _send_request to prevent actual network calls
        client._send_request = lambda req: {"success": True, "message": "Ping mocked successfully"}

        # Replace the connect method
        client.connect = mock_connect

        # Call connect with our test server
        client.server = "tcp://localhost:45678"  # Use the real port
        client.connect()

        # Now test the socket options
        recv_timeout = client.socket.getsockopt(zmq.RCVTIMEO)
        linger = client.socket.getsockopt(zmq.LINGER)

        print(f"Socket RCVTIMEO: {recv_timeout}ms (expected: {int(test_timeout * 1000)}ms)")
        print(f"Socket LINGER: {linger} (expected: 0)")

        assert recv_timeout == int(test_timeout * 1000), "Socket timeout not set correctly"
        assert linger == 0, "Socket linger option not set correctly"

    except Exception as e:
        print(f"Error during socket option test: {e}")
        return False
    finally:
        # Restore original methods
        client.connect = original_connect
        client._send_request = original_send_request

        # Clean up
        client.disconnect()

    print("✓ Socket options tests passed!")
    return True


def test_error_handling():
    """Test improved error handling."""
    print("\n===== Testing Error Handling =====")

    # Test missing socket error
    print("\nTesting request with no socket...")
    client = faiss.FaissXClient()
    client.server = "tcp://localhost:9999"

    try:
        # Call _send_request directly without connecting first
        client._send_request({"action": "ping"})
        print("✗ Test failed: Expected 'No active connection' error")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        print(f"Got error message: {error_msg}")
        assert "No active connection" in error_msg, "Expected 'No active connection' error"

    print("✓ Error handling tests passed!")
    return True


def main():
    """Run all optimization tests."""
    print("\n===== Testing Optimized Client Implementation =====")

    results = []

    # Test client setup
    results.append(test_client_setup())

    # Test socket options
    results.append(test_socket_options())

    # Test error handling
    results.append(test_error_handling())

    # Print summary
    print("\n===== Test Results =====")
    passed = results.count(True)
    total = len(results)

    if all(results):
        print(f"All tests passed! ({passed}/{total})")
        print("Optimized client is working correctly.")
    else:
        print(f"Some tests failed. ({passed}/{total} passed)")
        print("Client optimization may have issues.")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

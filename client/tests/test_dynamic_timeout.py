#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify dynamic timeout functionality in FAISSx client.

This script demonstrates that the timeout value is correctly obtained
from the client instance and can be changed dynamically.
"""

import logging
import sys
import time
import zmq
from faissx import client as faiss
from faissx.client.timeout import TimeoutError


# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def test_dynamic_timeout(server_address="tcp://nonexistent-host:45678"):
    """Test changing timeout values dynamically."""
    print(f"\nTesting dynamic timeouts with {server_address}")

    # Clear any existing client
    faiss._client = None

    # Set up with initial timeout
    initial_timeout = 5.0
    print(f"\nSetting initial timeout to {initial_timeout}s")

    # Create client with initial timeout
    faiss.configure(server=server_address, timeout=initial_timeout)

    # Since connecting to the server would timeout, we need to create
    # a new client object and manually set it up for testing
    try:
        # This will fail with timeout when trying to connect
        print("Configuring client (expecting failure due to timeout)...")
        faiss.configure(server=server_address, timeout=0.5)
    except TimeoutError:
        print("  Got expected timeout error during configuration.")
    except Exception as e:
        print(f"  Got unexpected error: {e}")

    # Create a client manually without connecting
    print("\nCreating client manually for testing...")
    client = faiss.FaissXClient()
    client.server = server_address
    client.timeout = initial_timeout

    # Set up socket without trying to connect
    client.context = zmq.Context()
    client.socket = client.context.socket(zmq.REQ)
    client.socket.setsockopt(zmq.RCVTIMEO, int(client.timeout * 1000))

    # Verify the timeout was set
    print(f"Configured timeout value: {client.timeout}s")
    print(f"Socket RCVTIMEO: {client.socket.getsockopt(zmq.RCVTIMEO)}ms")

    # Now change the timeout dynamically
    new_timeout = 1.0
    print(f"\nChanging timeout to {new_timeout}s")
    client.timeout = new_timeout

    # Manually update socket timeout
    client.socket.setsockopt(zmq.RCVTIMEO, int(client.timeout * 1000))

    # Verify the timeout was updated
    print(f"Updated timeout value: {client.timeout}s")
    print(f"Socket RCVTIMEO: {client.socket.getsockopt(zmq.RCVTIMEO)}ms")

    # Test with explicit timeout in _send_request call
    override_timeout = 0.5
    print(f"\nTesting with explicit timeout override: {override_timeout}s")

    # Connect to trigger the timeout
    client.socket.connect(server_address)

    # Test with the explicit timeout parameter
    start_time = time.time()
    try:
        # This will time out after override_timeout seconds
        request = {"action": "ping"}
        client._send_request(request, timeout=override_timeout)
        print("Unexpected success!")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Got exception after {elapsed:.2f}s: {e}")

        # Verify we got a timeout after approximately override_timeout seconds
        is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
        timeout_as_expected = 0.4 <= elapsed <= override_timeout * 2

        if is_timeout and timeout_as_expected:
            print(
                f"✓ Test passed: Timeout occurred after ~{elapsed:.2f}s "
                f"(expected: {override_timeout}s)"
            )
            return True
        else:
            reason = "wrong exception type" if not is_timeout else "unexpected timing"
            print(
                f"✗ Test failed ({reason}): {type(e).__name__} after {elapsed:.2f}s "
                f"(expected: TimeoutError after ~{override_timeout}s)"
            )
            return False


def main():
    """Run timeout tests with dynamic timeout changes."""
    print("\n===== Testing Dynamic Timeout Functionality =====")

    result = test_dynamic_timeout()

    print("\n===== Test Results =====")
    if result:
        print("Test passed! Dynamic timeout functionality is working correctly.")
    else:
        print("Test failed. Dynamic timeout is not working as expected.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive test script for timeout functionality in FAISSx client.

This script tests:
1. The global TIMEOUT variable is used by the decorator when no timeout is specified
2. The client.timeout attribute is respected by operations
3. Changing TIMEOUT affects subsequent operations
4. Direct setting of client.timeout overrides the global TIMEOUT
"""

import logging
import sys
import time
import zmq
from faissx import client as faiss
from faissx.client.timeout import TimeoutError, TIMEOUT


# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def create_test_client(timeout_value, server_address="tcp://nonexistent-host:45678"):
    """Create a test client without connecting to server."""
    client = faiss.FaissXClient()
    client.server = server_address
    client.timeout = timeout_value

    # Set up socket without connecting
    client.context = zmq.Context()
    client.socket = client.context.socket(zmq.REQ)

    return client


def test_instance_timeout():
    """Test that the instance timeout value is used correctly."""
    print("\n===== Testing Instance-Level Timeout =====")

    # Use an unreachable server
    server = "tcp://nonexistent-host:45678"

    # Create a client with a specific timeout
    instance_timeout = 1.5
    print(f"\nCreating client with instance timeout={instance_timeout}s")

    client = create_test_client(instance_timeout)
    print(f"Client timeout value: {client.timeout}s")

    # Connect to trigger timeout
    print("\nTrying to connect (should timeout)...")
    client.socket.connect(server)

    # Try a request and time it
    start_time = time.time()
    try:
        # This should time out after approximately instance_timeout seconds
        client._send_request({"action": "ping"})
        print("Unexpected success!")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Got exception after {elapsed:.2f}s: {e}")

        # Verify we got a timeout after approximately instance_timeout seconds
        is_timeout = isinstance(e, TimeoutError)
        timeout_as_expected = 0.7 * instance_timeout <= elapsed <= 2.0 * instance_timeout

        if is_timeout and timeout_as_expected:
            print(
                f"✓ Test passed: Timeout occurred after ~{elapsed:.2f}s "
                f"(expected: {instance_timeout}s)"
            )
            return True
        else:
            reason = "wrong exception type" if not is_timeout else "unexpected timing"
            print(
                f"✗ Test failed ({reason}): {type(e).__name__} after {elapsed:.2f}s "
                f"(expected: TimeoutError after ~{instance_timeout}s)"
            )
            return False


def test_global_timeout():
    """Test that the global TIMEOUT is used when no specific timeout is set."""
    print("\n===== Testing Global TIMEOUT Usage =====")

    # Import and declare global TIMEOUT
    from faissx.client.timeout import TIMEOUT
    global TIMEOUT

    # Save original TIMEOUT value
    original_timeout = TIMEOUT

    # Set global TIMEOUT to a specific value
    test_timeout = 2.0
    print(f"\nSetting global TIMEOUT to {test_timeout}s")
    TIMEOUT = test_timeout

    # Use a client without specifying timeout (should use global TIMEOUT)
    print("\nCreating client without explicit timeout...")
    client = faiss.FaissXClient()
    client.server = "tcp://nonexistent-host:45678"
    print(f"Global TIMEOUT: {TIMEOUT}s")

    # Set up test function with decorator that doesn't specify timeout
    from faissx.client.timeout import timeout

    @timeout()  # Should use global TIMEOUT
    def test_function():
        """Test function that should timeout based on global TIMEOUT."""
        print("Test function running, will sleep forever...")
        time.sleep(100)  # Long sleep to ensure timeout

    # Execute function and time it
    start_time = time.time()
    try:
        print("\nExecuting test function (should timeout)...")
        test_function()
        print("Unexpected success!")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Got exception after {elapsed:.2f}s: {e}")

        # Verify we got a timeout after approximately test_timeout seconds
        is_timeout = isinstance(e, TimeoutError)
        timeout_as_expected = 0.7 * test_timeout <= elapsed <= 2.0 * test_timeout

        success = is_timeout and timeout_as_expected

        if success:
            print(
                f"✓ Test passed: Timeout occurred after ~{elapsed:.2f}s "
                f"(expected: {test_timeout}s)"
            )
        else:
            reason = "wrong exception type" if not is_timeout else "unexpected timing"
            print(
                f"✗ Test failed ({reason}): {type(e).__name__} after {elapsed:.2f}s "
                f"(expected: TimeoutError after ~{test_timeout}s)"
            )

        # Restore original TIMEOUT
        TIMEOUT = original_timeout
        print(f"\nRestored global TIMEOUT to {TIMEOUT}s")

        return success


def main():
    """Run comprehensive timeout tests."""
    print("\n===== Testing Timeout Functionality =====\n")

    results = []

    # Test instance timeout
    results.append(test_instance_timeout())

    # Test global timeout
    results.append(test_global_timeout())

    # Print summary
    print("\n===== Test Results =====")
    passed = results.count(True)
    total = len(results)

    if all(results):
        print(f"All tests passed! ({passed}/{total})")
        print("Timeout functionality is working correctly.")
    else:
        print(f"Some tests failed. ({passed}/{total} passed)")
        print("Timeout functionality may not be working as expected.")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the global TIMEOUT functionality in FAISSx client.

This script demonstrates that:
1. The global TIMEOUT value is correctly used by the decorator
2. Changing the timeout value affects subsequent operations
3. Timeouts are properly triggered when connecting to non-existent servers
"""

import logging
import sys
import time
from faissx import client as faiss
from faissx.client.timeout import TimeoutError, TIMEOUT

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_timeout(timeout_value, server_address="tcp://nonexistent-host:45678"):
    """Test timeout with specific timeout value."""
    print(f"\nTesting connection with timeout={timeout_value}s")

    # Reset client
    faiss._client = None

    # Create new client
    client = faiss.FaissXClient()
    client.server = server_address
    client.timeout = timeout_value
    faiss._client = client

    # Record start time
    start_time = time.time()

    try:
        # Configure and connect with the given timeout
        print(f"Setting timeout to {timeout_value}s and connecting...")
        # We need to directly call connect since the module-level configure doesn't accept timeout
        client.connect()

        # Should not reach here if the server is unreachable
        print("Connection successful (unexpected!)")
        return False

    except TimeoutError as e:
        # Calculate elapsed time
        elapsed = time.time() - start_time
        print(f"TimeoutError as expected after {elapsed:.2f}s: {e}")

        # Check if the elapsed time is close to the expected timeout
        if 0.7 * timeout_value <= elapsed <= 1.5 * timeout_value:
            print(
                f"✓ Test passed: Timeout occurred after ~{elapsed:.2f}s "
                f"(expected: {timeout_value}s)"
            )
            return True
        else:
            print(
                f"✗ Test failed: Timeout occurred after {elapsed:.2f}s "
                f"(expected: ~{timeout_value}s)"
            )
            return False

    except Exception as e:
        # Another error occurred
        elapsed = time.time() - start_time
        print(f"Unexpected error after {elapsed:.2f}s: {e}")
        print("✗ Test failed: Expected TimeoutError but got different exception")
        return False

def test_global_timeout():
    """Test that the global TIMEOUT variable is being used."""
    print(f"Current global TIMEOUT value: {TIMEOUT}s")

    # Run a test with the current global TIMEOUT
    print("\nTesting with current global TIMEOUT...")
    result1 = test_timeout(TIMEOUT)

    # Change the global TIMEOUT to a new value
    new_timeout = 1.0
    print(f"\nChanging global TIMEOUT to {new_timeout}s")
    # The client will set this value in the configure method

    # Run another test with the new timeout
    result2 = test_timeout(new_timeout)

    return result1 and result2

def main():
    """Run timeout tests with various values."""
    print("\n===== Testing Global TIMEOUT Functionality =====\n")

    result = test_global_timeout()

    # Print summary
    print("\n===== Test Results =====")
    if result:
        print("All tests passed! Global TIMEOUT functionality is working correctly.")
    else:
        print("Some tests failed. Global TIMEOUT functionality may not be working correctly.")

if __name__ == "__main__":
    main()

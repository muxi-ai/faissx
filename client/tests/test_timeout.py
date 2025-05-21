#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the timeout functionality in FAISSx client.

This script demonstrates the timeout mechanism when connecting to
non-existent servers or servers that don't respond in time.
"""

import logging
import sys
import time
from faissx import client as faiss
from faissx.client.timeout import TimeoutError

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_timeout(server_address, timeout):
    """Test timeout with specific timeout value."""
    print(f"\nTesting connection to {server_address} with timeout={timeout}s")

    # Record start time
    start_time = time.time()

    try:
        # Clear any existing client
        faiss._client = None

        # Try to configure with the given timeout
        print(f"Attempting to connect with {timeout}s timeout...")
        faiss.configure(server=server_address, timeout=timeout)

        # Should not reach here if the server is unreachable
        print("Connection successful (unexpected!)")
        return False

    except TimeoutError as e:
        # Calculate elapsed time
        elapsed = time.time() - start_time
        print(f"TimeoutError as expected after {elapsed:.2f}s: {e}")
        print("✓ Test passed: Timeout occurred as expected")
        return True

    except Exception as e:
        # Another error occurred
        elapsed = time.time() - start_time
        print(f"Unexpected error after {elapsed:.2f}s: {e}")
        print("✗ Test failed: Expected TimeoutError but got different exception")
        return False

def main():
    """Run timeout tests with various values."""
    # Test with non-existent server
    nonexistent_server = "tcp://nonexistent-host:45678"

    # Test with different timeout values
    results = []

    print("\n===== Testing Timeout Functionality =====\n")

    # Test with short timeout
    results.append(test_timeout(nonexistent_server, 1.0))

    # Test with longer timeout
    results.append(test_timeout(nonexistent_server, 2.0))

    # Print summary
    print("\n===== Test Results =====")
    if all(results):
        print("All tests passed! Timeout functionality is working correctly.")
    else:
        print(f"Some tests failed. {results.count(True)}/{len(results)} passed.")

if __name__ == "__main__":
    main()

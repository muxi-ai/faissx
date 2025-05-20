#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test the get_client() function to diagnose why it's returning None."""

import sys
import os
import traceback

# Add parent directory to path to import faissx
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import faissx


def test_get_client():
    """Test the get_client() function."""
    from faissx.client import get_client

    # Configure with remote URL
    server_url = "tcp://localhost:45679"
    print(f"Configuring FAISSx with URL: {server_url}")
    faissx.configure(url=server_url)

    # Verify configuration
    print(f"Configured URL: {faissx._API_URL}")

    # Print code of get_client
    import inspect
    print("\nget_client() function code:")
    print(inspect.getsource(get_client))

    # Try to get client
    print("\nCalling get_client()...")
    try:
        client = get_client()
        print(f"Result: {client}")

        if client is None:
            print("ERROR: get_client() returned None without raising an exception!")
            return False

        print("Successfully got client")
        return True
    except Exception as e:
        print(f"Exception raised: {e}")
        traceback.print_exc()
        return False


def test_client_directly():
    """Test creating a client instance directly."""
    from faissx.client.client import FaissXClient

    server_url = "tcp://localhost:45679"
    print(f"\nCreating FaissXClient directly with URL: {server_url}")

    try:
        client = FaissXClient(server=server_url)
        print(f"Result: {client}")
        print("Successfully created client directly")
        return True
    except Exception as e:
        print(f"Exception raised: {e}")
        traceback.print_exc()
        return False


def test_singleton_behavior():
    """Test the singleton behavior of get_client()."""
    from faissx.client import get_client

    print("\nTesting singleton behavior:")

    # Get direct access to the module-level _client variable
    import faissx.client.client
    print(f"Current _client: {faissx.client.client._client}")

    print("Setting _client to None...")
    faissx.client.client._client = None

    print("Calling get_client() again...")
    try:
        client = get_client()
        print(f"Result: {client}")

        if client is None:
            print("ERROR: get_client() still returned None!")
            return False

        print("Successfully got client after resetting")
        return True
    except Exception as e:
        print(f"Exception raised: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the tests
    print("===== Testing get_client() function =====")
    get_client_success = test_get_client()

    # Test direct client creation
    direct_success = test_client_directly()

    # Test singleton behavior
    singleton_success = test_singleton_behavior()

    # Overall result
    print("\n===== Test Summary =====")
    print(f"get_client() test: {'PASS' if get_client_success else 'FAIL'}")
    print(f"Direct client test: {'PASS' if direct_success else 'FAIL'}")
    print(f"Singleton behavior test: {'PASS' if singleton_success else 'FAIL'}")

    sys.exit(0 if (get_client_success and direct_success and singleton_success) else 1)

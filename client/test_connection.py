#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test connection to FAISSx server."""

import sys
import os

# Add parent directory to path to import faissx
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import faissx
from faissx.client import get_client


def test_connection():
    """Test connection to server."""
    print("Testing connection to FAISSx server on port 45679")

    # Configure with remote URL
    faissx.configure(url="tcp://localhost:45679")

    # Print current configuration
    print(f"API URL: {faissx._API_URL}")

    try:
        # Get client directly
        client = get_client()
        print(f"Client: {client}")

        if client is None:
            print("ERROR: Client is None, but no exception was raised!")
            print("This suggests get_client() is suppressing errors.")
            return False

        print("Successfully connected to server!")

        # Try a simple operation to verify the connection
        index_name = "test-connection-index"
        dimension = 32
        response = client.create_index(name=index_name, dimension=dimension)
        print(f"Created index: {response.get('index_id', '(not found)')}")

        return True
    except Exception as e:
        print(f"Connection error: {e}")

        # Print the exception traceback for debugging
        import traceback
        traceback.print_exc()

        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Direct test of IndexIDMap in remote mode that bypasses get_client()."""

import sys
import os
import numpy as np

# Add parent directory to path to import faissx
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import faissx
from faissx.client.client import FaissXClient

# Configure for server URL
SERVER_URL = "tcp://localhost:45679"


def test_direct_remote_idmap():
    """Test IndexIDMap with direct client creation."""
    print(f"Testing direct remote IndexIDMap with server at {SERVER_URL}")

    # Configure faissx but don't use get_client()
    faissx.configure(url=SERVER_URL)

    # Create client directly
    try:
        client = FaissXClient(server=SERVER_URL)
        print("Created client directly")

        # Create a base index
        base_name = f"test-flat-{os.urandom(4).hex()}"
        dimension = 32
        # create_index just returns the index_id string
        base_index_id = client.create_index(name=base_name, dimension=dimension)
        print(f"Created base index: {base_index_id}")

        # Add some vectors to base index
        vectors = np.random.random((10, dimension)).astype(np.float32)
        vectors_list = vectors.tolist()

        request = {
            "action": "add_vectors",
            "index_id": base_index_id,
            "vectors": vectors_list
        }

        response = client._send_request(request)
        print(f"Added vectors to base index: {response}")

        # Create IndexIDMap directly
        idmap_name = f"test-idmap-{os.urandom(4).hex()}"
        idmap_request = {
            "action": "create_index",
            "index_id": idmap_name,
            "dimension": dimension,
            "index_type": f"IDMap:{base_index_id}"
        }

        idmap_response = client._send_request(idmap_request)
        idmap_id = idmap_response.get("index_id", idmap_name)
        print(f"Created IDMap index: {idmap_id}")

        # Add vectors with IDs
        id_vectors = np.random.random((5, dimension)).astype(np.float32)
        ids = np.array([100, 200, 300, 400, 500], dtype=np.int64)

        add_request = {
            "action": "add_with_ids",
            "index_id": idmap_id,
            "vectors": id_vectors.tolist(),
            "ids": ids.tolist()
        }

        add_response = client._send_request(add_request)
        print(f"Added vectors with IDs: {add_response}")

        # Search in IDMap index
        query = np.random.random((1, dimension)).astype(np.float32)

        search_request = {
            "action": "search",
            "index_id": idmap_id,
            "query_vectors": query.tolist(),
            "k": 3
        }

        search_response = client._send_request(search_request)
        print(f"Search results: {search_response}")

        # Check if the search was successful
        if not search_response.get("success", False):
            print("ERROR: Search failed")
            return False

        results = search_response.get("results", [])
        if not results or not results[0].get("indices", []):
            print("ERROR: No search results returned")
            return False

        print("Basic IndexIDMap operations in remote mode work correctly!")
        return True
    except Exception as e:
        import traceback
        print(f"Error in direct test: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    success = test_direct_remote_idmap()
    sys.exit(0 if success else 1)

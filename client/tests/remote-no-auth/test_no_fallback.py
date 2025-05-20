#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test no-fallback behavior for FAISSx
#
# Copyright (C) 2025 Ran Aroussi

"""
Test that FAISSx raises errors instead of falling back to local mode.

This script demonstrates how the client behaves when configured to
not fall back to local mode when remote operations fail.
"""

import unittest
import numpy as np
import os
import sys
import time

# Get the absolute path to the fix_imports module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import fix_imports

# Add parent directory to path to import faissx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import directly to test no-fallback behavior
import faissx
from faissx.client.client import FaissXClient, get_client


class TestNoFallback(unittest.TestCase):
    """Test that FAISSx client doesn't fall back to local mode."""

    def setUp(self):
        """Set up test environment with remote server configuration."""
        # Configure to use remote server with no-fallback behavior
        # This mimics what we've done in the client.__init__.py file
        os.environ["FAISSX_FALLBACK_TO_LOCAL"] = "0"  # Explicitly disable fallback

        # Directly modify the module variable to ensure no fallback
        faissx.client._FALLBACK_TO_LOCAL = False

        # Configure the client to connect to the remote server
        faissx.configure(url="tcp://0.0.0.0:45678")

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_idmap_no_fallback(self):
        """Test that IndexIDMap raises an error instead of falling back to local mode."""
        # Create a base index
        base_index = faissx.client.IndexFlatL2(32)

        try:
            # This should raise an error since IndexIDMap is not implemented in remote mode
            index = faissx.client.IndexIDMap(base_index)
            self.fail("Expected an error but none was raised")
        except Exception as e:
            # Verify it's the expected error message
            self.assertIn("Remote mode for IndexIDMap not implemented yet", str(e))

    def test_unavailable_server_no_fallback(self):
        """Test that connecting to an unavailable server raises an error after retries."""
        # Configure with an invalid server URL
        faissx.configure(url="tcp://nonexistent-server:12345")

        start_time = time.time()

        try:
            # Attempt to create an index which should trigger connection attempts
            index = faissx.client.IndexFlatL2(32)
            self.fail("Expected a connection error but none was raised")
        except Exception as e:
            # Verify it's a connection error
            self.assertIn("connect", str(e).lower())

            # Verify that retry attempts were made (should take at least 2 seconds)
            elapsed_time = time.time() - start_time
            self.assertGreaterEqual(elapsed_time, 2.0,
                                    "Retry mechanism not working - should take at least 2 seconds")


if __name__ == "__main__":
    unittest.main()

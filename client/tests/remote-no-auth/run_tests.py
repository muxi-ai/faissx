#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Run all remote mode tests for FAISSx
#
# Copyright (C) 2025 Ran Aroussi

"""
Test runner for FAISSx remote mode tests.

This script runs all test files for FAISSx's remote mode to verify
it works as a drop-in replacement for FAISS when connected to a server.
"""

import unittest
import sys
import os
import argparse

# Make sure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import faissx
import faissx

# Import the remote configuration function
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from fix_imports import configure_remote


def run_tests(verbosity=1, pattern=None):
    """
    Run tests with the specified verbosity and pattern.

    Args:
        verbosity: Level of test output (1=default, 2=verbose)
        pattern: Pattern to match test files (default: test_*.py)

    Returns:
        True if all tests pass, False otherwise
    """
    if pattern is None:
        pattern = "test_*.py"

    # Configure client to connect to remote server
    faissx.configure(
        url="tcp://0.0.0.0:45678",
        tenant_id=None  # No tenant ID for unauthenticated mode
    )

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__), pattern=pattern)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Return True if tests were successful, False otherwise
    return result.wasSuccessful()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run remote mode tests for FAISSx")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-p", "--pattern", default="test_*.py", help="Pattern to match test files")
    args = parser.parse_args()

    # Set verbosity level based on arguments
    verbosity = 2 if args.verbose else 1

    # Run tests
    success = run_tests(verbosity=verbosity, pattern=args.pattern)
    sys.exit(0 if success else 1)

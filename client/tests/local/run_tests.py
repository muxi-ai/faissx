#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Run all local mode tests for FAISSx
#
# Copyright (C) 2025 Ran Aroussi

"""
Test runner for FAISSx local mode tests.

This script runs all test files for FAISSx's local mode to verify
it works as a drop-in replacement for FAISS.
"""

import unittest
import sys
import os
import argparse

# Make sure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


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

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__), pattern=pattern)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Return True if tests were successful, False otherwise
    return result.wasSuccessful()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run local mode tests for FAISSx")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-p", "--pattern", default="test_*.py", help="Pattern to match test files")
    args = parser.parse_args()

    # Set verbosity level based on arguments
    verbosity = 2 if args.verbose else 1

    # Clear any environment variables that would make tests connect to a server
    env_vars = [
        'FAISSX_SERVER',
        'FAISSX_API_KEY',
        'FAISSX_TENANT_ID'
    ]

    original_env = {}
    for key in env_vars:
        original_env[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]

    try:
        # Run tests
        success = run_tests(verbosity=verbosity, pattern=args.pattern)
        sys.exit(0 if success else 1)
    finally:
        # Restore environment variables
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

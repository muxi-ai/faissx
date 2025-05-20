#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test runner for local mode FAISSx tests.
This script discovers and runs all test cases in the local-mode directory.
"""

import unittest
import os
import sys


def run_all_tests():
    """Find and run all tests in the current directory."""
    # Find directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Add parent directories to path to enable imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(script_dir)))

    # Discover all tests in this directory
    loader = unittest.TestLoader()
    suite = loader.discover(script_dir, pattern="test_*.py")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 70)
    print("Running FAISSx Local Mode Tests")
    print("=" * 70)
    print("These tests verify that FAISSx works as a drop-in replacement for FAISS")
    print("in local mode (without calling configure()).")
    print("-" * 70)

    success = run_all_tests()

    print("=" * 70)
    if success:
        print("All local mode tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)

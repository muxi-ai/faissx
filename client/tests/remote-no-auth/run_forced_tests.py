#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Run all FAISSx remote tests with forced remote mode
#
# Copyright (C) 2025 Ran Aroussi

"""
Run all FAISSx remote mode tests with progress display.

This script discovers and runs all test files in the remote-no-auth directory,
providing a summary of results and focusing on the basic tests first.
The tests communicate with a remote FAISSx server at 0.0.0.0:45678.

This script explicitly configures remote mode and ensures all tests use
remote server functionality even if configure_remote() is not explicitly called in tests.
"""

import os
import sys
import time

# Add parent directory to path to import faissx
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import faissx and explicitly configure remote server
import faissx
faissx.configure(
    url="tcp://0.0.0.0:45678",
    tenant_id=None,  # No tenant ID for unauthenticated mode
    force_remote=True  # Force remote mode for all operations
)

# Import the run_priority_tests function from run_all_tests.py
from run_all_tests import run_priority_tests, run_discovery_tests


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run FAISSx remote tests')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests via discovery')
    args = parser.parse_args()

    start_time = time.time()

    print("Running tests with forced remote mode - explicitly configured to tcp://0.0.0.0:45678")

    if args.all:
        result = run_discovery_tests()
    else:
        # Run tests in priority order
        run_priority_tests()
        result = None  # Individual results already printed

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"Test execution completed in {elapsed:.2f} seconds")

    if result:
        print(f"Total: {result.testsRun} tests run")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

    print("=" * 80)

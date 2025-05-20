#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Run all FAISSx remote tests with progress display
#
# Copyright (C) 2025 Ran Aroussi

"""
Run all FAISSx remote mode tests with progress display.

This script discovers and runs all test files in the remote-no-auth directory,
providing a summary of results and focusing on the basic tests first.
The tests communicate with a remote FAISSx server at 0.0.0.0:45678.
"""

import os
import sys
import unittest
import time
import importlib.util

# Add parent directory to path to import faissx
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import faissx and configure connection to remote server
import faissx
faissx.configure(
    url="tcp://0.0.0.0:45678",
    tenant_id=None  # No tenant ID for unauthenticated mode
)

# Import configure_remote function to confirm configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from fix_imports import configure_remote


def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)


def import_module(file_path, module_name):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_priority_tests():
    """Run the tests in order of priority."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # First, run the most basic test to verify IndexFlatL2 works
    print_header("Running IndexFlatL2 tests (highest priority)")
    flat_module = import_module(os.path.join(script_dir, "test_index_flat.py"), "test_index_flat")
    flat_suite = unittest.TestLoader().loadTestsFromTestCase(flat_module.TestIndexFlatL2)
    result = unittest.TextTestRunner(verbosity=2).run(flat_suite)
    print(f"\nIndexFlatL2 Tests: {result.testsRun} run, "
          f"{len(result.errors)} errors, {len(result.failures)} failures")

    # Next, try factory tests which should have some working tests
    print_header("Running Factory tests")
    factory_module = import_module(os.path.join(script_dir, "test_factory.py"), "test_factory")
    factory_suite = unittest.TestLoader().loadTestsFromTestCase(factory_module.TestIndexFactory)
    result = unittest.TextTestRunner(verbosity=2).run(factory_suite)
    print(f"\nFactory Tests: {result.testsRun} run, "
          f"{len(result.errors)} errors, {len(result.failures)} failures")

    # Then run the more complex index tests which are mostly skipped
    print_header("Running IndexIVFFlat tests")
    ivf_module = import_module(os.path.join(script_dir, "test_index_ivf_flat.py"), "test_index_ivf_flat")
    ivf_suite = unittest.TestLoader().loadTestsFromTestCase(ivf_module.TestIndexIVFFlat)
    result = unittest.TextTestRunner(verbosity=1).run(ivf_suite)
    print(f"\nIndexIVFFlat Tests: {result.testsRun} run, "
          f"{len(result.errors)} errors, {len(result.failures)} failures")

    print_header("Running IndexIDMap tests")
    idmap_module = import_module(os.path.join(script_dir, "test_index_idmap.py"), "test_index_idmap")
    idmap_suite = unittest.TestLoader().loadTestsFromTestCase(idmap_module.TestIndexIDMap)
    idmap2_suite = unittest.TestLoader().loadTestsFromTestCase(idmap_module.TestIndexIDMap2)
    result = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite([idmap_suite, idmap2_suite]))
    print(f"\nIndexIDMap Tests: {result.testsRun} run, "
          f"{len(result.errors)} errors, {len(result.failures)} failures")

    print_header("Running Persistence tests")
    persistence_module = import_module(os.path.join(script_dir, "test_persistence.py"), "test_persistence")
    persistence_suite = unittest.TestLoader().loadTestsFromTestCase(persistence_module.TestIndexPersistence)
    result = unittest.TextTestRunner(verbosity=1).run(persistence_suite)
    print(f"\nPersistence Tests: {result.testsRun} run, "
          f"{len(result.errors)} errors, {len(result.failures)} failures")


def run_discovery_tests():
    """Run all tests using test discovery."""
    print_header("Running all tests via discovery")
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.defaultTestLoader.discover(start_dir, pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run FAISSx remote tests')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests via discovery')
    args = parser.parse_args()

    start_time = time.time()

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

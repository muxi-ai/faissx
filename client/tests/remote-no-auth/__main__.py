#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Main entry point for running remote mode tests for FAISSx
#
# Copyright (C) 2025 Ran Aroussi

"""
Main module for running FAISSx remote mode tests.

This module allows running the remote mode tests using:
python -m client.tests.remote-no-auth

It automatically discovers and runs all test files in the remote-no-auth directory.
The tests communicate with a remote FAISSx server at 0.0.0.0:45678.
"""

import sys
import os

# Add parent directory to path to import faissx
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(current_dir), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import and configure faissx
import faissx
faissx.configure(
    url="tcp://0.0.0.0:45678",
    tenant_id=None  # No tenant ID for unauthenticated mode
)

# Import run_tests after configuring faissx
from .run_tests import run_tests

if __name__ == "__main__":
    # Run all tests with verbose output
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

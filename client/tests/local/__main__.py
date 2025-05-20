#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Main entry point for running local mode tests for FAISSx
#
# Copyright (C) 2025 Ran Aroussi

"""
Main module for running FAISSx local mode tests.

This module allows running the local mode tests using:
python -m client.tests.local_tests

It automatically discovers and runs all test files in the local_tests directory.
"""

from .run_tests import run_tests
import sys

if __name__ == "__main__":
    # Run all tests with verbose output
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

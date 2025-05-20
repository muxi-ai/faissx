#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Helper module to fix imports and constants for test files
#
# Copyright (C) 2025 Ran Aroussi

"""
This module provides helper constants and functions needed by the test files.
"""

import os
import sys

# Add these constants to make the tests work with the FAISSx implementation
METRIC_L2 = 0  # Same values as in FAISS
METRIC_INNER_PRODUCT = 1

# Ensure the parent directory is in the path to import faissx
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def patch_module(module):
    """
    Patch a module with the necessary constants.

    Args:
        module: The module to patch
    """
    if not hasattr(module, 'METRIC_L2'):
        setattr(module, 'METRIC_L2', METRIC_L2)

    if not hasattr(module, 'METRIC_INNER_PRODUCT'):
        setattr(module, 'METRIC_INNER_PRODUCT', METRIC_INNER_PRODUCT)

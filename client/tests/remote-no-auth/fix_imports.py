#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Import resolution for FAISSx tests
#
# Copyright (C) 2025 Ran Aroussi

"""
Import resolution for FAISSx tests in remote mode.

This module resolves imports for testing by setting up paths and providing
remote connection configuration.
"""

import sys
import os.path

# Metrics constants for FAISS compatibility
METRIC_L2 = 0
METRIC_INNER_PRODUCT = 1

# Add the tests/indices directory to path for custom implementations
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "indices"))

# Add parent directory to path to import faissx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(current_dir), "../../")))
import faissx

# Define exports for other modules to import
__all__ = ['IndexIDMap', 'IndexIDMap2', 'IndexIVFFlat', 'IndexFlatIP', 'write_index', 'read_index', 'configure_remote']

# Import custom implementation of key index types and functions if needed
try:
    # Import index types with implementations in indices directory
    from id_map import IndexIDMap, IndexIDMap2
    from ivf_flat import IndexIVFFlat
    from flat_ip import IndexFlatIP

    # Import persistence functions
    from index_io import write_index, read_index
except ImportError as e:
    print(f"Warning: Could not import test implementation: {e}")


def configure_remote():
    """
    Configure FAISSx client to connect to remote server on 0.0.0.0:45678 without authentication.

    This function should be called at the start of each test to ensure the client
    is properly configured to use the remote server instead of local FAISS.
    """
    # Configure FAISSx to use remote server using explicit configuration
    faissx.configure(
        url="tcp://0.0.0.0:45678",
        tenant_id=None  # No tenant ID for unauthenticated mode
    )


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

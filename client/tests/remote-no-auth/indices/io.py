#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test implementation of index persistence for FAISSx as a drop-in replacement for FAISS
#
# Copyright (C) 2025 Ran Aroussi

"""
Test implementation of index persistence functions for local mode testing.

This module provides test implementations of read_index and write_index functions,
designed to verify that FAISSx works as a drop-in replacement in local mode.
"""

import faiss
from typing import Any


def write_index(index: Any, fname: str) -> None:
    """
    Write an index to a file.

    This test implementation simply delegates to the FAISS implementation,
    extracting the local index from FAISSx wrapper objects when needed.

    Args:
        index: The index to save
        fname: The file name to save to
    """
    # Extract the underlying FAISS index if this is a FAISSx wrapper
    local_index = index._local_index if hasattr(index, '_local_index') else index

    # Use the FAISS implementation to write the index
    faiss.write_index(local_index, fname)


def read_index(fname: str) -> Any:
    """
    Read an index from a file.

    This test implementation simply delegates to the FAISS implementation.
    For test purposes, we return the raw FAISS index without wrapping.

    Args:
        fname: The file name to read from

    Returns:
        The loaded index
    """
    # Use the FAISS implementation to read the index
    return faiss.read_index(fname)

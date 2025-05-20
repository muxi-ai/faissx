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
import numpy as np
from typing import Any, Union
import os
import tempfile


def write_index(index: Any, fname: str) -> None:
    """
    Write an index to a file.

    This test implementation delegates to the FAISS implementation,
    extracting the local index from FAISSx wrapper objects when needed.

    Args:
        index: The index to save
        fname: The file name to save to
    """
    # Handle IndexIDMap and IndexIDMap2 specially
    if hasattr(index, '_id_map') and hasattr(index, '_vectors_by_id') and hasattr(index, 'index'):
        # Save the base index first
        base_index = getattr(index, 'index')
        base_index_local = base_index._local_index if hasattr(base_index, '_local_index') else base_index

        # Save base index to temporary file
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            base_path = tmp.name
            faiss.write_index(base_index_local, base_path)

            # Read the base index as bytes
            with open(base_path, 'rb') as f:
                base_bytes = f.read()

        # Now save the ID mappings and vectors
        id_map = index._id_map
        vectors_by_id = index._vectors_by_id

        # Convert to numpy arrays for storage
        id_list = []
        vector_list = []

        for idx, id_val in id_map.items():
            if id_val in vectors_by_id:
                id_list.append(id_val)
                vector_list.append(vectors_by_id[id_val])

        if vector_list:
            id_array = np.array(id_list, dtype=np.int64)
            vector_array = np.vstack(vector_list).astype(np.float32)

            # Now write everything to the output file
            with open(fname, 'wb') as f:
                # Write a marker to identify this is our custom format
                f.write(b'IDMAP\0\0\0')

                # Write is_idmap2 flag (1 byte)
                is_idmap2 = 1 if hasattr(index, 'replace_vector') else 0
                f.write(bytes([is_idmap2]))

                # Write dimension and count
                dim = index.d
                count = len(id_list)
                f.write(np.array([dim, count], dtype=np.int64).tobytes())

                # Write ID array
                f.write(id_array.tobytes())

                # Write vector array
                f.write(vector_array.tobytes())

                # Write base index
                f.write(len(base_bytes).to_bytes(8, byteorder='little'))
                f.write(base_bytes)
        else:
            # No vectors, just save the base index
            faiss.write_index(base_index_local, fname)

        # Clean up temporary file
        if os.path.exists(base_path):
            os.unlink(base_path)

        return

    # Extract the underlying FAISS index if this is a FAISSx wrapper
    local_index = index._local_index if hasattr(index, '_local_index') else index

    # Use the FAISS implementation to write the index
    faiss.write_index(local_index, fname)


def read_index(fname: str) -> Union[Any, faiss.Index]:
    """
    Read an index from a file.

    This test implementation either uses the FAISS implementation directly
    or reconstructs a special index type depending on the file contents.

    Args:
        fname: The file name to read from

    Returns:
        The loaded index
    """
    if not os.path.exists(fname):
        raise RuntimeError(f"File not found: {fname}")

    # Check if this is our custom format
    with open(fname, 'rb') as f:
        header = f.read(8)
        if header == b'IDMAP\0\0\0':
            # This is our custom format for IndexIDMap

            # Read is_idmap2 flag
            is_idmap2 = f.read(1)[0]

            # Read dimension and count
            dim_count = np.frombuffer(f.read(16), dtype=np.int64)
            dim, count = dim_count[0], dim_count[1]

            # Read ID array
            id_array = np.frombuffer(f.read(count * 8), dtype=np.int64)

            # Read vector array
            vector_array = np.frombuffer(f.read(count * dim * 4),
                                         dtype=np.float32).reshape(count, dim)

            # Read base index size
            base_size = int.from_bytes(f.read(8), byteorder='little')

            # Read base index bytes
            base_bytes = f.read(base_size)

            # Save to temporary file and load with FAISS
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                base_path = tmp.name
                tmp.write(base_bytes)

            try:
                base_index = faiss.read_index(base_path)

                # Import the correct IDMap class
                from id_map import IndexIDMap, IndexIDMap2

                # Create the appropriate index
                IDMapClass = IndexIDMap2 if is_idmap2 else IndexIDMap
                idmap_index = IDMapClass(base_index)

                # Add vectors with IDs
                if count > 0:
                    idmap_index.add_with_ids(vector_array, id_array)
                    # Ensure ntotal matches original count
                    idmap_index.ntotal = count

                return idmap_index
            finally:
                # Clean up
                if os.path.exists(base_path):
                    os.unlink(base_path)

    # If not a custom format, use FAISS directly
    return faiss.read_index(fname)

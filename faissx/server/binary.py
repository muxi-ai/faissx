#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Binary Index Support
# https://github.com/muxi-ai/faissx
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FAISSx Server Binary Index Support

This module provides functionality for handling binary indices and binary vector
operations in the FAISSx server.
"""

import faiss
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

# Binary index types supported by FAISS
BINARY_INDEX_TYPES = {
    "BINARY": faiss.IndexBinaryFlat,
    "BINARY_FLAT": faiss.IndexBinaryFlat,
    "BINARY_IVF": faiss.IndexBinaryIVF,
    "BINARY_HASH": faiss.IndexBinaryHash
}

def is_binary_index_type(index_type: str) -> bool:
    """
    Check if the index type is a binary index.

    Args:
        index_type: Index type string

    Returns:
        bool: True if binary index type, False otherwise
    """
    # Check if the index type starts with "BINARY"
    if index_type.startswith("BINARY"):
        return True

    # Or check if it's in the BINARY_INDEX_TYPES dictionary
    return index_type in BINARY_INDEX_TYPES

def extract_binary_params(index_type: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract parameters from the binary index type string.

    Examples:
        - "BINARY_IVF100" -> ("BINARY_IVF", {"nlist": 100})
        - "BINARY_HASH8" -> ("BINARY_HASH", {"bits_per_dim": 8})

    Args:
        index_type: Index type string with potential parameters

    Returns:
        tuple: (base_index_type, params_dict)
    """
    params = {}
    base_type = index_type

    # Handle BINARY_IVF with nlist parameter
    if index_type.startswith("BINARY_IVF") and len(index_type) > 10:
        try:
            nlist = int(index_type[10:])
            params["nlist"] = nlist
            base_type = "BINARY_IVF"
        except ValueError:
            pass

    # Handle BINARY_HASH with bits_per_dim parameter
    elif index_type.startswith("BINARY_HASH") and len(index_type) > 11:
        try:
            bits_per_dim = int(index_type[11:])
            params["bits_per_dim"] = bits_per_dim
            base_type = "BINARY_HASH"
        except ValueError:
            pass

    return base_type, params

def create_binary_index(index_type: str, dimension: int, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a binary index based on the specified type and dimension.

    Args:
        index_type: Type of binary index
        dimension: Dimension of vectors (in bits)
        **kwargs: Additional parameters for specific index types

    Returns:
        tuple: (index, index_info)

    Raises:
        ValueError: If index type is not supported or parameters are invalid
    """
    # Extract base type and params from index_type string
    base_type, params = extract_binary_params(index_type)

    # Merge with provided kwargs
    params.update(kwargs)

    # Calculate dimension in bytes (8 bits per byte)
    dimension_bytes = (dimension + 7) // 8

    if base_type not in BINARY_INDEX_TYPES:
        raise ValueError(f"Unsupported binary index type: {base_type}")

    index_info = {
        "type": base_type,
        "dimension": dimension,
        "dimension_bytes": dimension_bytes,
        "is_binary": True
    }

    # Create the appropriate binary index
    if base_type == "BINARY" or base_type == "BINARY_FLAT":
        index = faiss.IndexBinaryFlat(dimension)

    elif base_type == "BINARY_IVF":
        nlist = params.get("nlist", 100)  # Default to 100 clusters
        quantizer = faiss.IndexBinaryFlat(dimension)
        index = faiss.IndexBinaryIVF(quantizer, dimension, nlist)
        index_info["requires_training"] = True
        index_info["nlist"] = nlist

    elif base_type == "BINARY_HASH":
        bits_per_dim = params.get("bits_per_dim", 8)  # Default to 8 bits per dimension
        index = faiss.IndexBinaryHash(dimension, bits_per_dim)
        index_info["bits_per_dim"] = bits_per_dim

    return index, index_info

def convert_to_binary(vectors: List[List[float]]) -> np.ndarray:
    """
    Convert floating point vectors to binary vectors.

    Args:
        vectors: List of vectors (each with float values 0.0 or 1.0)

    Returns:
        numpy.ndarray: Binary vectors as uint8 array
    """
    # Convert to numpy array
    vectors_np = np.array(vectors, dtype=np.float32)

    # Convert to boolean array
    bool_array = vectors_np > 0.5

    # Get the number of vectors and the dimension
    n_vectors = len(vectors)
    dimension = len(vectors[0])

    # Calculate bytes needed (8 bits per byte)
    n_bytes = (dimension + 7) // 8

    # Initialize output array
    binary_vectors = np.zeros((n_vectors, n_bytes), dtype=np.uint8)

    # Pack the bits into bytes
    for i in range(n_vectors):
        for j in range(dimension):
            if j < bool_array.shape[1]:  # Ensure we don't exceed input dimensions
                byte_idx = j // 8
                bit_idx = j % 8
                if bool_array[i, j]:
                    binary_vectors[i, byte_idx] |= (1 << bit_idx)

    return binary_vectors

def binary_to_float(binary_vectors: np.ndarray, dimension: int) -> List[List[float]]:
    """
    Convert binary vectors back to floating point vectors.

    Args:
        binary_vectors: Binary vectors as uint8 array
        dimension: Original dimension in bits

    Returns:
        list: List of vectors with float values (0.0 or 1.0)
    """
    # Get the number of vectors
    n_vectors = binary_vectors.shape[0]

    # Initialize output array
    float_vectors = np.zeros((n_vectors, dimension), dtype=np.float32)

    # Unpack the bits from bytes
    for i in range(n_vectors):
        for j in range(dimension):
            byte_idx = j // 8
            bit_idx = j % 8

            # Check if the bit is set
            if byte_idx < binary_vectors.shape[1]:
                bit_value = (binary_vectors[i, byte_idx] >> bit_idx) & 1
                float_vectors[i, j] = float(bit_value)

    return float_vectors.tolist()

def compute_hamming_distance(vector1: List[float], vector2: List[float]) -> int:
    """
    Compute Hamming distance between two binary vectors.

    Args:
        vector1: First binary vector (as list of 0.0 and 1.0)
        vector2: Second binary vector (as list of 0.0 and 1.0)

    Returns:
        int: Hamming distance
    """
    v1 = np.array(vector1, dtype=np.bool_)
    v2 = np.array(vector2, dtype=np.bool_)

    return np.sum(v1 != v2)

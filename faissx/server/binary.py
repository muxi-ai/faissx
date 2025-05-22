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
import logging

logger = logging.getLogger("faissx.server")

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
    # First check if the input is a string
    if not isinstance(index_type, str):
        return False

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
            if byte_idx < binary_vectors.shape[1]:  # Ensure we don't exceed binary_vectors dimensions
                if binary_vectors[i, byte_idx] & (1 << bit_idx):
                    float_vectors[i, j] = 1.0

    return float_vectors.tolist()

def compute_hamming_distance(vector1: List[float], vector2: List[float]) -> int:
    """
    Compute Hamming distance between two vectors.

    Args:
        vector1: First vector (list of floats, treated as binary)
        vector2: Second vector (list of floats, treated as binary)

    Returns:
        int: Hamming distance (number of differing bits)
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")

    distance = 0
    for i in range(len(vector1)):
        if (vector1[i] > 0.5) != (vector2[i] > 0.5):
            distance += 1

    return distance

def create_binary_idmap_index(dimension: int, index_type: str = "BINARY_FLAT") -> Tuple[Any, Dict[str, Any]]:
    """
    Create a binary IDMap index based on the specified type and dimension.

    This enables ID mapping for binary indices, allowing retrieval by custom IDs.

    Args:
        dimension: Dimension of vectors (in bits)
        index_type: Type of base binary index

    Returns:
        tuple: (index, index_info)

    Raises:
        ValueError: If index type is not supported or parameters are invalid
    """
    # Create the base binary index
    base_index, base_info = create_binary_index(index_type, dimension)

    # Create IDMap wrapper
    index = faiss.IndexBinaryIDMap(base_index)

    # Prepare index info
    index_info = {
        "type": f"IDMap:{base_info['type']}",
        "dimension": dimension,
        "dimension_bytes": base_info.get('dimension_bytes', (dimension + 7) // 8),
        "is_binary": True,
        "is_idmap": True,
        "base_type": base_info['type']
    }

    return index, index_info

def create_binary_idmap2_index(dimension: int, index_type: str = "BINARY_FLAT") -> Tuple[Any, Dict[str, Any]]:
    """
    Create a binary IDMap2 index based on the specified type and dimension.

    IDMap2 indices support faster random access and removal operations compared to IDMap.

    Args:
        dimension: Dimension of vectors (in bits)
        index_type: Type of base binary index

    Returns:
        tuple: (index, index_info)

    Raises:
        ValueError: If index type is not supported or parameters are invalid
    """
    # Create the base binary index
    base_index, base_info = create_binary_index(index_type, dimension)

    # Create IDMap2 wrapper
    index = faiss.IndexBinaryIDMap2(base_index)

    # Prepare index info
    index_info = {
        "type": f"IDMap2:{base_info['type']}",
        "dimension": dimension,
        "dimension_bytes": base_info.get('dimension_bytes', (dimension + 7) // 8),
        "is_binary": True,
        "is_idmap": True,
        "is_idmap2": True,
        "base_type": base_info['type']
    }

    return index, index_info

def compute_binary_distance_matrix(binary_vectors: np.ndarray) -> np.ndarray:
    """
    Compute a distance matrix between all pairs of binary vectors.

    This is useful for clustering or visualization of binary vectors.

    Args:
        binary_vectors: Binary vectors as uint8 array

    Returns:
        numpy.ndarray: Distance matrix (shape: n_vectors x n_vectors)
    """
    n_vectors = binary_vectors.shape[0]
    distance_matrix = np.zeros((n_vectors, n_vectors), dtype=np.int32)

    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            # Compute Hamming distance using bitwise operations
            xor_result = np.bitwise_xor(binary_vectors[i], binary_vectors[j])

            # Count number of set bits
            distance = 0
            for byte in xor_result:
                # Count bits using popcount
                distance += bin(byte).count('1')

            # Store the results (matrix is symmetric)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def get_binary_index_supported_parameters(index_type: str) -> List[str]:
    """
    Get a list of supported parameters for a given binary index type.

    Args:
        index_type: Type of binary index

    Returns:
        list: List of parameter names supported by this index type
    """
    supported_params = ["dimension"]

    if index_type == "BINARY_FLAT":
        pass  # No additional parameters for BINARY_FLAT

    elif index_type.startswith("BINARY_IVF"):
        supported_params.extend(["nlist", "nprobe"])

    elif index_type.startswith("BINARY_HASH"):
        supported_params.append("bits_per_dim")

    # Parameters for IDMap wrappers
    if "IDMap" in index_type:
        supported_params.extend(["add_with_ids", "remove_ids"])

    return supported_params

def optimize_binary_index(index: Any, optimization_level: int = 1) -> bool:
    """
    Optimize a binary index for better performance.

    This function applies various optimizations based on the level.

    Args:
        index: Binary index to optimize
        optimization_level: Level of optimization (1-3, higher is more aggressive)

    Returns:
        bool: True if optimization was applied successfully
    """
    try:
        if isinstance(index, faiss.IndexBinaryIVF):
            # Optimize IVF binary index
            if optimization_level >= 1:
                # Basic optimization - set nprobe to a reasonable value based on nlist
                nlist = index.nlist
                index.nprobe = max(1, min(nlist // 10, 32))  # 10% of nlist but at most 32

            if optimization_level >= 2:
                # Precompute some data structures for faster search (if available in the future)
                pass

        elif isinstance(index, faiss.IndexBinaryHash):
            # Currently no specific optimizations for binary hash
            pass

        return True

    except Exception as e:
        logger.warning(f"Error optimizing binary index: {e}")
        return False

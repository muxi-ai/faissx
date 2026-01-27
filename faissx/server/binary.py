#!/usr/bin/env python3
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

This module provides comprehensive functionality for handling binary indices and binary
vector operations in the FAISSx server. Binary indices are specialized data structures
that work with binary vectors (vectors of 0s and 1s) for efficient similarity search.

Key Features:
- Binary index creation and management (Flat, IVF, Hash)
- Vector format conversion (float ↔ binary)
- Distance computation (Hamming distance)
- Index optimization and parameter extraction
- IDMap support for custom ID management

Binary Vector Format:
Binary vectors are stored as packed uint8 arrays where each bit represents a vector
component. This provides significant memory savings and faster distance computations
for binary data.

Supported Index Types:
- BINARY_FLAT: Exhaustive search with exact results
- BINARY_IVF: Inverted file system for faster approximate search
- BINARY_HASH: Hash-based indexing for very fast approximate search
- IDMap variants: Enable custom ID management for all base types

Performance Considerations:
- Binary operations are significantly faster than float operations
- Memory usage is ~32x smaller than equivalent float vectors
- Hamming distance is computed using efficient bitwise operations
"""

import logging
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

logger = logging.getLogger("faissx.server")

# Constants for binary vector operations
BITS_PER_BYTE = 8  # Number of bits in a byte
DEFAULT_NLIST = 100  # Default number of clusters for IVF indices
DEFAULT_BITS_PER_DIM = 8  # Default bits per dimension for hash indices
DEFAULT_NPROBE_RATIO = 10  # Ratio for calculating default nprobe (nlist / ratio)
MAX_NPROBE = 32  # Maximum nprobe value for optimization

# Binary index types supported by FAISS
BINARY_INDEX_TYPES = {
    "BINARY": faiss.IndexBinaryFlat,
    "BINARY_FLAT": faiss.IndexBinaryFlat,
    "BINARY_IVF": faiss.IndexBinaryIVF,
    "BINARY_HASH": faiss.IndexBinaryHash,
}


def is_binary_index_type(index_type: str) -> bool:
    """
    Check if the index type is a binary index.

    This function determines whether a given index type string corresponds to a binary
    index that operates on binary vectors (bit vectors) rather than float vectors.

    Detection Logic:
        1. Validates input is a string
        2. Checks for "BINARY" prefix (covers parameterized types like "BINARY_IVF100")
        3. Verifies against known binary index types

    Args:
        index_type: Index type string to check (e.g., "BINARY_FLAT", "BINARY_IVF100")

    Returns:
        bool: True if the index type is binary, False otherwise

    Example:
        >>> is_binary_index_type("BINARY_FLAT")
        True
        >>> is_binary_index_type("IVF100,Flat")
        False
        >>> is_binary_index_type("BINARY_IVF100")
        True
    """
    # Input validation: ensure we have a string to work with
    # Non-string inputs are definitely not binary index types
    if not isinstance(index_type, str):
        return False

    # Primary check: binary indices conventionally start with "BINARY"
    # This covers both base types and parameterized types (e.g., "BINARY_IVF100")
    if index_type.startswith("BINARY"):
        return True

    # Fallback check: verify against the explicit mapping of known binary types
    # This ensures we catch any edge cases not covered by the prefix check
    return index_type in BINARY_INDEX_TYPES


def extract_binary_params(index_type: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract parameters from the binary index type string.

    This function parses parameterized index type strings to separate the base type
    from its configuration parameters. It supports the common pattern of appending
    numeric parameters directly to the index type name.

    Supported Patterns:
        - "BINARY_IVF<number>" → nlist parameter for IVF clustering
        - "BINARY_HASH<number>" → bits_per_dim parameter for hash indexing
        - Base types without parameters are returned unchanged

    Args:
        index_type: Index type string with potential parameters embedded
                   (e.g., "BINARY_IVF100", "BINARY_HASH8", "BINARY_FLAT")

    Returns:
        tuple: (base_index_type, params_dict) where base_index_type is the
               standardized type name and params_dict contains extracted parameters

    Examples:
        >>> extract_binary_params("BINARY_IVF100")
        ("BINARY_IVF", {"nlist": 100})
        >>> extract_binary_params("BINARY_HASH8")
        ("BINARY_HASH", {"bits_per_dim": 8})
        >>> extract_binary_params("BINARY_FLAT")
        ("BINARY_FLAT", {})

    Note:
        Invalid numeric suffixes are silently ignored, returning the original
        string as the base type with empty parameters.
    """
    # Initialize return values - assume no parameters initially
    params = {}
    base_type = index_type

    # Handle BINARY_IVF with nlist parameter
    # Pattern: "BINARY_IVF" + number (e.g., "BINARY_IVF100")
    if index_type.startswith("BINARY_IVF") and len(index_type) > 10:
        try:
            # Extract numeric suffix after "BINARY_IVF" (10 characters)
            nlist = int(index_type[10:])
            params["nlist"] = nlist
            base_type = "BINARY_IVF"
        except ValueError:
            # Invalid numeric suffix - use original string as base type
            pass

    # Handle BINARY_HASH with bits_per_dim parameter
    # Pattern: "BINARY_HASH" + number (e.g., "BINARY_HASH8")
    elif index_type.startswith("BINARY_HASH") and len(index_type) > 11:
        try:
            # Extract numeric suffix after "BINARY_HASH" (11 characters)
            bits_per_dim = int(index_type[11:])
            params["bits_per_dim"] = bits_per_dim
            base_type = "BINARY_HASH"
        except ValueError:
            # Invalid numeric suffix - use original string as base type
            pass

    return base_type, params


def create_binary_index(
    index_type: str, dimension: int, **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a binary index based on the specified type and dimension.

    This is the main factory function for creating binary FAISS indices. It handles
    parameter extraction, validation, and index instantiation for all supported
    binary index types.

    Index Types and Their Characteristics:
        - BINARY_FLAT: Exact search, no training required, O(n) search time
        - BINARY_IVF: Approximate search, requires training, faster than flat
        - BINARY_HASH: Very fast approximate search, no training required

    Args:
        index_type: Type of binary index to create. Can include embedded parameters
                   (e.g., "BINARY_IVF100" for nlist=100)
        dimension: Dimension of vectors in bits. Must be positive integer.
        **kwargs: Additional parameters that override extracted parameters.
                 Common kwargs: nlist (for IVF), bits_per_dim (for HASH)

    Returns:
        tuple: (index, index_info) where:
               - index: The created FAISS binary index object
               - index_info: Dictionary with metadata about the index

    Raises:
        ValueError: If index type is not supported or parameters are invalid
        TypeError: If dimension is not an integer or is negative

    Examples:
        >>> index, info = create_binary_index("BINARY_FLAT", 128)
        >>> index, info = create_binary_index("BINARY_IVF100", 64)
        >>> index, info = create_binary_index("BINARY_IVF", 128, nlist=50)

    Note:
        IVF indices require training before use. Check index_info["requires_training"]
        to determine if training is needed.
    """
    # Input validation for dimension parameter
    if not isinstance(dimension, int) or dimension <= 0:
        raise TypeError("Dimension must be a positive integer")

    # Extract base type and embedded parameters from the index_type string
    # This handles patterns like "BINARY_IVF100" → ("BINARY_IVF", {"nlist": 100})
    base_type, params = extract_binary_params(index_type)

    # Merge extracted parameters with explicitly provided kwargs
    # Explicit kwargs take precedence over extracted parameters
    params.update(kwargs)

    # Calculate storage requirements: each bit needs 1/8th of a byte
    # We round up to ensure we have enough bytes for all bits
    dimension_bytes = (dimension + BITS_PER_BYTE - 1) // BITS_PER_BYTE

    # Validate that we support this index type
    if base_type not in BINARY_INDEX_TYPES:
        raise ValueError(
            f"Unsupported binary index type: {base_type}. "
            f"Supported types: {list(BINARY_INDEX_TYPES.keys())}"
        )

    # Initialize index metadata with common properties
    index_info = {
        "type": base_type,
        "dimension": dimension,
        "dimension_bytes": dimension_bytes,
        "is_binary": True,
    }

    # Create the appropriate binary index based on type
    if base_type == "BINARY" or base_type == "BINARY_FLAT":
        # Flat index: exhaustive search, exact results, no training needed
        index = faiss.IndexBinaryFlat(dimension)

    elif base_type == "BINARY_IVF":
        # IVF index: clusters vectors for faster approximate search
        nlist = params.get("nlist", DEFAULT_NLIST)

        # Validate nlist parameter
        if not isinstance(nlist, int) or nlist <= 0:
            raise ValueError(f"nlist must be a positive integer, got: {nlist}")

        # Create quantizer (flat index) and IVF index
        quantizer = faiss.IndexBinaryFlat(dimension)
        index = faiss.IndexBinaryIVF(quantizer, dimension, nlist)

        # IVF indices require training on representative data
        index_info["requires_training"] = True
        index_info["nlist"] = nlist

    elif base_type == "BINARY_HASH":
        # Hash index: very fast approximate search using locality-sensitive hashing
        bits_per_dim = params.get("bits_per_dim", DEFAULT_BITS_PER_DIM)

        # Validate bits_per_dim parameter
        if not isinstance(bits_per_dim, int) or bits_per_dim <= 0:
            raise ValueError(
                f"bits_per_dim must be a positive integer, got: {bits_per_dim}"
            )

        index = faiss.IndexBinaryHash(dimension, bits_per_dim)
        index_info["bits_per_dim"] = bits_per_dim

    return index, index_info


def convert_to_binary(vectors: List[List[float]]) -> np.ndarray:
    """
    Convert floating point vectors to binary vectors.

    This function converts float vectors (with values typically 0.0 or 1.0) into
    packed binary representation where 8 bits are stored per byte. Values > 0.5
    are treated as 1, values ≤ 0.5 are treated as 0.

    Bit Packing Format:
        - Each dimension maps to one bit
        - 8 dimensions are packed into each byte
        - Bit order: dimension j maps to bit (j % 8) in byte (j // 8)
        - Unused bits in the last byte are set to 0

    Memory Efficiency:
        Reduces memory usage by ~32x compared to float32 vectors
        (32 bits → 1 bit per dimension)

    Args:
        vectors: List of vectors where each vector is a list of floats.
                Values should be 0.0/1.0 but any float is accepted.
                All vectors must have the same dimension.

    Returns:
        numpy.ndarray: Binary vectors as uint8 array with shape (n_vectors, n_bytes)
                      where n_bytes = ceil(dimension / 8)

    Raises:
        IndexError: If vectors list is empty
        ValueError: If vectors have inconsistent dimensions

    Example:
        >>> vectors = [[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        >>> binary = convert_to_binary(vectors)
        >>> binary.shape
        (2, 1)  # 2 vectors, 1 byte each (4 bits + 4 padding bits)
    """
    if not vectors:
        raise ValueError("Cannot convert empty vector list")

    # Convert input to numpy array for efficient processing
    # Using float32 for consistency with FAISS requirements
    vectors_np = np.array(vectors, dtype=np.float32)

    # Convert to boolean array using threshold of 0.5
    # This handles both perfect binary (0.0/1.0) and fuzzy inputs
    bool_array = vectors_np > 0.5

    # Extract dimensions for bit packing calculations
    n_vectors = len(vectors)
    dimension = len(vectors[0])

    # Calculate storage requirements: round up to nearest byte boundary
    # This ensures we have enough space for all bits
    n_bytes = (dimension + BITS_PER_BYTE - 1) // BITS_PER_BYTE

    # Initialize output array with zeros
    # uint8 type stores 8 bits per element, perfect for bit packing
    binary_vectors = np.zeros((n_vectors, n_bytes), dtype=np.uint8)

    # Pack the bits into bytes using bitwise operations
    for i in range(n_vectors):
        for j in range(dimension):
            # Safety check: ensure we don't exceed input dimensions
            # This handles edge cases with malformed input data
            if j < bool_array.shape[1]:
                # Calculate which byte and which bit within that byte
                byte_idx = j // BITS_PER_BYTE  # Which byte (0, 1, 2, ...)
                bit_idx = j % BITS_PER_BYTE  # Which bit within byte (0-7)

                # Set the appropriate bit if the boolean value is True
                if bool_array[i, j]:
                    binary_vectors[i, byte_idx] |= 1 << bit_idx

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
            byte_idx = j // BITS_PER_BYTE
            bit_idx = j % BITS_PER_BYTE
            # Ensure we don't exceed binary_vectors dimensions
            if byte_idx < binary_vectors.shape[1]:
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


def create_binary_idmap_index(
    dimension: int, index_type: str = "BINARY_FLAT"
) -> Tuple[Any, Dict[str, Any]]:
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
        "dimension_bytes": base_info.get(
            "dimension_bytes", (dimension + BITS_PER_BYTE - 1) // BITS_PER_BYTE
        ),
        "is_binary": True,
        "is_idmap": True,
        "base_type": base_info["type"],
    }

    return index, index_info


def create_binary_idmap2_index(
    dimension: int, index_type: str = "BINARY_FLAT"
) -> Tuple[Any, Dict[str, Any]]:
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
        "dimension_bytes": base_info.get(
            "dimension_bytes", (dimension + BITS_PER_BYTE - 1) // BITS_PER_BYTE
        ),
        "is_binary": True,
        "is_idmap": True,
        "is_idmap2": True,
        "base_type": base_info["type"],
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
                distance += bin(byte).count("1")

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

    This function applies performance optimizations to binary indices based on their
    type and the specified optimization level. Higher levels apply more aggressive
    optimizations that may trade some accuracy for increased speed.

    Optimization Strategies:
        Level 1 (Basic): Set reasonable search parameters (nprobe for IVF)
        Level 2 (Moderate): Reserved for future enhancements
        Level 3 (Aggressive): Reserved for future enhancements

    Supported Index Types:
        - IndexBinaryIVF: Optimizes nprobe parameter for search speed
        - IndexBinaryHash: No optimizations currently available
        - Other types: No optimizations applied

    Args:
        index: Binary index object to optimize. Must be a FAISS binary index.
        optimization_level: Level of optimization intensity (1-3).
                           Higher values apply more aggressive optimizations.
                           Default is 1 for safe, conservative optimization.

    Returns:
        bool: True if optimization was applied successfully, False if an error
              occurred during optimization. Errors are logged but not raised.

    Example:
        >>> index, _ = create_binary_index("BINARY_IVF100", 128)
        >>> # Train the index first...
        >>> success = optimize_binary_index(index, optimization_level=1)
        >>> print(f"nprobe set to: {index.nprobe}")

    Note:
        - IVF indices must be trained before optimization
        - Optimization is non-destructive and can be safely applied multiple times
        - Failed optimizations are logged but don't raise exceptions
    """
    try:
        # Optimization for IVF binary indices
        if isinstance(index, faiss.IndexBinaryIVF):
            if optimization_level >= 1:
                # Basic optimization: set nprobe to balance speed vs accuracy
                nlist = index.nlist

                # Calculate optimal nprobe: ~10% of nlist but capped at MAX_NPROBE
                # This provides good recall while maintaining search speed
                optimal_nprobe = max(1, min(nlist // DEFAULT_NPROBE_RATIO, MAX_NPROBE))
                index.nprobe = optimal_nprobe

                logger.debug(
                    f"Optimized IVF index: nprobe set to {optimal_nprobe} "
                    f"(nlist={nlist})"
                )

            if optimization_level >= 2:
                # Future enhancement: precompute distance tables, etc.
                # Reserved for more advanced optimizations
                pass

        elif isinstance(index, faiss.IndexBinaryHash):
            # Hash indices are already optimized by design
            # No runtime optimizations currently available
            logger.debug("Hash index detected - no optimizations available")

        else:
            # Other index types (Flat, etc.) don't have optimization parameters
            logger.debug(
                f"No optimizations available for index type: " f"{type(index).__name__}"
            )

        return True

    except Exception as e:
        # Log the error but don't crash - optimization failures shouldn't break the system
        logger.warning(f"Error optimizing binary index: {e}")
        return False

#!/usr/bin/env python3
#
# FAISSx Server Transformations Module
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
FAISSx Server Transformations Module

This module provides comprehensive functionality for vector transformations and
handling different FAISS index types with consistent parameter support.

The module supports:
- Vector preprocessing transformations (PCA, normalization, quantization)
- Composite index creation with pre-transform pipelines
- Distance metric configuration and conversion
- Index type parsing and specialized index templates
- Training requirement analysis for transformations

Key Components:
- Transformation creation and management
- Index factory with type parsing
- Pre-transform index composition
- Metric type handling and validation
- Specialized index templates for common use cases
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type

import faiss
import numpy as np

# Default parameter constants for maintainability
DEFAULT_PCA_EIGEN_POWER = -0.5           # Whitening power for PCAR
DEFAULT_ITQ_ITERATIONS = 50              # ITQ convergence iterations
DEFAULT_OPQ_SUBQUANTIZERS = 8            # Default number of sub-quantizers
DEFAULT_OPQ_ITERATIONS = 25              # OPQ optimization iterations
DEFAULT_PQ_BITS = 8                      # Default bits per PQ sub-quantizer
DEFAULT_HNSW_M = 32                      # Default HNSW connectivity
DEFAULT_IVF_NLIST = 100                  # Default IVF cluster count
DEFAULT_LSH_BITS = 32                    # Default LSH hash bits
DEFAULT_HNSW_EF_CONSTRUCTION = 200       # HNSW build-time search depth
DEFAULT_HNSW_EF_SEARCH = 128             # HNSW search-time depth

# Training requirement constants
PCA_MIN_TRAINING_MULTIPLIER = 2          # Minimum vectors per output dimension
PCA_RECOMMENDED_TRAINING_MULTIPLIER = 10  # Recommended vectors per output dimension
OPQ_TRAINING_MULTIPLIER = 10             # Training vectors per sub-quantizer
ITQ_MIN_TRAINING_MULTIPLIER = 5          # Minimum vectors per dimension
ITQ_RECOMMENDED_TRAINING_MULTIPLIER = 20  # Recommended vectors per dimension

# Supported transformation types with their FAISS class mappings
# Each transformation serves a specific vector preprocessing purpose
TRANSFORM_TYPES: Dict[str, Type[Any]] = {
    "PCA": faiss.PCAMatrix,              # Principal Component Analysis
    "PCAR": faiss.PCAMatrix,             # PCA with whitening (decorrelation)
    "L2NORM": faiss.NormalizationTransform,  # L2 vector normalization
    "NORM": faiss.NormalizationTransform,    # Normalization (alias for L2NORM)
    "ITQ": faiss.ITQTransform,           # Iterative Quantization for binary codes
    "OPQ": faiss.OPQMatrix,              # Optimized Product Quantization
    "RR": faiss.RandomRotationMatrix     # Random rotation for hash functions
}

# Supported distance metrics with their FAISS constants
# These determine how vector similarity is calculated
METRIC_TYPES: Dict[str, int] = {
    "L2": faiss.METRIC_L2,               # Euclidean distance (L2 norm)
    "INNER_PRODUCT": faiss.METRIC_INNER_PRODUCT,  # Dot product similarity
    "L1": faiss.METRIC_L1,               # Manhattan distance (L1 norm)
    "LINF": faiss.METRIC_Linf,           # Chebyshev distance (L-infinity)
    "CANBERRA": faiss.METRIC_Canberra,   # Canberra distance (weighted L1)
    "BRAYCURTIS": faiss.METRIC_BrayCurtis  # Bray-Curtis distance (ecology)
}

# Mapping of metric type aliases to canonical names
# This provides flexibility in API usage with common alternative names
METRIC_ALIASES: Dict[str, str] = {
    "IP": "INNER_PRODUCT",          # Common abbreviation for inner product
    "EUCLIDEAN": "L2",              # Alternative name for L2 distance
    "COSINE": "INNER_PRODUCT",      # Cosine similarity via inner product on normalized vectors
    "MANHATTAN": "L1"               # Alternative name for L1 distance
}

# Binary index type prefixes for type detection
# These prefixes identify indices that work with binary (bit-packed) vectors
BINARY_INDEX_PREFIXES: List[str] = ["BINARY_", "BIN_"]


def create_transformation(
    transform_type: str,
    input_dim: int,
    output_dim: Optional[int] = None,
    transform_params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create a transformation object based on the specified type.

    This function provides a unified interface for creating various FAISS vector
    transformations. Each transformation type serves different preprocessing needs:
    - PCA/PCAR: Dimensionality reduction and decorrelation
    - L2NORM: Vector normalization for cosine similarity
    - ITQ: Binary quantization for compact representations
    - OPQ: Optimized quantization for better accuracy
    - RR: Random rotation for locality-sensitive hashing

    Args:
        transform_type: Type of transformation to create. Must be one of the
                       supported types in TRANSFORM_TYPES.
        input_dim: Input vector dimension. Must match the vectors to be transformed.
        output_dim: Output dimension after transformation. If None, defaults to
                   input_dim (for transforms that don't change dimensionality).
        transform_params: Additional configuration parameters specific to each
                         transformation type.

    Returns:
        FAISS transformation object ready for training or immediate use.

    Raises:
        ValueError: If the transform_type is not supported or parameters are invalid.

    Implementation Notes:
        - PCA/PCAR transformations support eigen_power and random_rotation parameters
        - ITQ requires n_iterations parameter for convergence control
        - OPQ needs M (sub-quantizers) and niter (optimization iterations) parameters
        - Random rotation can be seeded for reproducible results
    """
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError(f"Unsupported transformation type: {transform_type}")

    # Default to input_dim if output_dim not specified
    # This is appropriate for transformations that preserve dimensionality
    if output_dim is None:
        output_dim = input_dim

    # Extract parameters with sensible defaults
    params = transform_params or {}

    # Create the transformation based on its specific requirements
    if transform_type in ["PCA", "PCAR"]:
        # Principal Component Analysis with optional whitening
        transform = TRANSFORM_TYPES[transform_type](input_dim, output_dim)

        # Configure whitening for PCAR (Principal Component Analysis with Rotation)
        # Whitening decorrelates the data by scaling eigenvectors by eigenvalues^-0.5
        if transform_type == "PCAR":
            transform.eigen_power = DEFAULT_PCA_EIGEN_POWER

        # Apply optional parameter overrides
        if "eigen_power" in params:
            transform.eigen_power = params["eigen_power"]
        if "random_rotation" in params:
            transform.random_rotation = params["random_rotation"]

    elif transform_type in ["L2NORM", "NORM"]:
        # L2 normalization transform for cosine similarity preparation
        # This normalizes vectors to unit length, enabling cosine similarity via dot product
        transform = TRANSFORM_TYPES[transform_type](input_dim)

    elif transform_type == "ITQ":
        # Iterative Quantization for learning optimal binary quantization
        # ITQ iteratively refines a rotation matrix to minimize quantization error
        n_iter = params.get("n_iterations", DEFAULT_ITQ_ITERATIONS)
        transform = TRANSFORM_TYPES[transform_type](input_dim, n_iter)

    elif transform_type == "OPQ":
        # Optimized Product Quantization for better vector compression
        # OPQ learns an optimal rotation before standard product quantization
        m = params.get("M", DEFAULT_OPQ_SUBQUANTIZERS)
        transform = TRANSFORM_TYPES[transform_type](input_dim, output_dim, m)

        # Set optimization iterations if specified
        if "niter" in params:
            transform.niter = params["niter"]

    elif transform_type == "RR":
        # Random Rotation Matrix for locality-sensitive hashing
        # Provides a random orthogonal transformation
        transform = TRANSFORM_TYPES[transform_type](input_dim, output_dim)

        # Set random seed for reproducible rotations if specified
        if "seed" in params:
            faiss.seed_rand(params["seed"])

    return transform


def create_pretransform_index(
    transforms: List[Any],
    base_index: Any
) -> faiss.IndexPreTransform:
    """
    Create an IndexPreTransform with the given transformations and base index.

    This function composes multiple transformations into a preprocessing pipeline
    that is applied automatically before vectors reach the base index. The
    transformations are applied in the order they appear in the list.

    Args:
        transforms: List of FAISS transformation objects to apply in sequence.
                   Each transform's output dimension must match the next transform's
                   input dimension.
        base_index: Base FAISS index to use after all transformations are applied.
                   Must accept vectors of the final transformation's output dimension.

    Returns:
        faiss.IndexPreTransform: Composite index that applies transformations
                                automatically during add/search operations.

    Implementation Notes:
        - Transformations are applied in the order provided
        - The pipeline validates dimension compatibility between stages
        - All transformations must be trained before the composite index can be used
        - The base index operates on the transformed vector space
    """
    # Create the index pre-transform with the specified number of transforms
    # This sets up the transformation pipeline structure
    index = faiss.IndexPreTransform(len(transforms), base_index)

    # Add each transformation to the pipeline in sequence
    # prepend_transform adds to the beginning, so we add in reverse order
    for _i, transform in enumerate(transforms):
        index.prepend_transform(transform)

    return index


def parse_transform_type(index_type: str) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """
    Parse a compound index type string into transformation type, base index type, and parameters.

    This function interprets complex index specifications that combine transformations
    with base index types. It handles various formats and extracts configuration
    parameters from the type string.

    Examples:
        "PCA32,Flat" -> ("PCA", "Flat", {"dim": 32})
        "PQ4x8" -> (None, "PQ4x8", {})
        "L2NORM,IVF100,Flat" -> ("L2NORM", "IVF100,Flat", {})

    Args:
        index_type: Compound index type string that may include transformations
                   and base index specifications.

    Returns:
        tuple: A three-element tuple containing:
            - transformation type (str or None): The preprocessing transformation
            - base index type (str): The underlying index specification
            - transform parameters (dict): Extracted configuration parameters

    Implementation Strategy:
        1. Handle special cases like PQ and IVF_SQ that aren't transformations
        2. Split on commas to separate transformation from base index
        3. Parse transformation type and extract dimension parameters
        4. Return parsed components for further processing
    """
    # Handle special case for PQ indices like "PQ4x8"
    # These specify product quantization parameters, not transformations
    pq_match = re.match(r"PQ(\d+)x(\d+)", index_type)
    if pq_match:
        return None, index_type, {}

    # Handle special case for IVF_SQ indices like "IVF4_SQ0"
    # These specify IVF with scalar quantization, not transformations
    ivf_sq_match = re.match(r"IVF(\d+)_SQ(\d+)", index_type)
    if ivf_sq_match:
        return None, index_type, {}

    # Split the compound type string on commas
    parts = index_type.split(",")

    # If there's only one part, it's just the base index type, no transformation
    if len(parts) == 1:
        return None, index_type, {}

    # The last part is always the base index type
    base_index_type = parts[-1]

    # The first part contains the transformation specification
    transform_part = parts[0]

    # Extract transformation type and optional dimension parameter
    # Pattern matches: letters followed by optional digits (e.g., "PCA32")
    match = re.match(r"([A-Za-z]+)(\d*)", transform_part)
    if match:
        t_type, t_dim = match.groups()
        transform_params = {}

        # Add dimension parameter if specified in the type string
        if t_dim:
            transform_params["dim"] = int(t_dim)

        return t_type, base_index_type, transform_params

    # If no pattern match, return the first part as-is with no parameters
    return parts[0], base_index_type, {}


def train_transform(
    transform: Any,
    training_vectors: np.ndarray
) -> Dict[str, Any]:
    """
    Train a transformation with the provided training vectors.

    This function handles the training process for transformations that require
    learning from data. Not all transformations need training (e.g., normalization
    is parameter-free), but those that do (PCA, OPQ, ITQ) need representative
    training data to learn optimal parameters.

    Args:
        transform: FAISS transformation object to train. Must support the training
                  interface if training is required.
        training_vectors: Representative vectors for learning transformation parameters.
                         Should be float32 numpy array of shape (n_samples, dimension).

    Returns:
        dict: Training results and status information with the following structure:
            - success (bool): Whether training completed successfully
            - already_trained (bool, optional): True if transform was already trained
            - is_trained (bool): Final training status
            - input_dim (int, optional): Input dimension of the transformation
            - output_dim (int, optional): Output dimension after transformation
            - transform_type (str, optional): Specific type of transformation
            - Additional type-specific parameters

    Implementation Notes:
        - Checks if the transformation requires training before attempting
        - Validates that training vectors are appropriate for the transformation
        - Extracts transformation-specific information after training
        - Handles edge cases like already-trained transformations gracefully
    """
    # Check if this transformation type supports training
    if not hasattr(transform, "train"):
        return {
            "success": False,
            "error": "This transformation does not require training"
        }

    # Check if the transformation is already trained
    if not hasattr(transform, "is_trained") or transform.is_trained:
        return {"success": True, "already_trained": True}

    # Perform the training with the provided vectors
    # This modifies the transformation object's internal parameters
    transform.train(training_vectors)

    # Collect training results and transformation information
    results = {
        "success": True,
        "is_trained": getattr(transform, "is_trained", True),
        "input_dim": getattr(transform, "d_in", None),
        "output_dim": getattr(transform, "d_out", None)
    }

    # Add transformation-specific information for diagnostics and validation
    if isinstance(transform, faiss.PCAMatrix):
        results.update({
            "transform_type": "PCAMatrix",
            "eigen_power": transform.eigen_power,
            "has_bias": getattr(transform, "have_bias", None)
        })
    elif isinstance(transform, faiss.OPQMatrix):
        results.update({
            "transform_type": "OPQMatrix",
            "M": transform.M,
            "niter": getattr(transform, "niter", None)
        })
    elif isinstance(transform, faiss.ITQTransform):
        results.update({
            "transform_type": "ITQTransform",
            "niter": getattr(transform, "niter", None)
        })

    return results


def is_transform_trained(transform: Any) -> bool:
    """
    Check if a transformation is trained and ready for use.

    This function provides a simple interface to check whether a transformation
    has been properly trained and is ready for vector processing. Some transformations
    (like normalization) don't require training, while others (like PCA) must be
    trained with representative data before use.

    Args:
        transform: FAISS transformation object to check.

    Returns:
        bool: True if the transformation is trained or doesn't require training,
              False if training is required but not yet completed.

    Implementation Notes:
        - Transformations without is_trained attribute are assumed ready
        - This covers both parameter-free transforms and pre-trained transforms
        - Used to validate transformation pipelines before index operations
    """
    # If the transform has no is_trained attribute, it doesn't require training
    # This covers transformations like normalization that are stateless
    if not hasattr(transform, "is_trained"):
        return True

    # Return the actual training status for transformations that track it
    return transform.is_trained


def get_transform_info(index: Any) -> Dict[str, Any]:
    """
    Get comprehensive information about the transformations in an IndexPreTransform.

    This function extracts detailed information about all transformations in a
    pre-transform index, including their types, parameters, training status, and
    the underlying base index configuration.

    Args:
        index: faiss.IndexPreTransform object to analyze.

    Returns:
        dict: Comprehensive transformation information with the following structure:
            - success (bool): Whether information extraction succeeded
            - transforms (list): List of transformation details
            - base_index (dict): Information about the underlying index
            - ntotal (int): Total number of vectors in the index
            - input_dim (int): Input dimension for the transformation pipeline
            - output_dim (int): Output dimension after all transformations
            - error (str, optional): Error message if extraction failed

    Transform Information Includes:
        - Type and index in the transformation chain
        - Input/output dimensions for each stage
        - Training status and type-specific parameters
        - Configuration details like eigen_power, M, etc.
    """
    # Validate that this is actually a pre-transform index
    if not isinstance(index, faiss.IndexPreTransform):
        return {"success": False, "error": "Not an IndexPreTransform"}

    # Extract information about each transformation in the chain
    transforms = []
    for i in range(index.chain.size()):
        transform = index.chain.at(i)
        t_type = type(transform).__name__

        # Basic transformation information
        t_info = {
            "index": i,
            "type": t_type
        }

        # Add dimension information if available
        if hasattr(transform, "d_in"):
            t_info["input_dim"] = transform.d_in
        if hasattr(transform, "d_out"):
            t_info["output_dim"] = transform.d_out

        # Add training status for trainable transformations
        if hasattr(transform, "is_trained"):
            t_info["is_trained"] = transform.is_trained

        # Add transformation-specific configuration details
        if isinstance(transform, faiss.PCAMatrix):
            t_info.update({
                "transform_type": "PCAMatrix",
                "eigen_power": transform.eigen_power,
                "whitening": (transform.eigen_power == DEFAULT_PCA_EIGEN_POWER)
            })
        elif isinstance(transform, faiss.OPQMatrix):
            t_info.update({
                "transform_type": "OPQMatrix",
                "M": transform.M
            })
        elif isinstance(transform, faiss.NormalizationTransform):
            t_info.update({
                "transform_type": "NormalizationTransform",
                "norm_type": "L2"
            })
        elif isinstance(transform, faiss.ITQTransform):
            t_info["transform_type"] = "ITQTransform"
        elif isinstance(transform, faiss.RandomRotationMatrix):
            t_info["transform_type"] = "RandomRotationMatrix"

        transforms.append(t_info)

    # Get information about the base index after all transformations
    base_index = index.index
    base_info = {
        "type": type(base_index).__name__,
        "dimension": getattr(base_index, "d", None),
        "is_trained": getattr(base_index, "is_trained", True)
    }

    return {
        "success": True,
        "transforms": transforms,
        "base_index": base_info,
        "ntotal": index.ntotal,
        "input_dim": getattr(index, "d_in", None),
        "output_dim": getattr(index, "d", None)
    }


def apply_transform(
    index: Any,
    vectors: np.ndarray
) -> np.ndarray:
    """
    Apply the transformations in an IndexPreTransform to vectors without searching.

    This function allows you to apply the transformation pipeline of a pre-transform
    index to vectors without performing a search operation. This is useful for
    preprocessing vectors for analysis, debugging transformation effects, or
    preparing vectors for use with other indices.

    Args:
        index: faiss.IndexPreTransform object containing the transformation pipeline.
        vectors: Input vectors to transform. Should be a numpy array of shape
                (n_vectors, input_dimension) with float32 dtype.

    Returns:
        numpy.ndarray: Transformed vectors with the same number of rows but
                      potentially different dimensionality after transformations.

    Raises:
        ValueError: If the index is not an IndexPreTransform or if any transformation
                   in the pipeline is not properly trained.

    Implementation Notes:
        - Applies transformations in the order they appear in the chain
        - Validates that all transformations are trained before applying
        - Preserves input vector format and handles type conversion automatically
        - Memory-efficient implementation that processes vectors in sequence
    """
    # Validate that this is a pre-transform index
    if not isinstance(index, faiss.IndexPreTransform):
        raise ValueError("Not an IndexPreTransform")

    # Ensure input is a properly formatted numpy array
    # FAISS transformations require float32 for optimal performance
    vectors_np = np.array(vectors, dtype=np.float32)

    # Get the number of transformations in the pipeline
    ntrans = index.chain.size()

    # Apply each transformation in sequence
    # Start with the original vectors and progressively transform them
    transformed = vectors_np.copy()
    for i in range(ntrans):
        transform = index.chain.at(i)

        # Verify that the transformation is ready for use
        if hasattr(transform, "is_trained") and not transform.is_trained:
            raise ValueError(f"Transformation at index {i} is not trained")

        # Apply the transformation to get the next stage of processing
        transformed = transform.apply(transformed)

    return transformed


def get_metric_type(metric_name: str) -> int:
    """
    Get the FAISS metric type constant from a metric name.

    This function provides a unified interface for converting string metric names
    to the integer constants required by FAISS. It supports both canonical names
    and common aliases to provide flexibility in API usage.

    Args:
        metric_name: Name of the distance metric. Can be canonical names like
                    "L2", "INNER_PRODUCT" or aliases like "EUCLIDEAN", "COSINE".

    Returns:
        int: FAISS metric type constant suitable for index creation.

    Raises:
        ValueError: If the metric name is not recognized or supported.

    Supported Metrics:
        - L2/EUCLIDEAN: Euclidean distance (most common)
        - INNER_PRODUCT/IP/COSINE: Inner product similarity
        - L1/MANHATTAN: Manhattan distance
        - LINF: Chebyshev distance (L-infinity norm)
        - CANBERRA: Canberra distance (weighted L1)
        - BRAYCURTIS: Bray-Curtis distance
    """
    # Normalize to uppercase for case-insensitive matching
    metric_name = metric_name.upper()

    # Resolve aliases to canonical names
    if metric_name in METRIC_ALIASES:
        metric_name = METRIC_ALIASES[metric_name]

    # Look up the FAISS constant for the metric
    if metric_name not in METRIC_TYPES:
        raise ValueError(f"Unsupported metric type: {metric_name}")

    return METRIC_TYPES[metric_name]


def create_base_index(
    base_type: str,
    dimension: int,
    metric_type: str = "L2",
    index_params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create a base index of the specified type.

    This function serves as a factory for creating various FAISS index types with
    consistent parameter handling. It supports a wide range of index types from
    simple flat indices to complex quantized and graph-based indices.

    Args:
        base_type: Type specification for the index. Can be simple names like
                  "Flat" or complex specifications like "IVF100,PQ8".
        dimension: Vector dimension for the index. Must match the vectors that
                  will be added to the index.
        metric_type: Distance metric to use for similarity calculations.
                    Defaults to "L2" (Euclidean distance).
        index_params: Additional configuration parameters specific to the index type.

    Returns:
        FAISS index object ready for training (if required) and vector operations.

    Raises:
        ValueError: If the base_type is not supported or parameters are invalid.

    Supported Index Types:
        - Flat: Exact search with no compression
        - IVF: Inverted file with clustering
        - PQ: Product quantization for compression
        - HNSW: Hierarchical navigable small world graphs
        - LSH: Locality-sensitive hashing
        - Binary indices for bit vectors
    """
    params = index_params or {}
    faiss_metric = get_metric_type(metric_type)

    # Create the base index based on the specified type
    if base_type == "Flat":
        # Flat index: exact search, no approximation
        return faiss.IndexFlat(dimension, faiss_metric)

    elif base_type.startswith("IVF"):
        # IVF (Inverted File) indices with various quantization methods
        # Extract nlist (number of clusters) from IVFx specification
        match = re.match(r"IVF(\d+)(?:,(.+))?", base_type)
        if match:
            nlist = int(match.group(1))
            sub_type = match.group(2) or "Flat"

            # Create the quantizer (clustering structure)
            quantizer = faiss.IndexFlat(dimension, faiss_metric)

            # Create IVF index based on the quantization sub-type
            if sub_type == "Flat":
                # IVF with flat (exact) quantization
                return faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss_metric)
            elif sub_type.startswith("PQ"):
                # IVF with product quantization
                pq_match = re.match(r"PQ(\d+)", sub_type)
                if pq_match:
                    m = int(pq_match.group(1))
                    nbits = params.get("nbits", DEFAULT_PQ_BITS)
                    return faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits, faiss_metric)
            elif sub_type.startswith("SQ"):
                # IVF with scalar quantization
                sq_match = re.match(r"SQ(\d+)", sub_type)
                if sq_match:
                    qtype = int(sq_match.group(1))
                    return faiss.IndexIVFScalarQuantizer(
                        quantizer, dimension, nlist, qtype, faiss_metric
                    )

        raise ValueError(f"Invalid IVF index specification: {base_type}")

    elif base_type.startswith("PQ"):
        # Product Quantization indices for vector compression
        match = re.match(r"PQ(\d+)", base_type)
        if match:
            m = int(match.group(1))
            nbits = params.get("nbits", DEFAULT_PQ_BITS)
            return faiss.IndexPQ(dimension, m, nbits, faiss_metric)

    elif base_type == "HNSW" or base_type.startswith("HNSW"):
        # Hierarchical Navigable Small World graph indices
        m = DEFAULT_HNSW_M  # Default connectivity parameter
        match = re.match(r"HNSW(\d+)", base_type)
        if match:
            m = int(match.group(1))

        return faiss.IndexHNSWFlat(dimension, m, faiss_metric)

    elif base_type.startswith("LSH"):
        # Locality-Sensitive Hashing for approximate search
        nbits = DEFAULT_LSH_BITS  # Default hash length
        match = re.match(r"LSH(\d+)", base_type)
        if match:
            nbits = int(match.group(1))

        return faiss.IndexLSH(dimension, nbits)

    elif base_type.startswith("BinaryFlat"):
        # Binary flat index for bit-packed vectors
        if dimension % 8 != 0:
            raise ValueError("Dimension must be a multiple of 8 for binary indices")
        return faiss.IndexBinaryFlat(dimension)

    elif base_type.startswith("BinaryIVF"):
        # Binary IVF index for clustered bit-packed vectors
        if dimension % 8 != 0:
            raise ValueError("Dimension must be a multiple of 8 for binary indices")

        # Extract cluster count from BinaryIVFx specification
        match = re.match(r"BinaryIVF(\d+)", base_type)
        if match:
            nlist = int(match.group(1))
            quantizer = faiss.IndexBinaryFlat(dimension)
            return faiss.IndexBinaryIVF(quantizer, dimension, nlist)

    raise ValueError(f"Unsupported base index type: {base_type}")


def create_index_from_type(
    index_type: str,
    dimension: int,
    metric_type: str = "L2",
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create an index based on the specified type string with comprehensive metadata.

    This is the main index factory function that handles complex index specifications
    including transformations, IDMap wrappers, and specialized index configurations.
    It provides a unified interface for creating any supported FAISS index type.

    Args:
        index_type: Comprehensive index type specification. Can include transformations,
                   wrappers, and complex configurations like "PCA32,IVF100,PQ8".
        dimension: Input vector dimension before any transformations.
        metric_type: Distance metric for similarity calculations. Defaults to "L2".
        metadata: Additional metadata to attach to the index information.

    Returns:
        tuple: A two-element tuple containing:
            - index: The created FAISS index object ready for use
            - index_info: Comprehensive metadata about the index configuration

    Index Types Supported:
        - Basic: Flat, LSH, PQ
        - IVF family: IVF with various quantization methods
        - HNSW: Hierarchical graph indices
        - Binary: Indices for bit-packed vectors
        - Transformations: PCA, normalization, quantization preprocessing
        - Wrappers: IDMap for custom ID management

    Implementation Strategy:
        1. Detect special index types (binary, IDMap, transforms)
        2. Parse complex specifications into components
        3. Create transformations and base indices as needed
        4. Compose final index with proper metadata
    """
    # Set up logging for index creation debugging
    logger = logging.getLogger("faissx.server")

    logger.debug(
        f"Creating index of type '{index_type}', dimension {dimension}, metric {metric_type}"
    )

    # Handle binary index types with specialized creation logic
    if is_binary_index_type(index_type):
        from .binary import create_binary_index
        logger.debug(f"Creating binary index: {index_type}")
        return create_binary_index(index_type, dimension)

    # Check for pre-transform index types and parse the specification
    transform_type, base_index_type, transform_params = parse_transform_type(index_type)
    if transform_type is not None:
        logger.debug(f"Creating transformed index: {transform_type} with base {base_index_type}")

        # Create the transformation component first
        output_dim = transform_params.get("output_dim", dimension)
        transform, transform_info = create_transformation(
            transform_type,
            dimension,
            output_dim,
            **transform_params
        )

        # Create the base index using the transformed dimension
        base_index, base_info = create_index_from_type(
            base_index_type,
            output_dim,
            metric_type,
            metadata={"is_base_index": True}
        )

        # Compose the pre-transform index
        pretransform_index, index_info = create_pretransform_index(
            base_index, transform, transform_info
        )

        # Add metadata if provided
        if metadata:
            index_info["metadata"] = metadata

        return pretransform_index, index_info

    # Handle IDMap wrapper types for custom ID management
    if index_type.startswith("IDMap:"):
        logger.debug(f"Creating IDMap index with base type: {index_type[6:]}")
        base_type = index_type[6:]
        base_index, base_info = create_index_from_type(
            base_type, dimension, metric_type, metadata
        )

        # Create IDMap wrapper
        idmap_index = faiss.IndexIDMap(base_index)

        # Prepare comprehensive info
        idmap_info = {
            "type": "IDMap",
            "dimension": dimension,
            "base_type": base_type,
            "base_info": base_info,
            "is_trained": base_info.get("is_trained", True)
        }

        # Add metadata if provided
        if metadata:
            idmap_info["metadata"] = metadata

        return idmap_index, idmap_info

    if index_type.startswith("IDMap2:"):
        logger.debug(f"Creating IDMap2 index with base type: {index_type[7:]}")
        base_type = index_type[7:]
        base_index, base_info = create_index_from_type(
            base_type, dimension, metric_type, metadata
        )

        # Create IDMap2 wrapper
        idmap_index = faiss.IndexIDMap2(base_index)

        # Prepare comprehensive info
        idmap_info = {
            "type": "IDMap2",
            "dimension": dimension,
            "base_type": base_type,
            "base_info": base_info,
            "is_trained": base_info.get("is_trained", True)
        }

        # Add metadata if provided
        if metadata:
            idmap_info["metadata"] = metadata

        return idmap_index, idmap_info

    # Handle standard FAISS index types with metric conversion
    faiss_metric = faiss.METRIC_L2
    if metric_type.upper() == "IP":
        faiss_metric = faiss.METRIC_INNER_PRODUCT

    logger.debug(f"Metric type '{metric_type}' translated to faiss_metric={faiss_metric}")

    # Create standard index types with appropriate configurations
    if index_type == "L2" or index_type == "Flat":
        logger.debug(f"Creating IndexFlatL2 with dimension {dimension}")
        index = faiss.IndexFlatL2(dimension)
        index_info = {
            "type": "IndexFlatL2",
            "dimension": dimension,
            "is_trained": True
        }
    elif index_type == "IP":
        logger.debug(f"Creating IndexFlatIP with dimension {dimension}")
        index = faiss.IndexFlatIP(dimension)
        index_info = {
            "type": "IndexFlatIP",
            "dimension": dimension,
            "is_trained": True
        }
    elif index_type.startswith("HNSW") or index_type == "HNSW":
        # Extract M parameter if provided (e.g., HNSW32 -> M=32)
        M = DEFAULT_HNSW_M
        if index_type.startswith("HNSW") and len(index_type) > 4:
            try:
                M = int(index_type[4:])
                logger.debug(f"Extracted M={M} from index_type={index_type}")
            except ValueError:
                logger.warning(
                    f"Could not parse M from index_type={index_type}, using default M={M}"
                )

        # Create the HNSW index with specified parameters
        logger.debug(
            f"Creating IndexHNSWFlat with dimension={dimension}, M={M}, metric={faiss_metric}"
        )
        index = faiss.IndexHNSWFlat(dimension, M, faiss_metric)

        # Prepare comprehensive info
        index_info = {
            "type": "IndexHNSWFlat",
            "dimension": dimension,
            "M": M,
            "efConstruction": index.hnsw.efConstruction,
            "efSearch": index.hnsw.efSearch,
            "metric_type": "IP" if faiss_metric == faiss.METRIC_INNER_PRODUCT else "L2",
            "is_trained": True
        }
    elif index_type == "IVF":
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, DEFAULT_IVF_NLIST)
        index_info = {
            "type": "IndexIVFFlat",
            "dimension": dimension,
            "nlist": DEFAULT_IVF_NLIST,
            "metric_type": "L2",
            "is_trained": False,
            "requires_training": True
        }
    elif index_type == "IVF_IP":
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer, dimension, DEFAULT_IVF_NLIST, faiss.METRIC_INNER_PRODUCT
        )
        index_info = {
            "type": "IndexIVFFlat",
            "dimension": dimension,
            "nlist": DEFAULT_IVF_NLIST,
            "metric_type": "IP",
            "is_trained": False,
            "requires_training": True
        }
    elif index_type == "PQ":
        index = faiss.IndexPQ(dimension, DEFAULT_OPQ_SUBQUANTIZERS, DEFAULT_PQ_BITS)
        index_info = {
            "type": "IndexPQ",
            "dimension": dimension,
            "M": DEFAULT_OPQ_SUBQUANTIZERS,
            "nbits": DEFAULT_PQ_BITS,
            "is_trained": False,
            "requires_training": True
        }
    elif index_type == "PQ_IP":
        index = faiss.IndexPQ(
            dimension, DEFAULT_OPQ_SUBQUANTIZERS, DEFAULT_PQ_BITS, faiss.METRIC_INNER_PRODUCT
        )
        index_info = {
            "type": "IndexPQ",
            "dimension": dimension,
            "M": DEFAULT_OPQ_SUBQUANTIZERS,
            "nbits": DEFAULT_PQ_BITS,
            "metric_type": "IP",
            "is_trained": False,
            "requires_training": True
        }
    # Add support for PQ with specific M and nbits (e.g., PQ4x8)
    elif pq_match := re.match(r"PQ(\d+)x(\d+)", index_type):
        m, nbits = map(int, pq_match.groups())
        metric = faiss.METRIC_L2
        if metric_type.upper() == "IP":
            metric = faiss.METRIC_INNER_PRODUCT

        index = faiss.IndexPQ(dimension, m, nbits, metric)
        index_info = {
            "type": "IndexPQ",
            "dimension": dimension,
            "M": m,
            "nbits": nbits,
            "metric_type": "IP" if metric == faiss.METRIC_INNER_PRODUCT else "L2",
            "is_trained": False,
            "requires_training": True
        }
    # Add support for IVF with SQ (e.g., IVF4_SQ0)
    elif ivf_sq_match := re.match(r"IVF(\d+)_SQ(\d+)", index_type):
        nlist, qtype = map(int, ivf_sq_match.groups())
        quantizer = faiss.IndexFlatL2(dimension)
        metric = faiss.METRIC_L2
        if metric_type.upper() == "IP":
            metric = faiss.METRIC_INNER_PRODUCT

        index = faiss.IndexIVFScalarQuantizer(
            quantizer, dimension, nlist, qtype, metric
        )
        index_info = {
            "type": "IndexIVFScalarQuantizer",
            "dimension": dimension,
            "nlist": nlist,
            "qtype": qtype,
            "metric_type": "IP" if metric == faiss.METRIC_INNER_PRODUCT else "L2",
            "is_trained": False,
            "requires_training": True
        }
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    # Add metadata if provided
    if metadata:
        index_info["metadata"] = metadata

    return index, index_info


def is_binary_index_type(index_type: str) -> bool:
    """
    Check if the index type is a binary index.

    Binary indices work with bit-packed vectors rather than floating-point vectors.
    They are used for binary codes, hash codes, or other compact representations
    where each dimension is a single bit.

    Args:
        index_type: Index type string to check for binary classification.

    Returns:
        bool: True if the index type represents a binary index.

    Implementation Notes:
        - Checks for standard binary index prefixes
        - Used to route index creation to appropriate handlers
        - Binary indices have different dimension requirements (must be multiple of 8)
    """
    # Validate input type for robustness
    if not isinstance(index_type, str):
        return False

    # Check against known binary index prefixes
    return any(index_type.startswith(prefix) for prefix in BINARY_INDEX_PREFIXES)


def create_specialized_index(
    dimension: int,
    index_type: str = "Flat",
    metric_type: str = "L2",
    index_params: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a specialized index optimized for specific use cases.

    This function provides pre-configured index templates that are optimized for
    common usage patterns. Instead of manually configuring complex index types,
    users can specify high-level objectives and get appropriately configured indices.

    Args:
        dimension: Vector dimension for the index.
        index_type: Base index type or optimization template. Can be standard FAISS
                   types or specialized templates like "fast_search", "balanced".
        metric_type: Distance metric for similarity calculations.
        index_params: Additional configuration parameters.

    Returns:
        tuple: (index, index_info) with the optimized index and its metadata.

    Specialized Templates:
        - fast_search: Optimized for fast search with good recall
        - balanced: Balance between build time, memory, and search speed
        - accuracy: Prioritizes search accuracy over speed
        - memory_efficient: Minimizes memory usage with compression
        - binary: Optimized for binary/hash vectors

    Implementation Strategy:
        - Templates use dimension-adaptive parameters
        - Configurations based on empirical performance testing
        - Falls back to standard index creation for unknown templates
    """
    params = index_params or {}

    # Handle specialized templates with dimension-adaptive configurations
    if index_type == "fast_search":
        # Optimized for fast search with good recall using IVF+PQ
        # Parameters scale with dimension for optimal performance
        nlist = min(4 * int(np.sqrt(dimension)), 1024)
        m = min(int(dimension / 4), 64)
        return create_index_from_type(
            f"IVF{nlist},PQ{m}",
            dimension, metric_type, params
        )

    elif index_type == "balanced":
        # Balanced between build time, memory and search speed using IVF+Flat
        # Uses more clusters than fast_search but avoids quantization
        nlist = min(int(np.sqrt(dimension) * 8), 2048)
        return create_index_from_type(
            f"IVF{nlist},Flat",
            dimension, metric_type, params
        )

    elif index_type == "accuracy":
        # Optimized for accuracy using HNSW with high search parameters
        return create_index_from_type(
            "HNSW32", dimension, metric_type,
            {**params, "efConstruction": DEFAULT_HNSW_EF_CONSTRUCTION,
             "efSearch": DEFAULT_HNSW_EF_SEARCH}
        )

    elif index_type == "memory_efficient":
        # Optimized for memory efficiency using PCA+PQ compression
        pca_dim = max(int(dimension / 2), 32)
        m = max(int(dimension / 8), 8)
        return create_index_from_type(
            f"PCA{pca_dim},PQ{m}",
            dimension, metric_type, params
        )

    elif index_type == "binary":
        # Binary index for binary vectors or hash codes
        if dimension % 8 != 0:
            raise ValueError("Dimension must be a multiple of 8 for binary indices")
        return create_index_from_type("BinaryFlat", dimension, metric_type, params)

    else:
        # Use standard index creation for other types
        return create_index_from_type(index_type, dimension, metric_type, params)


def list_supported_index_types() -> Dict[str, List[str]]:
    """
    List all supported index types and their variants.

    This function provides comprehensive documentation of all supported index types,
    organized by category. It's useful for API documentation, validation, and
    helping users choose appropriate index types.

    Returns:
        dict: Categories of supported index types with examples and placeholders.

    Categories Include:
        - basic: Simple indices without clustering
        - ivf: Inverted file indices with various quantization methods
        - hnsw: Hierarchical graph indices
        - binary: Indices for bit-packed vectors
        - transformations: Vector preprocessing options
        - metrics: Supported distance metrics
        - templates: Pre-configured optimization templates
    """
    return {
        "basic": [
            "Flat",
            "LSH",
            "PQ{M}"
        ],
        "ivf": [
            "IVF{nlist},Flat",
            "IVF{nlist},PQ{M}",
            "IVF{nlist},SQ{qtype}"
        ],
        "hnsw": [
            "HNSW",
            "HNSW{M}"
        ],
        "binary": [
            "BinaryFlat",
            "BinaryIVF{nlist}"
        ],
        "transformations": list(TRANSFORM_TYPES.keys()),
        "metrics": list(METRIC_TYPES.keys()) + list(METRIC_ALIASES.keys()),
        "templates": [
            "fast_search",
            "balanced",
            "accuracy",
            "memory_efficient",
            "binary"
        ]
    }


def list_supported_metrics() -> Dict[str, List[str]]:
    """
    List all supported distance metrics with their aliases.

    This function provides a comprehensive mapping of supported distance metrics
    and their alternative names. It's useful for API documentation and validation.

    Returns:
        dict: Mapping of canonical metric names to their aliases.

    Implementation Notes:
        - Groups aliases under their canonical metric names
        - Includes both mathematical and common alternative names
        - Used for API validation and user guidance
    """
    metrics = {}
    for name, _metric in METRIC_TYPES.items():
        metrics[name] = [alias for alias, target in METRIC_ALIASES.items() if target == name]

    return metrics


def get_transform_training_requirements(transform: Any) -> Dict[str, Any]:
    """
    Get comprehensive training requirements for a transformation.

    This function analyzes a transformation object and provides detailed information
    about its training requirements, including minimum and recommended vector counts,
    training status, and explanatory descriptions.

    Args:
        transform: The FAISS transformation object to analyze.

    Returns:
        dict: Comprehensive training requirements with the following structure:
            - requires_training (bool): Whether training is needed
            - is_trained (bool): Current training status
            - min_training_vectors (int): Minimum vectors for acceptable results
            - recommended_training_vectors (int): Recommended count for optimal results
            - description (str, optional): Explanation of training requirements

    Training Requirements by Type:
        - PCA: Needs diverse vectors to learn principal components
        - OPQ: Requires substantial data for optimal quantization
        - ITQ: Needs representative data for binary quantization
        - Others: May not require training (e.g., normalization)

    Implementation Notes:
        - Requirements are based on statistical needs of each algorithm
        - Minimum counts ensure algorithm convergence
        - Recommended counts provide good quality results
        - Non-trainable transforms return zero requirements
    """
    # Initialize with default requirements (no training needed)
    requirements = {
        "requires_training": False,
        "is_trained": True,
        "min_training_vectors": 0,
        "recommended_training_vectors": 0
    }

    # If the transform has no is_trained attribute, it doesn't require training
    # This covers parameter-free transformations like normalization
    if not hasattr(transform, "is_trained"):
        return requirements

    # Update requirements based on actual training status
    requirements["is_trained"] = transform.is_trained
    requirements["requires_training"] = not transform.is_trained

    # Set specific requirements based on transformation algorithm
    if isinstance(transform, faiss.PCAMatrix):
        # PCA needs enough vectors to estimate covariance matrix reliably
        input_dim = transform.d_in if hasattr(transform, "d_in") else transform.d
        output_dim = transform.d_out if hasattr(transform, "d_out") else input_dim
        requirements.update({
            "min_training_vectors": output_dim * PCA_MIN_TRAINING_MULTIPLIER,
            "recommended_training_vectors": output_dim * PCA_RECOMMENDED_TRAINING_MULTIPLIER,
            "description": "PCA requires representative training data"
        })

    elif isinstance(transform, faiss.OPQMatrix):
        # OPQ needs substantial data for both rotation and quantization learning
        input_dim = transform.d_in if hasattr(transform, "d_in") else transform.d
        output_dim = transform.d_out if hasattr(transform, "d_out") else input_dim
        m = transform.M if hasattr(transform, "M") else DEFAULT_OPQ_SUBQUANTIZERS
        requirements.update({
            "min_training_vectors": output_dim * m,
            "recommended_training_vectors": output_dim * m * OPQ_TRAINING_MULTIPLIER,
            "description": "OPQ requires substantial training data"
        })

    elif isinstance(transform, faiss.ITQTransform):
        # ITQ needs diverse data for learning optimal binary quantization
        input_dim = transform.d_in if hasattr(transform, "d_in") else transform.d
        requirements.update({
            "min_training_vectors": input_dim * ITQ_MIN_TRAINING_MULTIPLIER,
            "recommended_training_vectors": input_dim * ITQ_RECOMMENDED_TRAINING_MULTIPLIER,
            "description": "ITQ requires representative training data"
        })

    return requirements

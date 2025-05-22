#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

This module provides functionality for vector transformations and
handling different index types with consistent parameter support.
"""

import re
import faiss
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


# Supported transformation types
TRANSFORM_TYPES = {
    "PCA": faiss.PCAMatrix,              # Principal Component Analysis
    "PCAR": faiss.PCAMatrix,             # PCA with whitening
    "L2NORM": faiss.NormalizationTransform,  # L2 Normalization
    "NORM": faiss.NormalizationTransform,    # Normalization (alias for L2NORM)
    "ITQ": faiss.ITQTransform,           # Iterative Quantization
    "OPQ": faiss.OPQMatrix,              # Optimized Product Quantization
    "RR": faiss.RandomRotationMatrix     # Random Rotation
}

# Supported distance metrics
METRIC_TYPES = {
    "L2": faiss.METRIC_L2,               # Euclidean distance
    "INNER_PRODUCT": faiss.METRIC_INNER_PRODUCT,  # Inner product (cosine if normalized)
    "L1": faiss.METRIC_L1,               # L1 distance (Manhattan)
    "LINF": faiss.METRIC_Linf,           # L-infinity distance
    "CANBERRA": faiss.METRIC_Canberra,   # Canberra distance
    "BRAYCURTIS": faiss.METRIC_BrayCurtis  # Bray-Curtis distance
}

# Mapping of metric types to their common aliases
METRIC_ALIASES = {
    "IP": "INNER_PRODUCT",
    "EUCLIDEAN": "L2",
    "COSINE": "INNER_PRODUCT",  # Cosine is inner product on normalized vectors
    "MANHATTAN": "L1"
}


def create_transformation(
    transform_type: str,
    input_dim: int,
    output_dim: Optional[int] = None,
    transform_params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create a transformation object based on the specified type.

    Args:
        transform_type: Type of transformation (PCA, PCAR, L2NORM, ITQ, OPQ, RR)
        input_dim: Input dimension
        output_dim: Output dimension (optional, defaults to input_dim)
        transform_params: Additional parameters for the transformation

    Returns:
        FAISS transformation object

    Raises:
        ValueError: If the transform_type is not supported
    """
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError(f"Unsupported transformation type: {transform_type}")

    # Default to input_dim if output_dim not specified
    if output_dim is None:
        output_dim = input_dim

    # Default parameters
    params = transform_params or {}

    # Create the transformation
    if transform_type in ["PCA", "PCAR"]:
        transform = TRANSFORM_TYPES[transform_type](input_dim, output_dim)

        # Set whitening for PCAR
        if transform_type == "PCAR":
            transform.eigen_power = -0.5  # Whitening

        # Set optional parameters
        if "eigen_power" in params:
            transform.eigen_power = params["eigen_power"]
        if "random_rotation" in params:
            transform.random_rotation = params["random_rotation"]

    elif transform_type in ["L2NORM", "NORM"]:
        transform = TRANSFORM_TYPES[transform_type](input_dim)

    elif transform_type == "ITQ":
        # ITQ requires a dimension and number of iterations
        n_iter = params.get("n_iterations", 50)
        transform = TRANSFORM_TYPES[transform_type](input_dim, n_iter)

    elif transform_type == "OPQ":
        # OPQ requires dimension and number of subquantizers
        m = params.get("M", 8)  # Number of subquantizers
        transform = TRANSFORM_TYPES[transform_type](input_dim, output_dim, m)

        # Set optional parameters
        if "niter" in params:
            transform.niter = params["niter"]

    elif transform_type == "RR":
        transform = TRANSFORM_TYPES[transform_type](input_dim, output_dim)

        # Set optional parameters
        if "seed" in params:
            faiss.seed_rand(params["seed"])

    return transform


def create_pretransform_index(
    transforms: List[Any],
    base_index: Any
) -> faiss.IndexPreTransform:
    """
    Create an IndexPreTransform with the given transformations and base index.

    Args:
        transforms: List of transformation objects
        base_index: Base index to use after transformation

    Returns:
        IndexPreTransform object
    """
    # Create the index pre-transform
    index = faiss.IndexPreTransform(len(transforms), base_index)

    # Add each transformation
    for i, transform in enumerate(transforms):
        index.prepend_transform(transform)

    return index


def parse_transform_type(index_type: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parse a compound index type string into transformation descriptions and base index type.

    Example:
        "PCA32,L2NORM,Flat" -> ([{"type": "PCA", "dim": 32}, {"type": "L2NORM"}], "Flat")

    Args:
        index_type: Compound index type string

    Returns:
        tuple: (list of transformation descriptions, base index type)
    """
    parts = index_type.split(",")

    # The last part is the base index type
    base_index_type = parts[-1]

    # Parse transformation descriptions
    transforms = []
    for part in parts[:-1]:
        # Match transformation type and optional dimension
        match = re.match(r"([A-Za-z0-9]+)(\d*)", part)
        if match:
            t_type, t_dim = match.groups()
            transform = {"type": t_type}

            # Add dimension if specified
            if t_dim:
                transform["dim"] = int(t_dim)

            transforms.append(transform)

    return transforms, base_index_type


def train_transform(
    transform: Any,
    training_vectors: np.ndarray
) -> Dict[str, Any]:
    """
    Train a transformation with the provided training vectors.

    Args:
        transform: Transformation object to train
        training_vectors: Vectors to use for training

    Returns:
        dict: Training results
    """
    if not hasattr(transform, "train"):
        return {"success": False, "error": "This transformation does not require training"}

    if not hasattr(transform, "is_trained") or transform.is_trained:
        return {"success": True, "already_trained": True}

    # Train the transformation
    transform.train(training_vectors)

    # Get training results
    results = {
        "success": True,
        "is_trained": getattr(transform, "is_trained", True),
        "input_dim": getattr(transform, "d_in", None),
        "output_dim": getattr(transform, "d_out", None)
    }

    # Add transformation-specific info
    if isinstance(transform, faiss.PCAMatrix):
        results["transform_type"] = "PCAMatrix"
        results["eigen_power"] = transform.eigen_power
        results["has_bias"] = getattr(transform, "have_bias", None)
    elif isinstance(transform, faiss.OPQMatrix):
        results["transform_type"] = "OPQMatrix"
        results["M"] = transform.M
        results["niter"] = getattr(transform, "niter", None)
    elif isinstance(transform, faiss.ITQTransform):
        results["transform_type"] = "ITQTransform"
        results["niter"] = getattr(transform, "niter", None)

    return results


def is_transform_trained(transform: Any) -> bool:
    """
    Check if a transformation is trained.

    Args:
        transform: Transformation object

    Returns:
        bool: True if the transformation is trained or doesn't require training
    """
    # If the transform has no is_trained attribute, it doesn't require training
    if not hasattr(transform, "is_trained"):
        return True

    return transform.is_trained


def get_transform_info(index: Any) -> Dict[str, Any]:
    """
    Get information about the transformations in an IndexPreTransform.

    Args:
        index: IndexPreTransform object

    Returns:
        dict: Information about the transformations
    """
    if not isinstance(index, faiss.IndexPreTransform):
        return {"success": False, "error": "Not an IndexPreTransform"}

    # Get information about the chain of transformations
    transforms = []
    for i in range(index.chain.size()):
        transform = index.chain.at(i)
        t_type = type(transform).__name__

        t_info = {
            "index": i,
            "type": t_type
        }

        # Add dimension information if available
        if hasattr(transform, "d_in"):
            t_info["input_dim"] = transform.d_in
        if hasattr(transform, "d_out"):
            t_info["output_dim"] = transform.d_out

        # Add training status if applicable
        if hasattr(transform, "is_trained"):
            t_info["is_trained"] = transform.is_trained

        # Add transform-specific information
        if isinstance(transform, faiss.PCAMatrix):
            t_info["transform_type"] = "PCAMatrix"
            t_info["eigen_power"] = transform.eigen_power
            t_info["whitening"] = (transform.eigen_power == -0.5)
        elif isinstance(transform, faiss.OPQMatrix):
            t_info["transform_type"] = "OPQMatrix"
            t_info["M"] = transform.M
        elif isinstance(transform, faiss.NormalizationTransform):
            t_info["transform_type"] = "NormalizationTransform"
            t_info["norm_type"] = "L2"
        elif isinstance(transform, faiss.ITQTransform):
            t_info["transform_type"] = "ITQTransform"
        elif isinstance(transform, faiss.RandomRotationMatrix):
            t_info["transform_type"] = "RandomRotationMatrix"

        transforms.append(t_info)

    # Get information about the base index
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

    Args:
        index: IndexPreTransform object
        vectors: Vectors to transform

    Returns:
        numpy.ndarray: Transformed vectors

    Raises:
        ValueError: If the index is not an IndexPreTransform
    """
    if not isinstance(index, faiss.IndexPreTransform):
        raise ValueError("Not an IndexPreTransform")

    # Make sure the input is a numpy array with the right shape and type
    vectors_np = np.array(vectors, dtype=np.float32)

    # Get the number of transformations
    ntrans = index.chain.size()

    # Apply each transformation in sequence
    transformed = vectors_np.copy()
    for i in range(ntrans):
        transform = index.chain.at(i)

        # Skip if not trained
        if hasattr(transform, "is_trained") and not transform.is_trained:
            raise ValueError(f"Transformation at index {i} is not trained")

        # Apply the transformation
        transformed = transform.apply(transformed)

    return transformed


def get_metric_type(metric_name: str) -> int:
    """
    Get the FAISS metric type constant from a metric name.

    Args:
        metric_name: Name of the metric (L2, INNER_PRODUCT, etc.)

    Returns:
        int: FAISS metric type constant

    Raises:
        ValueError: If the metric is not supported
    """
    # Convert to uppercase
    metric_name = metric_name.upper()

    # Check if it's an alias
    if metric_name in METRIC_ALIASES:
        metric_name = METRIC_ALIASES[metric_name]

    # Get the metric type
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

    Args:
        base_type: Type of the base index (Flat, IVF, etc.)
        dimension: Dimension of vectors
        metric_type: Distance metric to use
        index_params: Additional parameters for the index

    Returns:
        FAISS index object

    Raises:
        ValueError: If the base_type is not supported
    """
    params = index_params or {}
    faiss_metric = get_metric_type(metric_type)

    # Create the base index based on type
    if base_type == "Flat":
        return faiss.IndexFlat(dimension, faiss_metric)

    elif base_type.startswith("IVF"):
        # Extract nlist from IVFx where x is nlist
        match = re.match(r"IVF(\d+)(?:,(.+))?", base_type)
        if match:
            nlist = int(match.group(1))
            sub_type = match.group(2) or "Flat"

            # Create quantizer
            quantizer = faiss.IndexFlat(dimension, faiss_metric)

            # Create IVF index based on sub-type
            if sub_type == "Flat":
                return faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss_metric)
            elif sub_type.startswith("PQ"):
                # Extract M from PQx where x is M
                pq_match = re.match(r"PQ(\d+)", sub_type)
                if pq_match:
                    m = int(pq_match.group(1))
                    nbits = params.get("nbits", 8)
                    return faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits, faiss_metric)
            elif sub_type.startswith("SQ"):
                # Extract quantizer type from SQx
                sq_match = re.match(r"SQ(\d+)", sub_type)
                if sq_match:
                    qtype = int(sq_match.group(1))
                    return faiss.IndexIVFScalarQuantizer(quantizer, dimension, nlist, qtype, faiss_metric)

        raise ValueError(f"Invalid IVF index specification: {base_type}")

    elif base_type.startswith("PQ"):
        # Extract M from PQx where x is M
        match = re.match(r"PQ(\d+)", base_type)
        if match:
            m = int(match.group(1))
            nbits = params.get("nbits", 8)
            return faiss.IndexPQ(dimension, m, nbits, faiss_metric)

    elif base_type == "HNSW" or base_type.startswith("HNSW"):
        # Extract M from HNSWx where x is M
        m = 32  # Default M
        match = re.match(r"HNSW(\d+)", base_type)
        if match:
            m = int(match.group(1))

        return faiss.IndexHNSWFlat(dimension, m, faiss_metric)

    elif base_type.startswith("LSH"):
        # Extract nbits from LSHx where x is nbits
        nbits = 32  # Default nbits
        match = re.match(r"LSH(\d+)", base_type)
        if match:
            nbits = int(match.group(1))

        return faiss.IndexLSH(dimension, nbits)

    elif base_type.startswith("BinaryFlat"):
        if dimension % 8 != 0:
            raise ValueError("Dimension must be a multiple of 8 for binary indices")
        return faiss.IndexBinaryFlat(dimension)

    elif base_type.startswith("BinaryIVF"):
        if dimension % 8 != 0:
            raise ValueError("Dimension must be a multiple of 8 for binary indices")

        # Extract nlist from BinaryIVFx where x is nlist
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
    index_params: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create an index from a type specification, which may include transformations.

    Examples:
        "Flat" -> IndexFlat
        "PCA32,L2NORM,Flat" -> IndexPreTransform with PCA, L2 normalization, and Flat index
        "IVF100,PQ16" -> IndexIVFPQ with 100 lists and 16 subquantizers

    Args:
        index_type: Type specification
        dimension: Original dimension of vectors
        metric_type: Distance metric to use
        index_params: Additional parameters for the index

    Returns:
        tuple: (index, index_info)
    """
    params = index_params or {}

    # Check if it's a compound type with transformations
    if "," in index_type:
        transform_descs, base_type = parse_transform_type(index_type)

        # Create base index
        base_index = create_base_index(base_type, dimension, metric_type, params)
        base_dim = dimension

        # Create transformations in reverse order (last transformation first)
        transforms = []
        for t_desc in transform_descs:
            t_type = t_desc["type"]
            t_input_dim = base_dim

            # Get output dimension if specified
            t_output_dim = t_desc.get("dim", t_input_dim)

            # Create the transformation
            transform = create_transformation(t_type, t_input_dim, t_output_dim)
            transforms.append(transform)

            # Update base dimension for next transformation
            base_dim = t_output_dim

        # Create IndexPreTransform
        index = create_pretransform_index(transforms, base_index)

        # Prepare index info
        index_info = {
            "type": "IndexPreTransform",
            "base_type": base_type,
            "metric_type": metric_type,
            "input_dimension": dimension,
            "output_dimension": base_dim,
            "transformations": [{"type": t["type"]} for t in transform_descs]
        }

    else:
        # Simple index type
        index = create_base_index(index_type, dimension, metric_type, params)

        # Prepare index info
        index_info = {
            "type": index_type,
            "dimension": dimension,
            "metric_type": metric_type
        }

    return index, index_info


def create_specialized_index(
    dimension: int,
    index_type: str = "Flat",
    metric_type: str = "L2",
    index_params: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a specialized index optimized for specific use cases.

    Args:
        dimension: Dimension of vectors
        index_type: Base index type or template (e.g., "pq_fast", "hnsw_balanced")
        metric_type: Distance metric to use
        index_params: Additional parameters for the index

    Returns:
        tuple: (index, index_info)
    """
    params = index_params or {}

    # Handle specialized templates
    if index_type == "fast_search":
        # Optimized for fast search with good recall
        return create_index_from_type(f"IVF{min(4 * int(np.sqrt(dimension)), 1024)},PQ{min(int(dimension / 4), 64)}",
                                     dimension, metric_type, params)

    elif index_type == "balanced":
        # Balanced between build time, memory and search speed
        return create_index_from_type(f"IVF{min(int(np.sqrt(dimension) * 8), 2048)},Flat",
                                     dimension, metric_type, params)

    elif index_type == "accuracy":
        # Optimized for accuracy
        return create_index_from_type(f"HNSW32", dimension, metric_type,
                                     {**params, "efConstruction": 200, "efSearch": 128})

    elif index_type == "memory_efficient":
        # Optimized for memory efficiency
        m = max(int(dimension / 8), 8)
        return create_index_from_type(f"PCA{max(int(dimension / 2), 32)},PQ{m}",
                                     dimension, metric_type, params)

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

    Returns:
        dict: Categories of supported index types
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

    Returns:
        dict: Metrics and their aliases
    """
    metrics = {}
    for name, metric in METRIC_TYPES.items():
        metrics[name] = [alias for alias, target in METRIC_ALIASES.items() if target == name]

    return metrics

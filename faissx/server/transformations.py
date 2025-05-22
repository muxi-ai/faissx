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
from typing import Any, Dict, List, Optional, Tuple
import logging


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


def parse_transform_type(index_type: str) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """
    Parse a compound index type string into transformation type, base index type, and parameters.

    Example:
        "PCA32,Flat" -> ("PCA", "Flat", {"dim": 32})
        "PQ4x8" -> (None, "PQ4x8", {})

    Args:
        index_type: Compound index type string

    Returns:
        tuple: (transformation type or None, base index type, transform parameters)
    """
    # Handle special case for PQ indices like "PQ4x8"
    pq_match = re.match(r"PQ(\d+)x(\d+)", index_type)
    if pq_match:
        m, nbits = pq_match.groups()
        return None, index_type, {}

    # Handle special case for IVF_SQ indices like "IVF4_SQ0"
    ivf_sq_match = re.match(r"IVF(\d+)_SQ(\d+)", index_type)
    if ivf_sq_match:
        nlist, qtype = ivf_sq_match.groups()
        return None, index_type, {}

    parts = index_type.split(",")

    # If there's only one part, it's just the base index type, no transformation
    if len(parts) == 1:
        return None, index_type, {}

    # The last part is the base index type
    base_index_type = parts[-1]

    # The first part is the transformation type
    transform_part = parts[0]

    # Match transformation type and optional dimension
    match = re.match(r"([A-Za-z]+)(\d*)", transform_part)
    if match:
        t_type, t_dim = match.groups()
        transform_params = {}

        # Add dimension if specified
        if t_dim:
            transform_params["dim"] = int(t_dim)

        return t_type, base_index_type, transform_params

    # If no match, return the first part as the transform type with no params
    return parts[0], base_index_type, {}


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


def create_index_from_type(index_type, dimension, metric_type="L2", metadata=None):
    """
    Create an index based on the specified type string.

    Args:
        index_type (str): Type of index to create
        dimension (int): Vector dimension
        metric_type (str): Distance metric type
        metadata (dict, optional): Additional metadata

    Returns:
        tuple: (index, index_info)
    """
    # Set up logging
    logger = logging.getLogger("faissx.server")

    logger.debug(
        f"Creating index of type '{index_type}', dimension {dimension}, metric {metric_type}"
    )

    # Handle binary index types
    if is_binary_index_type(index_type):
        from .binary import create_binary_index
        logger.debug(f"Creating binary index: {index_type}")
        return create_binary_index(index_type, dimension)

    # Check for pre-transform index types
    transform_type, base_index_type, transform_params = parse_transform_type(index_type)
    if transform_type is not None:
        logger.debug(f"Creating transformed index: {transform_type} with base {base_index_type}")
        # First create the transformation
        output_dim = transform_params.get("output_dim", dimension)
        transform, transform_info = create_transformation(
            transform_type,
            dimension,
            output_dim,
            **transform_params
        )

        # Create the base index using the output dimension from the transform
        base_index, base_info = create_index_from_type(
            base_index_type,
            output_dim,
            metric_type,
            metadata={"is_base_index": True}
        )

        # Create the pretransform index
        pretransform_index, index_info = create_pretransform_index(
            base_index, transform, transform_info
        )

        # Add metadata if provided
        if metadata:
            index_info["metadata"] = metadata

        return pretransform_index, index_info

    # Handle IDMap types
    if index_type.startswith("IDMap:"):
        logger.debug(f"Creating IDMap index with base type: {index_type[6:]}")
        base_type = index_type[6:]
        base_index, base_info = create_index_from_type(
            base_type, dimension, metric_type, metadata
        )

        # Create IDMap wrapper
        idmap_index = faiss.IndexIDMap(base_index)

        # Prepare info
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

        # Prepare info
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

    # Handle standard FAISS index types
    faiss_metric = faiss.METRIC_L2
    if metric_type.upper() == "IP":
        faiss_metric = faiss.METRIC_INNER_PRODUCT

    logger.debug(f"Metric type '{metric_type}' translated to faiss_metric={faiss_metric}")

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
        M = 32  # Default value
        if index_type.startswith("HNSW") and len(index_type) > 4:
            try:
                M = int(index_type[4:])
                logger.debug(f"Extracted M={M} from index_type={index_type}")
            except ValueError:
                logger.warning(f"Could not parse M from index_type={index_type}, using default M={M}")

        # Create the HNSW index
        logger.debug(f"Creating IndexHNSWFlat with dimension={dimension}, M={M}, metric={faiss_metric}")
        index = faiss.IndexHNSWFlat(dimension, M, faiss_metric)

        # Prepare info
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
        index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 centroids by default
        index_info = {
            "type": "IndexIVFFlat",
            "dimension": dimension,
            "nlist": 100,
            "metric_type": "L2",
            "is_trained": False,
            "requires_training": True
        }
    elif index_type == "IVF_IP":
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
        index_info = {
            "type": "IndexIVFFlat",
            "dimension": dimension,
            "nlist": 100,
            "metric_type": "IP",
            "is_trained": False,
            "requires_training": True
        }
    elif index_type == "PQ":
        index = faiss.IndexPQ(dimension, 8, 8)  # 8 subquantizers with 8 bits each by default
        index_info = {
            "type": "IndexPQ",
            "dimension": dimension,
            "M": 8,
            "nbits": 8,
            "is_trained": False,
            "requires_training": True
        }
    elif index_type == "PQ_IP":
        index = faiss.IndexPQ(dimension, 8, 8, faiss.METRIC_INNER_PRODUCT)
        index_info = {
            "type": "IndexPQ",
            "dimension": dimension,
            "M": 8,
            "nbits": 8,
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


def is_binary_index_type(index_type):
    """
    Check if the index type is a binary index.

    Args:
        index_type (str): Index type to check

    Returns:
        bool: True if it's a binary index type
    """
    # First check if the input is a string
    if not isinstance(index_type, str):
        return False

    binary_prefixes = ["BINARY_", "BIN_"]
    return any(index_type.startswith(prefix) for prefix in binary_prefixes)


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


def get_transform_training_requirements(transform: Any) -> Dict[str, Any]:
    """
    Get training requirements for a transformation.

    Args:
        transform: The transformation to check

    Returns:
        dict: Training requirements information
    """
    requirements = {
        "requires_training": False,
        "is_trained": True,
        "min_training_vectors": 0,
        "recommended_training_vectors": 0
    }

    # If the transform has no is_trained attribute, it doesn't require training
    if not hasattr(transform, "is_trained"):
        return requirements

    # Update requirements based on training status
    requirements["is_trained"] = transform.is_trained
    requirements["requires_training"] = not transform.is_trained

    # Set specific requirements based on transform type
    if isinstance(transform, faiss.PCAMatrix):
        input_dim = transform.d_in if hasattr(transform, "d_in") else transform.d
        output_dim = transform.d_out if hasattr(transform, "d_out") else input_dim
        requirements["min_training_vectors"] = output_dim * 2
        requirements["recommended_training_vectors"] = output_dim * 10
        requirements["description"] = "PCA requires representative training data"

    elif isinstance(transform, faiss.OPQMatrix):
        input_dim = transform.d_in if hasattr(transform, "d_in") else transform.d
        output_dim = transform.d_out if hasattr(transform, "d_out") else input_dim
        m = transform.M if hasattr(transform, "M") else 8
        requirements["min_training_vectors"] = output_dim * m
        requirements["recommended_training_vectors"] = output_dim * m * 10
        requirements["description"] = "OPQ requires substantial training data"

    elif isinstance(transform, faiss.ITQTransform):
        input_dim = transform.d_in if hasattr(transform, "d_in") else transform.d
        requirements["min_training_vectors"] = input_dim * 5
        requirements["recommended_training_vectors"] = input_dim * 20
        requirements["description"] = "ITQ requires representative training data"

    return requirements

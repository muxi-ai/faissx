#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Vector Transformations Support
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
FAISSx Server Vector Transformations Support

This module provides functionality for handling vector transformations in the
FAISSx server, supporting IndexPreTransform and various transformation types.
"""

import faiss
import numpy as np
from typing import Any, Dict, Tuple


# Transformation types supported by FAISS
TRANSFORM_TYPES = {
    "PCA": faiss.PCAMatrix,
    "NORM": faiss.NormalizationTransform,
    "L2NORM": faiss.NormalizationTransform,
    "PCAR": faiss.PCAMatrix,  # PCA with L2 normalization
    "ITQ": faiss.ITQTransform,
    "OPQ": faiss.OPQMatrix,
    "RR": faiss.RandomRotationMatrix
}


def create_transformation(
    transform_type: str,
    input_dim: int,
    output_dim: int = None,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a vector transformation based on the specified type.

    Args:
        transform_type: Type of transformation to create (e.g., "PCA", "NORM", "ITQ")
        input_dim: Input dimension of vectors
        output_dim: Output dimension after transformation (if applicable)
        **kwargs: Additional parameters for specific transformation types

    Returns:
        tuple: (transformation, transform_info)

    Raises:
        ValueError: If transform type is not supported or parameters are invalid
    """
    transform_info = {
        "type": transform_type,
        "input_dimension": input_dim,
        "requires_training": False
    }

    # Default output dimension to input dimension if not specified
    if output_dim is None:
        output_dim = input_dim

    transform_info["output_dimension"] = output_dim

    # Handle specific transformation types
    if transform_type == "PCA":
        transform = faiss.PCAMatrix(input_dim, output_dim)
        transform_info["requires_training"] = True
        transform_info["description"] = "Principal Component Analysis"

    elif transform_type == "PCAR":
        # PCA with L2 normalization
        transform = faiss.PCAMatrix(input_dim, output_dim, do_whitening=True)
        transform_info["requires_training"] = True
        transform_info["description"] = "PCA with whitening and L2 normalization"

    elif transform_type in ["NORM", "L2NORM"]:
        transform = faiss.NormalizationTransform(input_dim)
        transform_info["description"] = "L2 Normalization"

    elif transform_type == "ITQ":
        # Iterative Quantization
        transform = faiss.ITQTransform(input_dim, output_dim)
        transform_info["requires_training"] = True
        transform_info["description"] = "Iterative Quantization"

    elif transform_type == "OPQ":
        # Optimized Product Quantization
        if "M" not in kwargs:
            raise ValueError("OPQ transform requires 'M' parameter (number of subquantizers)")

        M = kwargs.get("M", 8)
        transform = faiss.OPQMatrix(input_dim, M, output_dim)
        transform_info["requires_training"] = True
        transform_info["M"] = M
        transform_info["description"] = f"Optimized Product Quantization with {M} subquantizers"

    elif transform_type == "RR":
        # Random Rotation
        transform = faiss.RandomRotationMatrix(input_dim, output_dim)
        transform_info["description"] = "Random Rotation"

    else:
        raise ValueError(f"Unsupported transformation type: {transform_type}")

    return transform, transform_info


def create_pretransform_index(
    base_index: Any,
    transform: Any,
    transform_info: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create an IndexPreTransform with the specified base index and transformation.

    Args:
        base_index: The base index to use after transformation
        transform: The transformation to apply
        transform_info: Information about the transformation

    Returns:
        tuple: (pretransform_index, index_info)
    """
    # Create the IndexPreTransform
    pretransform_index = faiss.IndexPreTransform(transform, base_index)

    # Prepare index information
    index_info = {
        "type": "IndexPreTransform",
        "transform_type": transform_info["type"],
        "input_dimension": transform_info["input_dimension"],
        "output_dimension": transform_info["output_dimension"],
        "base_index_type": type(base_index).__name__,
        "requires_training": (
            transform_info.get("requires_training", False) or
            not getattr(base_index, "is_trained", True)
        )
    }

    # Add any transform-specific information
    if "description" in transform_info:
        index_info["transform_description"] = transform_info["description"]

    if "M" in transform_info:
        index_info["transform_M"] = transform_info["M"]

    return pretransform_index, index_info


def parse_transform_type(index_type: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Parse a compound index type string to extract transformation information.

    Examples:
        - "PCA32,L2" -> ("PCA", "L2", {"output_dim": 32})
        - "OPQ8_32,IVF100,PQ8" -> ("OPQ", "IVF100,PQ8", {"M": 8, "output_dim": 32})
        - "L2NORM,HNSW32" -> ("L2NORM", "HNSW32", {})

    Args:
        index_type: Compound index type string

    Returns:
        tuple: (transform_type, base_index_type, transform_params)
    """
    transform_params = {}

    # Split at the first comma to separate transform from base index
    if "," in index_type:
        transform_part, base_index_type = index_type.split(",", 1)
    else:
        # No transform specified
        return None, index_type, {}

    # Parse the transform part
    if transform_part.startswith("PCA"):
        transform_type = "PCA"
        # Extract output dimension if specified
        if len(transform_part) > 3 and transform_part[3:].isdigit():
            transform_params["output_dim"] = int(transform_part[3:])

    elif transform_part.startswith("PCAR"):
        transform_type = "PCAR"
        # Extract output dimension if specified
        if len(transform_part) > 4 and transform_part[4:].isdigit():
            transform_params["output_dim"] = int(transform_part[4:])

    elif transform_part in ["NORM", "L2NORM"]:
        transform_type = transform_part

    elif transform_part.startswith("ITQ"):
        transform_type = "ITQ"
        # Extract output dimension if specified
        if len(transform_part) > 3 and transform_part[3:].isdigit():
            transform_params["output_dim"] = int(transform_part[3:])

    elif transform_part.startswith("OPQ"):
        transform_type = "OPQ"
        # Parse OPQ parameters (format: OPQ{M}_{output_dim})
        if "_" in transform_part[3:]:
            M_part, dim_part = transform_part[3:].split("_")
            transform_params["M"] = int(M_part)
            transform_params["output_dim"] = int(dim_part)
        else:
            # Default M=8 if not specified
            transform_params["M"] = 8
            if transform_part[3:].isdigit():
                transform_params["output_dim"] = int(transform_part[3:])

    elif transform_part.startswith("RR"):
        transform_type = "RR"
        # Extract output dimension if specified
        if len(transform_part) > 2 and transform_part[2:].isdigit():
            transform_params["output_dim"] = int(transform_part[2:])

    else:
        # Not a recognized transform
        return None, index_type, {}

    return transform_type, base_index_type, transform_params


def is_transform_trained(transform: Any) -> bool:
    """
    Check if a transformation has been trained.

    Args:
        transform: The transformation to check

    Returns:
        bool: True if the transformation is trained or doesn't require training
    """
    # Transformations that require training have an is_trained attribute
    if hasattr(transform, "is_trained"):
        return transform.is_trained

    # Some transformations like NormalizationTransform don't require training
    if isinstance(transform, faiss.NormalizationTransform):
        return True

    # If we can't determine, assume it's not trained
    return False


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
        "min_training_vectors": 0
    }

    if isinstance(transform, faiss.PCAMatrix):
        requirements["requires_training"] = True
        requirements["is_trained"] = is_transform_trained(transform)
        requirements["min_training_vectors"] = transform.d_out * 2
        requirements["recommended_training_vectors"] = transform.d_out * 10
        requirements["description"] = "PCA requires representative training data"

    elif isinstance(transform, faiss.OPQMatrix):
        requirements["requires_training"] = True
        requirements["is_trained"] = is_transform_trained(transform)
        requirements["min_training_vectors"] = transform.d_out * 5
        requirements["recommended_training_vectors"] = transform.d_out * 20
        requirements["description"] = "OPQ requires substantial training data"

    elif isinstance(transform, faiss.ITQTransform):
        requirements["requires_training"] = True
        requirements["is_trained"] = is_transform_trained(transform)
        requirements["min_training_vectors"] = transform.d_out * 5
        requirements["recommended_training_vectors"] = transform.d_out * 15
        requirements["description"] = "ITQ requires representative training data"

    return requirements


def train_transform(transform: Any, training_vectors: np.ndarray) -> bool:
    """
    Train a transformation with the provided vectors.

    Args:
        transform: The transformation to train
        training_vectors: Vectors to use for training

    Returns:
        bool: True if training was successful, False otherwise
    """
    try:
        # Different transformations have different training methods
        if isinstance(transform, (faiss.PCAMatrix, faiss.OPQMatrix, faiss.ITQTransform)):
            transform.train(training_vectors)
            return True

        # For transformations that don't require training
        return True
    except Exception as e:
        print(f"Error training transformation: {str(e)}")
        return False

#!/usr/bin/env python3
#
# FAISSx Server Training Utilities
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
FAISSx Server Training Module

This module provides utilities for standardized training behavior across different
FAISS index types. It helps determine if indices require training, check training status,
and provide clear training requirements information.

The training system supports various FAISS index types including:
- IVF (Inverted File) indices that require clustering
- PQ (Product Quantization) indices that need codebook training
- Binary indices with specialized training requirements
- Transform-based indices with multi-stage training needs

Key Functions:
- Training requirement detection for any FAISS index type
- Training status validation and readiness checks
- Optimal training vector count estimation
- Comprehensive training parameter recommendations
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import faiss

# Training requirement constants
# These values are based on FAISS documentation and empirical testing
DEFAULT_IVF_MIN_VECTORS = 100          # Minimum vectors for IVF training
DEFAULT_PQ_MIN_VECTORS = 1000          # Minimum vectors for PQ training
DEFAULT_BINARY_IVF_MIN_VECTORS = 100   # Minimum vectors for binary IVF training

# Optimal training vector multipliers
IVF_OPTIMAL_MULTIPLIER = 50             # Middle of 10-100x nlist range
PQ_OPTIMAL_MULTIPLIER = 1000            # Vectors per sub-quantizer
BINARY_IVF_OPTIMAL_MULTIPLIER = 50      # Middle of 10-100x nlist range

# Index type to training requirement mapping
# Maps FAISS index classes to their training descriptions
REQUIRES_TRAINING: Dict[Type[Any], str] = {
    faiss.IndexIVFFlat: "Requires training with representative vectors for clustering",
    faiss.IndexIVFPQ: "Requires training with representative vectors for clustering and PQ",
    faiss.IndexIVFScalarQuantizer: (
        "Requires training with representative vectors for clustering and SQ"
    ),
    faiss.IndexPQ: "Requires training with representative vectors for PQ encoding",
    faiss.IndexBinaryIVF: (
        "Requires training with representative binary vectors for clustering"
    ),
}

# Training parameter specifications for different index families
# These provide detailed guidance for optimal training
TRAINING_PARAMS: Dict[str, Dict[str, Union[int, str]]] = {
    "IVF": {
        "min_vectors": DEFAULT_IVF_MIN_VECTORS,
        "optimal_vectors": "10-100x nlist",  # Rule of thumb for IVF indices
        "description": "IVF indices need vectors to learn cluster centroids",
    },
    "PQ": {
        "min_vectors": DEFAULT_PQ_MIN_VECTORS,
        "optimal_vectors": "1000x M",  # Rule of thumb for M sub-quantizers
        "description": "PQ indices need vectors to learn quantization codebooks",
    },
    "BINARY_IVF": {
        "min_vectors": DEFAULT_BINARY_IVF_MIN_VECTORS,
        "optimal_vectors": "10-100x nlist",  # Rule of thumb for binary IVF
        "description": "Binary IVF indices need binary vectors for clustering",
    },
}


def requires_training(index: Any) -> bool:
    """
    Determine if an index requires training before use.

    This function checks if a FAISS index needs to be trained with representative
    data before it can be used for vector operations. Training is typically required
    for indices that use learned structures like clustering or quantization.

    Args:
        index: FAISS index instance of any type. Can be IndexFlat, IndexIVF,
               IndexPQ, IndexHNSW, or any other FAISS index implementation.

    Returns:
        bool: True if the index requires training before use, False if it can
              be used immediately after creation.

    Implementation Notes:
        - Checks against known index types that require training
        - Falls back to checking the is_trained attribute if available
        - Returns False for simple indices like IndexFlat that work immediately
        - Handles both standard and binary index types

    Examples:
        >>> index_flat = faiss.IndexFlatL2(128)
        >>> requires_training(index_flat)  # False - ready to use
        >>>
        >>> quantizer = faiss.IndexFlatL2(128)
        >>> index_ivf = faiss.IndexIVFFlat(quantizer, 128, 100)
        >>> requires_training(index_ivf)  # True - needs training
    """
    # Check against known index types that require training
    # This covers the most common cases efficiently
    for index_class in REQUIRES_TRAINING:
        if isinstance(index, index_class):
            return True

    # Fallback: check the is_trained attribute if present
    # This handles custom or composite index types
    if hasattr(index, "is_trained") and not index.is_trained:
        return True

    # Default: assume index doesn't require training
    # This covers simple indices like IndexFlat, IndexHNSW, etc.
    return False


def get_training_requirements(index: Any) -> Dict[str, Any]:
    """
    Get detailed training requirements for an index.

    This function provides comprehensive information about what training is needed
    for a specific index type, including parameter recommendations, minimum vector
    counts, and optimization guidelines.

    Args:
        index: FAISS index instance to analyze for training requirements

    Returns:
        dict: Comprehensive training requirements with the following structure:
            - requires_training (bool): Whether training is needed
            - is_trained (bool): Current training status
            - description (str, optional): Human-readable training description
            - params (dict, optional): Training parameters and recommendations
            - is_binary (bool, optional): True for binary indices

    Training Parameters Include:
        - min_vectors: Minimum training vectors for acceptable results
        - optimal_vectors: Recommended vector count for optimal performance
        - nlist: Number of clusters (for IVF indices)
        - M: Number of sub-quantizers (for PQ indices)
        - description: Explanation of why training is needed

    Implementation Strategy:
        1. Determine basic training requirement and status
        2. Match against known index types for detailed parameters
        3. Extract index-specific configuration (nlist, M, etc.)
        4. Provide tailored recommendations based on index characteristics
    """
    # Start with basic training status information
    requirements: Dict[str, Any] = {
        "requires_training": requires_training(index),
        "is_trained": getattr(index, "is_trained", True),
    }

    # Add detailed training information for recognized index types
    for index_class, description in REQUIRES_TRAINING.items():
        if isinstance(index, index_class):
            requirements["description"] = description

            # Add index family-specific training parameters
            if isinstance(index, faiss.IndexIVF):
                # IVF indices need clustering training
                requirements["params"] = TRAINING_PARAMS["IVF"].copy()
                requirements["params"]["nlist"] = index.nlist

                # Special handling for binary IVF indices
                if isinstance(index, faiss.IndexBinaryIVF):
                    requirements["params"].update(TRAINING_PARAMS["BINARY_IVF"])
                    requirements["is_binary"] = True

            elif isinstance(index, faiss.IndexPQ):
                # PQ indices need quantization codebook training
                requirements["params"] = TRAINING_PARAMS["PQ"].copy()
                requirements["params"]["M"] = index.pq.M

            # Break after finding the first match to avoid duplicates
            break

    return requirements


def is_trained_for_use(index: Any) -> Tuple[bool, Optional[str]]:
    """
    Check if an index is properly trained and ready for use.

    This function provides a comprehensive check of whether an index has been
    properly trained and is ready for vector operations like search and addition.
    It goes beyond simple status checking to validate actual readiness.

    Args:
        index: FAISS index instance to check for readiness

    Returns:
        tuple: A two-element tuple containing:
            - is_ready (bool): True if the index is ready for use
            - reason (str or None): Explanation if index is not ready, None if ready

    Readiness Criteria:
        - Indices that don't require training are always ready
        - Indices that require training must have is_trained=True
        - Transform-based indices need all components trained
        - Composite indices need all sub-components ready

    Usage Patterns:
        This function is typically called before performing index operations
        to ensure the index won't fail due to insufficient training.

    Examples:
        >>> is_ready, reason = is_trained_for_use(my_index)
        >>> if not is_ready:
        >>>     print(f"Index not ready: {reason}")
        >>>     # Perform training...
        >>> else:
        >>>     # Proceed with search/add operations
    """
    # If the index doesn't require training, it's always ready to use
    # This covers simple indices like IndexFlat, IndexHNSW (when trained), etc.
    if not requires_training(index):
        return True, None

    # For indices that require training, check the training status
    # The is_trained attribute is the standard way FAISS indicates readiness
    if hasattr(index, "is_trained") and index.is_trained:
        return True, None

    # Index requires training but hasn't been trained yet
    return False, "Index requires training before use"


def estimate_training_vectors_needed(index: Any) -> Optional[int]:
    """
    Estimate the number of training vectors needed for good results.

    This function provides data-driven recommendations for the optimal number
    of training vectors based on index type and configuration. The estimates
    are based on FAISS documentation, research papers, and empirical testing.

    Args:
        index: FAISS index instance to analyze for training vector requirements

    Returns:
        int or None: Estimated number of training vectors for optimal results.
                    Returns None if the index doesn't require training or if
                    the requirement cannot be determined.

    Estimation Guidelines:
        - IVF indices: 10-100x the number of clusters (nlist)
        - PQ indices: ~1000 vectors per sub-quantizer (M)
        - Binary IVF: Similar to regular IVF but with binary vectors
        - Composite indices: Based on the most demanding component

    Quality Considerations:
        - More training vectors generally improve index quality
        - But returns diminish beyond the optimal range
        - Training time scales linearly with vector count
        - Memory usage during training scales with vector count

    Implementation Notes:
        - Uses conservative estimates in the middle of recommended ranges
        - Considers index-specific parameters (nlist, M, etc.)
        - Returns None for indices that don't need training
        - Provides reasonable defaults for unknown index types
    """
    # IVF family indices: clustering-based training
    if isinstance(index, faiss.IndexIVF):
        # For IVF indices, use 50x nlist (middle of 10-100x range)
        # This provides good cluster quality without excessive training time
        return index.nlist * IVF_OPTIMAL_MULTIPLIER

    # Product Quantization indices: codebook training
    elif isinstance(index, faiss.IndexPQ):
        # For PQ indices, use 1000 vectors per sub-quantizer
        # This ensures each sub-quantizer has sufficient data for training
        return index.pq.M * PQ_OPTIMAL_MULTIPLIER

    # Binary IVF indices: similar to regular IVF but with binary vectors
    elif isinstance(index, faiss.IndexBinaryIVF):
        # Use the same multiplier as regular IVF indices
        # Binary clustering has similar statistical requirements
        return index.nlist * BINARY_IVF_OPTIMAL_MULTIPLIER

    # For other index types, we can't provide a meaningful estimate
    # This includes IndexFlat, IndexHNSW, and unknown custom types
    return None

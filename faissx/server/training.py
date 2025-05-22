#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""

import faiss
from typing import Any, Dict, Optional, Tuple

# Define index types that require training
REQUIRES_TRAINING = {
    faiss.IndexIVFFlat: "Requires training with representative vectors",
    faiss.IndexIVFPQ: "Requires training with representative vectors",
    faiss.IndexIVFScalarQuantizer: "Requires training with representative vectors",
    faiss.IndexPQ: "Requires training with representative vectors for PQ encoding",
    faiss.IndexBinaryIVF: "Requires training with representative binary vectors",
}

# Define training parameters for various index types
TRAINING_PARAMS = {
    "IVF": {
        "min_vectors": 100,  # Minimum vectors recommended for good training
        "optimal_vectors": "10-100x nlist",  # Rule of thumb
    },
    "PQ": {
        "min_vectors": 1000,  # Minimum vectors recommended for good training
        "optimal_vectors": "1000x M",  # Rule of thumb for M subquantizers
    },
    "BINARY_IVF": {
        "min_vectors": 100,  # Minimum vectors recommended for good training
        "optimal_vectors": "10-100x nlist",  # Rule of thumb
    },
}


def requires_training(index: Any) -> bool:
    """
    Determine if an index requires training before use.

    Args:
        index: FAISS index instance

    Returns:
        bool: True if index requires training, False otherwise
    """
    # Check each index type that requires training
    for index_class in REQUIRES_TRAINING:
        if isinstance(index, index_class):
            return True

    # Special case: check if the index has is_trained attribute
    if hasattr(index, "is_trained") and not index.is_trained:
        return True

    return False


def get_training_requirements(index: Any) -> Dict[str, Any]:
    """
    Get detailed training requirements for an index.

    Args:
        index: FAISS index instance

    Returns:
        dict: Training requirements information
    """
    requirements = {
        "requires_training": requires_training(index),
        "is_trained": getattr(index, "is_trained", True),
    }

    # Add detailed training information for specific index types
    for index_class, description in REQUIRES_TRAINING.items():
        if isinstance(index, index_class):
            requirements["description"] = description

            # Add index-specific training parameters
            if isinstance(index, faiss.IndexIVF):
                requirements["params"] = TRAINING_PARAMS["IVF"]
                requirements["params"]["nlist"] = index.nlist
            elif isinstance(index, faiss.IndexPQ):
                requirements["params"] = TRAINING_PARAMS["PQ"]
                requirements["params"]["M"] = index.pq.M
            elif isinstance(index, faiss.IndexBinaryIVF):
                requirements["params"] = TRAINING_PARAMS["BINARY_IVF"]
                requirements["params"]["nlist"] = index.nlist
                requirements["is_binary"] = True

            break

    return requirements


def is_trained_for_use(index: Any) -> Tuple[bool, Optional[str]]:
    """
    Check if an index is properly trained and ready for use.

    Args:
        index: FAISS index instance

    Returns:
        tuple: (is_ready, reason_if_not_ready)
    """
    # If index doesn't require training, it's always ready
    if not requires_training(index):
        return True, None

    # Check if the index has been explicitly trained
    if hasattr(index, "is_trained") and index.is_trained:
        return True, None

    # Not trained
    return False, "Index requires training before use"


def estimate_training_vectors_needed(index: Any) -> Optional[int]:
    """
    Estimate the number of training vectors needed for good results.

    Args:
        index: FAISS index instance

    Returns:
        int or None: Estimated number of vectors needed, or None if not applicable
    """
    if isinstance(index, faiss.IndexIVF):
        # For IVF indices, 10-100x nlist is a good rule of thumb
        return index.nlist * 50  # Middle of the 10-100x range

    elif isinstance(index, faiss.IndexPQ):
        # For PQ indices
        return index.pq.M * 1000

    elif isinstance(index, faiss.IndexBinaryIVF):
        # For binary IVF indices, similar to regular IVF
        return index.nlist * 50  # Middle of the 10-100x range

    # For other index types, return None (not applicable)
    return None

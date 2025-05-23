#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Get Parameter
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
FAISS Index Parameter Retrieval Module

This module provides comprehensive parameter retrieval capabilities for various FAISS index types.
It supports a wide range of index parameters across different FAISS implementations including:

Supported Index Types:
- IVF (Inverted File) indices with configurable probe parameters
- HNSW (Hierarchical Navigable Small World) indices with search parameters
- PQ (Product Quantization) indices with quantization configuration
- SQ (Scalar Quantization) indices with quantization type settings
- Binary indices with specialized binary vector parameters
- Hash-based binary indices with bit configuration

Common Parameters:
- Universal parameters like dimension, training status, and vector count
- Metric type information (L2, Inner Product)
- Index-specific performance tuning parameters

Usage:
This module is designed to be integrated with the FAISSx server to provide runtime
parameter introspection capabilities for monitoring and debugging purposes.
"""

import faiss  # Facebook AI Similarity Search library
import logging  # Structured logging for monitoring and debugging
from typing import Dict, Union  # Type hints for better code safety

# Logging configuration for parameter retrieval operations
logger = logging.getLogger(__name__)

# Constants for parameter names to avoid magic strings and improve maintainability
# IVF Index Parameters
PARAM_NPROBE = "nprobe"  # Number of clusters to probe during search
PARAM_NLIST = "nlist"  # Number of clusters in IVF index
PARAM_QUANTIZER_TYPE = "quantizer_type"  # Type of quantizer used

# HNSW Index Parameters
PARAM_EF_SEARCH = "efSearch"  # Size of candidate list during search
PARAM_EF_CONSTRUCTION = "efConstruction"  # Size of candidate list during construction
PARAM_M = "M"  # Number of bi-directional links for each node

# PQ Index Parameters
PARAM_PQ_M = "M"  # Number of sub-quantizers in PQ
PARAM_NBITS = "nbits"  # Number of bits per sub-quantizer
PARAM_USE_PRECOMPUTED_TABLE = "use_precomputed_table"  # Use precomputed distance tables

# SQ Index Parameters
PARAM_QTYPE = "qtype"  # Scalar quantization type

# Binary Index Parameters
PARAM_BITS_PER_DIM = "bits_per_dim"  # Bits per dimension for binary hash

# Universal Index Parameters
PARAM_IS_TRAINED = "is_trained"  # Whether index has been trained
PARAM_DIMENSION = "dimension"  # Vector dimension
PARAM_NTOTAL = "ntotal"  # Total number of vectors in index
PARAM_METRIC_TYPE = "metric_type"  # Distance metric used (L2, IP)

# Metric type mappings for consistent string representation
METRIC_L2 = "L2"  # L2 (Euclidean) distance metric
METRIC_IP = "IP"  # Inner Product distance metric

# Response structure constants
RESPONSE_SUCCESS = "success"  # Success indicator in response
RESPONSE_ERROR = "error"  # Error message in response
RESPONSE_PARAM_NAME = "param_name"  # Parameter name in response
RESPONSE_PARAM_VALUE = "param_value"  # Parameter value in response


def _create_success_response(
    param_name: str, param_value: Union[str, int, float, bool]
) -> Dict[str, Union[bool, str, int, float]]:
    """
    Create a standardized success response for parameter retrieval.

    Args:
        param_name: Name of the parameter that was retrieved
        param_value: Value of the parameter

    Returns:
        Dictionary with success response format
    """
    return {
        RESPONSE_SUCCESS: True,
        RESPONSE_PARAM_NAME: param_name,
        RESPONSE_PARAM_VALUE: param_value,
    }


def _create_error_response(error_message: str) -> Dict[str, Union[bool, str]]:
    """
    Create a standardized error response for parameter retrieval.

    Args:
        error_message: Descriptive error message

    Returns:
        Dictionary with error response format
    """
    return {RESPONSE_SUCCESS: False, RESPONSE_ERROR: error_message}


def get_parameter(
    self, index_id: str, param_name: str
) -> Dict[str, Union[bool, str, int, float]]:
    """
    Retrieve the current value of a specified parameter for a FAISS index.

    This function provides comprehensive parameter introspection capabilities for various
    FAISS index types. It supports both index-specific parameters (like nprobe for IVF
    indices) and universal parameters (like dimension and training status).

    Args:
        self: Index manager instance containing the indices dictionary
        index_id: Unique identifier for the target index
        param_name: Name of the parameter to retrieve (see module constants)

    Returns:
        Dictionary containing either:
        - Success response: {success: True, param_name: str, param_value: value}
        - Error response: {success: False, error: str}

    Supported Parameters by Index Type:
        IVF Indices: nprobe, nlist, quantizer_type
        HNSW Indices: efSearch, efConstruction, M
        PQ Indices: M, nbits, use_precomputed_table
        SQ Indices: qtype
        Binary IVF: nprobe, nlist
        Binary Hash: bits_per_dim
        Universal: is_trained, dimension, ntotal, metric_type

    Example:
        >>> result = manager.get_parameter("index_123", "nprobe")
        >>> if result["success"]:
        ...     print(f"nprobe value: {result['param_value']}")

    Note:
        This function is designed to be used as a method of an index manager class
        that maintains an 'indexes' attribute containing FAISS index instances.
    """
    # Validate that the requested index exists
    if index_id not in self.indexes:
        return _create_error_response(f"Index {index_id} not found")

    index = self.indexes[index_id]

    try:
        # IVF (Inverted File) index parameter handling
        # Supports configurable probe parameters for search quality vs speed trade-offs
        if isinstance(index, faiss.IndexIVF):
            if param_name == PARAM_NPROBE:
                return _create_success_response(PARAM_NPROBE, index.nprobe)
            elif param_name == PARAM_NLIST:
                return _create_success_response(PARAM_NLIST, index.nlist)
            elif param_name == PARAM_QUANTIZER_TYPE and hasattr(index, "quantizer"):
                quantizer_name = type(index.quantizer).__name__
                return _create_success_response(PARAM_QUANTIZER_TYPE, quantizer_name)

        # HNSW (Hierarchical Navigable Small World) index parameter handling
        # Graph-based indices with configurable search and construction parameters
        elif isinstance(index, faiss.IndexHNSW):
            if param_name == PARAM_EF_SEARCH:
                return _create_success_response(PARAM_EF_SEARCH, index.hnsw.efSearch)
            elif param_name == PARAM_EF_CONSTRUCTION:
                return _create_success_response(
                    PARAM_EF_CONSTRUCTION, index.hnsw.efConstruction
                )
            elif param_name == PARAM_M:
                return _create_success_response(PARAM_M, index.hnsw.M)

        # Product Quantization (PQ) index parameters
        elif isinstance(index, faiss.IndexPQ):
            if param_name == "M":
                return {"success": True, "param_name": "M", "param_value": index.pq.M}
            elif param_name == "nbits":
                return {
                    "success": True,
                    "param_name": "nbits",
                    "param_value": index.pq.nbits,
                }
            elif param_name == "use_precomputed_table":
                return {
                    "success": True,
                    "param_name": "use_precomputed_table",
                    "param_value": getattr(index, "use_precomputed_table", False),
                }

        # Scalar Quantization (SQ) index parameters
        elif isinstance(index, faiss.IndexScalarQuantizer):
            if param_name == "qtype":
                return {
                    "success": True,
                    "param_name": "qtype",
                    "param_value": str(index.sq_type),
                }

        # Binary index parameters
        elif isinstance(index, faiss.IndexBinaryIVF):
            if param_name == "nprobe":
                return {
                    "success": True,
                    "param_name": "nprobe",
                    "param_value": index.nprobe,
                }
            elif param_name == "nlist":
                return {
                    "success": True,
                    "param_name": "nlist",
                    "param_value": index.nlist,
                }

        # Binary hash index parameters
        elif isinstance(index, faiss.IndexBinaryHash):
            if param_name == "bits_per_dim":
                return {
                    "success": True,
                    "param_name": "bits_per_dim",
                    "param_value": index.b,
                }

        # Parameters for all index types
        if param_name == "is_trained":
            # This parameter is available for all index types
            is_trained = getattr(index, "is_trained", True)
            return {
                "success": True,
                "param_name": "is_trained",
                "param_value": is_trained,
            }
        elif param_name == "dimension":
            return {
                "success": True,
                "param_name": "dimension",
                "param_value": self.dimensions[index_id],
            }
        elif param_name == "ntotal":
            return {
                "success": True,
                "param_name": "ntotal",
                "param_value": index.ntotal,
            }
        elif param_name == "metric_type":
            if hasattr(index, "metric_type"):
                metric_type = index.metric_type
                metric_name = "L2"
                if metric_type == faiss.METRIC_INNER_PRODUCT:
                    metric_name = "IP"
                return {
                    "success": True,
                    "param_name": "metric_type",
                    "param_value": metric_name,
                }

        # If we get here, the parameter is not supported
        return {
            "success": False,
            "error": (
                f"Parameter {param_name} not supported for this index type "
                f"({type(index).__name__})"
            ),
        }

    except Exception as e:
        logger.exception(f"Error getting parameter: {e}")
        return {"success": False, "error": f"Error getting parameter: {str(e)}"}

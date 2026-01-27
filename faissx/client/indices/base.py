#!/usr/bin/env python3
#
# Unified interface for LLM providers using OpenAI format
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
Base module for common imports and utilities used by all index classes.

This module provides the foundation for FAISSx index implementations, including
common imports, constants, and the base index class with shared functionality.

It centralizes common operations and interfaces, ensuring consistent behavior
across different index types, and integrates with the FAISSx memory management
system for efficient resource utilization.
"""

import logging
import uuid
from typing import Any, Dict, Tuple, Union

import faiss
import numpy as np

# Define fallback constants for environments where FAISS isn't fully available
# This ensures basic functionality even without the full FAISS library
if not hasattr(faiss, "METRIC_L2"):
    class FaissConstants:
        """
        Constants for FAISS metrics when full library is unavailable.

        Provides basic metric definitions to maintain functionality in environments
        where the complete FAISS library cannot be loaded. This allows FAISSx
        to operate in remote mode even when FAISS is not installed locally.
        """
        METRIC_L2 = 0  # L2 (Euclidean) distance metric
        METRIC_INNER_PRODUCT = 1  # Inner product (dot product) metric

    faiss = FaissConstants()  # type: ignore

from ..client import get_client
from ..optimization import IndexParameters, memory_manager

# Define type for parameter values that can be stored in IndexParameters
ParamValueType = Union[int, float, bool, str, Dict[str, Any], list]

# Define public exports for this module
__all__ = [
    "uuid", "np", "Tuple", "faiss", "logging", "get_client",
    "FAISSxBaseIndex"  # Export the base index class
]

# Configure module-level logger
logger = logging.getLogger(__name__)


class FAISSxBaseIndex:
    """
    Base class for all FAISSx indices.

    This class provides common functionality and a consistent interface for all FAISSx index
    implementations. It handles parameter management, memory tracking, and provides the
    foundation for index operations.

    All index implementations should inherit from this class to ensure consistent
    behavior and access to core utilities like parameter management and memory tracking.

    Attributes:
        _params (IndexParameters): Manager for index-specific parameters and settings
    """

    def __init__(self) -> None:
        """
        Initialize the base index with default parameters.

        Creates a new IndexParameters instance to manage index-specific settings
        and configurations. This enables consistent parameter handling across all
        index implementations.
        """
        self._params = IndexParameters(self)

    def register_access(self) -> None:
        """
        Register an access to this index for memory management purposes.

        This method should be called at the beginning of any operation that uses the index
        (e.g., search, add, etc.) to track usage patterns and optimize memory allocation.

        The memory manager uses this information to make informed decisions about resource
        allocation, prioritization, and potential cleanup of unused indices.
        """
        memory_manager.register_index_access(self)

    def get_parameter(self, name: str) -> ParamValueType:
        """
        Get a parameter value for this index.

        Retrieves the current value of the specified parameter. Parameters control
        various aspects of index behavior like batch sizes, search quality settings,
        and optimization factors.

        Args:
            name: The name of the parameter to retrieve

        Returns:
            The current value of the specified parameter

        Raises:
            ValueError: If the parameter does not exist for this index type
        """
        try:
            return self._params.get_parameter(name)
        except KeyError as e:
            # Convert KeyError to ValueError with a more informative message
            raise ValueError(f"Parameter '{name}' does not exist for this index type") from e

    def set_parameter(self, name: str, value: ParamValueType) -> None:
        """
        Set a parameter value for this index.

        Updates the specified parameter with a new value. Different index types
        support different sets of parameters that control their behavior and
        performance characteristics.

        Args:
            name: The name of the parameter to set
            value: The new value for the parameter

        Raises:
            ValueError: If the parameter is invalid or incompatible with this index type
        """
        self._params.set_parameter(name, value)

    def get_parameters(self) -> Dict[str, ParamValueType]:
        """
        Get all parameters applicable to this index.

        Returns a dictionary containing all configurable parameters and their
        current values for this index. Useful for inspecting the current
        configuration or saving/restoring index settings.

        Returns:
            A dictionary mapping parameter names to their current values.
            Includes all configurable parameters for this index type.
        """
        return self._params.get_all_parameters()

    def reset_parameters(self) -> None:
        """
        Reset all parameters to their default values.

        This method restores all index parameters to their initial configuration
        as defined in the IndexParameters class. Useful when you want to clear
        all customizations and revert to default behavior.
        """
        self._params.reset_parameters()

    def estimate_memory_usage(self) -> int:
        """
        Estimate the memory usage of this index in bytes.

        Provides an approximation of the total memory required by the index,
        including vectors, metadata, and internal structures. This estimate
        can be used for resource planning, monitoring, and optimization.

        The accuracy of the estimate may vary depending on the index type
        and configuration.

        Returns:
            The estimated memory usage in bytes, including all index components
        """
        return memory_manager.estimate_index_size(self)

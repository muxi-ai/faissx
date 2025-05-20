#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""

import uuid
import numpy as np
from typing import Tuple, Any
import faiss
import logging

# Define fallback constants for environments where FAISS isn't fully available
# This ensures basic functionality even without the full FAISS library
if not hasattr(faiss, "METRIC_L2"):

    class FaissConstants:
        METRIC_L2 = 0  # L2 (Euclidean) distance metric
        METRIC_INNER_PRODUCT = 1  # Inner product (dot product) metric

    faiss = FaissConstants()

from ..client import get_client

# Import optimization controls for memory and performance management
from ..optimization import IndexParameters, memory_manager

# Define public exports for this module
__all__ = ["uuid", "np", "Tuple", "faiss", "logging", "get_client"]

# Configure module-level logger
logger = logging.getLogger(__name__)


class FAISSxBaseIndex:
    """
    Base class for all FAISSx indices.

    This class provides common functionality and a consistent interface for all FAISSx index
    implementations. It handles parameter management, memory tracking, and provides the
    foundation for index operations.
    """

    def __init__(self):
        """
        Initialize the base index.

        Creates a new IndexParameters instance to manage index-specific configuration
        and optimization settings.
        """
        self._params = IndexParameters(self)

    def register_access(self):
        """
        Register an access to this index for memory management purposes.

        This method is called whenever the index is accessed to track usage patterns
        and optimize memory allocation.
        """
        memory_manager.register_index_access(self)

    def get_parameter(self, name: str) -> Any:
        """
        Get a parameter value for this index.

        Args:
            name: The name of the parameter to retrieve

        Returns:
            The current value of the specified parameter
        """
        return self._params.get_parameter(name)

    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set a parameter value for this index.

        Args:
            name: The name of the parameter to set
            value: The new value for the parameter
        """
        self._params.set_parameter(name, value)

    def get_parameters(self) -> dict:
        """
        Get all parameters applicable to this index.

        Returns:
            A dictionary mapping parameter names to their current values
        """
        return self._params.get_all_parameters()

    def reset_parameters(self) -> None:
        """
        Reset all parameters to their default values.

        This method restores all index parameters to their initial configuration
        as defined in the IndexParameters class.
        """
        self._params.reset_parameters()

    def estimate_memory_usage(self) -> int:
        """
        Estimate the memory usage of this index in bytes.

        This method provides an approximation of the total memory required by the
        index, including vectors, metadata, and internal structures.

        Returns:
            The estimated memory usage in bytes
        """
        return memory_manager.estimate_index_size(self)

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
"""

import uuid
import numpy as np
from typing import Tuple, Any, Optional
import faiss
import logging

# Define constants for when faiss isn't fully available
if not hasattr(faiss, 'METRIC_L2'):
    class FaissConstants:
        METRIC_L2 = 0
        METRIC_INNER_PRODUCT = 1
    faiss = FaissConstants()

from ..client import get_client

# Import the optimization controls
from ..optimization import IndexParameters, memory_manager

__all__ = ['uuid', 'np', 'Tuple', 'faiss', 'logging', 'get_client']

logger = logging.getLogger(__name__)

class FAISSxBaseIndex:
    """
    Base class for all FAISSx indices.

    This class provides common functionality and a consistent interface
    for all FAISSx index implementations.
    """

    def __init__(self):
        """Initialize the base index."""
        self._params = IndexParameters(self)

    def register_access(self):
        """
        Register an access to this index for memory management purposes.
        """
        memory_manager.register_index_access(self)

    def get_parameter(self, name: str) -> Any:
        """
        Get a parameter value.

        Args:
            name: Parameter name

        Returns:
            Current parameter value
        """
        return self._params.get_parameter(name)

    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set a parameter value.

        Args:
            name: Parameter name
            value: Parameter value
        """
        self._params.set_parameter(name, value)

    def get_parameters(self) -> dict:
        """
        Get all parameters applicable to this index.

        Returns:
            Dictionary of parameter name to value
        """
        return self._params.get_all_parameters()

    def reset_parameters(self) -> None:
        """Reset all parameters to their default values."""
        self._params.reset_parameters()

    def estimate_memory_usage(self) -> int:
        """
        Estimate the memory usage of this index in bytes.

        Returns:
            Estimated memory usage in bytes
        """
        return memory_manager.estimate_index_size(self)

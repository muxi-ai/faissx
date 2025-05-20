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
FAISSx optimization controls module.

This module provides fine-grained parameter controls and memory management options
for FAISSx indices. It allows customizing performance characteristics and resource
usage for both local and remote modes.

Key features:
1. Fine-grained parameters: Control search quality, training, and other performance settings
2. Memory management options: Configure memory usage, caching, and resource limits
"""

import logging
import threading
import math
from typing import Dict, Any, Set
import weakref

import faiss

logger = logging.getLogger(__name__)

# Global registry for index parameters - stores parameters per index to allow dynamic adjustments
_index_params: Dict[int, Dict[str, Any]] = {}
_registry_lock = threading.RLock()

# Global memory management settings with conservative defaults
_memory_options: Dict[str, Any] = {
    "max_memory_usage_mb": None,  # No memory limit by default
    "use_memory_mapping": False,  # Standard loading instead of memory mapping
    "index_cache_size": 100,  # Maximum number of indices to keep in memory
    "vector_cache_size_mb": 256,  # Size of vector cache in MB
    "auto_unload_unused_indices": False,  # Manual index unloading by default
    "io_buffer_size_kb": 1024,  # 1MB IO buffer for index operations
}


class IndexParameters:
    """
    Manages fine-grained parameters for FAISS indices.

    This class provides a unified interface for setting and getting performance-related parameters
    for indices, regardless of their type. It handles proper parameter validation and application
    for both local and remote index modes.
    """

    # Parameter definitions with validation rules and descriptions
    PARAMETER_DEFINITIONS = {
        # Search parameters that control query performance
        "nprobe": {
            "applicable_to": ["IVFFlat", "IVFPQ", "IVFScalarQuantizer"],
            "description": "Number of clusters to scan during search",
            "type": int,
            "min": 1,
            "max": 1024,
            "default": 1,
        },
        "efSearch": {
            "applicable_to": ["HNSW"],
            "description": "Exploration factor for HNSW search",
            "type": int,
            "min": 1,
            "max": 1024,
            "default": 16,
        },
        "k_factor": {
            "applicable_to": [
                "Flat",
                "IVFFlat",
                "IVFPQ",
                "HNSW",
                "PQ",
                "IVFScalarQuantizer",
            ],
            "description": "Multiply k by this factor internally (returns top k after reranking)",
            "type": float,
            "min": 1.0,
            "max": 10.0,
            "default": 1.0,
        },
        # Training parameters that affect index quality
        "n_iter": {
            "applicable_to": ["IVFFlat", "IVFPQ", "IVFScalarQuantizer"],
            "description": "Number of iterations when training the index",
            "type": int,
            "min": 1,
            "max": 1000,
            "default": 25,
        },
        "min_points_per_centroid": {
            "applicable_to": ["IVFFlat", "IVFPQ", "IVFScalarQuantizer"],
            "description": "Minimum number of points per centroid during training",
            "type": int,
            "min": 5,
            "max": 1000,
            "default": 39,
        },
        # HNSW-specific parameters for graph construction
        "efConstruction": {
            "applicable_to": ["HNSW"],
            "description": "Construction time exploration factor for HNSW",
            "type": int,
            "min": 8,
            "max": 512,
            "default": 40,
        },
        # Batch operation parameters for performance tuning
        "batch_size": {
            "applicable_to": [
                "Flat",
                "IVFFlat",
                "IVFPQ",
                "HNSW",
                "PQ",
                "IVFScalarQuantizer",
            ],
            "description": "Batch size for add/search operations",
            "type": int,
            "min": 1,
            "max": 1000000,
            "default": 10000,
        },
        # Quality vs speed tradeoff parameters
        "quantizer_effort": {
            "applicable_to": ["IVFScalarQuantizer"],
            "description": "Quantizer encoding effort (higher = better quality but slower)",
            "type": int,
            "min": 1,
            "max": 10,
            "default": 4,
        },
    }

    def __init__(self, index):
        """
        Initialize parameter management for a FAISS index.

        Args:
            index: The FAISSx index object to manage parameters for
        """
        # Store weakref to prevent circular references
        self._index_ref = weakref.ref(index)
        self._index_id = id(index)

        # Identify index type for parameter validation
        self._index_type = self._get_index_type(index)

        # Initialize parameters dictionary in the global registry
        with _registry_lock:
            if self._index_id not in _index_params:
                _index_params[self._index_id] = {}

    def _get_index_type(self, index) -> str:
        """
        Determine the type of the index for parameter validation.

        Args:
            index: FAISSx index object

        Returns:
            String representation of the index type
        """
        class_name = index.__class__.__name__

        # Map class names to parameter types
        if "IndexFlatL2" in class_name:
            return "Flat"
        elif "IndexIVFFlat" in class_name:
            return "IVFFlat"
        elif "IndexIVFPQ" in class_name:
            return "IVFPQ"
        elif "IndexHNSW" in class_name:
            return "HNSW"
        elif "IndexPQ" in class_name:
            return "PQ"
        elif "IndexScalarQuantizer" in class_name:
            return "IVFScalarQuantizer"
        else:
            # Default to Flat for unknown indices
            return "Flat"

    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set a parameter value with validation.

        Args:
            name: Parameter name
            value: Parameter value

        Raises:
            ValueError: If the parameter is not applicable, out of range, or invalid
        """
        # Check parameter exists and is applicable to this index type
        if name not in self.PARAMETER_DEFINITIONS:
            raise ValueError(f"Unknown parameter: {name}")

        param_def = self.PARAMETER_DEFINITIONS[name]

        # Check if parameter applies to this index type
        if self._index_type not in param_def["applicable_to"]:
            raise ValueError(
                f"Parameter '{name}' not applicable to index type {self._index_type}"
            )

        # Validate value type and range
        if not isinstance(value, param_def["type"]):
            raise ValueError(
                f"Param '{name}' expects {param_def['type'].__name__}, got {type(value).__name__}"
            )

        if "min" in param_def and value < param_def["min"]:
            raise ValueError(f"Parameter '{name}' must be >= {param_def['min']}")

        if "max" in param_def and value > param_def["max"]:
            raise ValueError(f"Parameter '{name}' must be <= {param_def['max']}")

        # Store in registry
        with _registry_lock:
            _index_params[self._index_id][name] = value

        # Apply to index if it still exists and the parameter can be applied directly
        index = self._index_ref() if self._index_ref else None
        if index is not None:
            self._apply_parameter(index, name, value)

    def get_parameter(self, name: str) -> Any:
        """
        Get a parameter value.

        Args:
            name: Parameter name

        Returns:
            Current parameter value, or the default if not set

        Raises:
            ValueError: If the parameter is not applicable to this index type
        """
        # Check parameter exists and is applicable to this index type
        if name not in self.PARAMETER_DEFINITIONS:
            raise ValueError(f"Unknown parameter: {name}")

        param_def = self.PARAMETER_DEFINITIONS[name]

        # Check if parameter applies to this index type
        if self._index_type not in param_def["applicable_to"]:
            raise ValueError(
                f"Parameter '{name}' not applicable to index type {self._index_type}"
            )

        # Get from registry or return default
        with _registry_lock:
            if (
                self._index_id in _index_params
                and name in _index_params[self._index_id]
            ):
                return _index_params[self._index_id][name]
            return param_def["default"]

    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Get all parameters applicable to this index.

        Returns:
            Dictionary of parameter name to value
        """
        result = {}

        # Get applicable parameters
        for name, param_def in self.PARAMETER_DEFINITIONS.items():
            if self._index_type in param_def["applicable_to"]:
                result[name] = self.get_parameter(name)

        return result

    def reset_parameters(self) -> None:
        """
        Reset all parameters to their default values.
        """
        with _registry_lock:
            if self._index_id in _index_params:
                _index_params[self._index_id] = {}

    def _apply_parameter(self, index, name: str, value: Any) -> None:
        """
        Apply a parameter directly to the index if possible.

        Args:
            index: The index object
            name: Parameter name
            value: Parameter value
        """
        # Some parameters can be applied directly
        try:
            if name == "nprobe" and hasattr(index, "_nprobe"):
                index._nprobe = value
                # If local index exists, set it there too
                if hasattr(index, "_local_index") and index._local_index is not None:
                    index._local_index.nprobe = value

            elif name == "efSearch" and hasattr(index, "_ef"):
                index._ef = value
                # If local index exists, set it there too
                if hasattr(index, "_local_index") and index._local_index is not None:
                    index._local_index.hnsw.efSearch = value

            elif name == "efConstruction" and hasattr(index, "_efc"):
                # This usually can't be changed after construction
                # But we store it for future reference
                index._efc = value

        except Exception as e:
            logger.warning(f"Failed to apply parameter {name}={value} to index: {e}")


class MemoryManager:
    """
    Manages memory usage for FAISS indices.

    This class provides utilities to control memory usage,
    including memory mapping, caching, and resource limits.
    """

    def __init__(self):
        """Initialize the memory manager."""
        self._active_indices: Set[int] = set()
        self._last_accessed: Dict[int, float] = {}
        self._index_sizes: Dict[int, int] = {}  # Estimated size in bytes
        self._unload_thread = None
        self._running = False

    @staticmethod
    def set_option(name: str, value: Any) -> None:
        """
        Set a global memory management option.

        Args:
            name: Option name
            value: Option value

        Raises:
            ValueError: If the option is invalid
        """
        if name not in _memory_options:
            raise ValueError(f"Unknown memory option: {name}")

        # Validate some options
        if name == "max_memory_usage_mb" and value is not None and value <= 0:
            raise ValueError("max_memory_usage_mb must be positive or None")

        if name == "index_cache_size" and value <= 0:
            raise ValueError("index_cache_size must be positive")

        if name == "vector_cache_size_mb" and value <= 0:
            raise ValueError("vector_cache_size_mb must be positive")

        # Set the option
        _memory_options[name] = value

    @staticmethod
    def get_option(name: str) -> Any:
        """
        Get a global memory management option.

        Args:
            name: Option name

        Returns:
            Current option value

        Raises:
            ValueError: If the option is invalid
        """
        if name not in _memory_options:
            raise ValueError(f"Unknown memory option: {name}")

        return _memory_options[name]

    @staticmethod
    def get_all_options() -> Dict[str, Any]:
        """
        Get all memory management options.

        Returns:
            Dictionary of all options
        """
        return dict(_memory_options)

    @staticmethod
    def reset_options() -> None:
        """
        Reset all memory options to defaults.
        """
        _memory_options["max_memory_usage_mb"] = None
        _memory_options["use_memory_mapping"] = False
        _memory_options["index_cache_size"] = 100
        _memory_options["vector_cache_size_mb"] = 256
        _memory_options["auto_unload_unused_indices"] = False
        _memory_options["io_buffer_size_kb"] = 1024

    @staticmethod
    def get_io_flags() -> int:
        """
        Get appropriate IO flags for FAISS based on current memory options.

        Returns:
            FAISS IO flags value
        """
        # Start with default flags
        flags = 0

        # Add memory mapping if enabled
        if _memory_options["use_memory_mapping"]:
            flags |= faiss.IO_FLAG_MMAP

        return flags

    def register_index_access(self, index) -> None:
        """
        Register an index access to track usage.

        Args:
            index: The index being accessed
        """
        import time

        index_id = id(index)
        now = time.time()

        with _registry_lock:
            self._active_indices.add(index_id)
            self._last_accessed[index_id] = now

            # Start background monitoring if auto-unload is enabled
            if (
                _memory_options["auto_unload_unused_indices"]
                and self._unload_thread is None
                and not self._running
            ):
                self._start_monitoring()

    def estimate_index_size(self, index) -> int:
        """
        Estimate the memory usage of an index in bytes.

        Args:
            index: The index to estimate size for

        Returns:
            Estimated size in bytes
        """
        index_id = id(index)

        # If we've calculated this before, return cached value
        if index_id in self._index_sizes:
            return self._index_sizes[index_id]

        # Basic estimation - this is very approximate
        size_bytes = 0

        # Account for dimension and vector count
        d = getattr(index, "d", 0)
        ntotal = getattr(index, "ntotal", 0)

        # Start with basic overhead
        size_bytes += 1024 * 100  # 100KB baseline overhead

        # Different index types have different memory profiles
        index_type = index.__class__.__name__

        if "Flat" in index_type:
            # Flat indices store full vectors
            size_bytes += ntotal * d * 4  # 4 bytes per float32

        elif "IVFPQ" in index_type:
            # IVF has centroids and PQ has codebooks
            nlist = getattr(index, "nlist", 100)
            m = getattr(index, "m", 8)
            nbits = getattr(index, "nbits", 8)

            # Centroid storage
            size_bytes += nlist * d * 4  # 4 bytes per float for centroids

            # PQ codebook storage
            size_bytes += (1 << nbits) * (d // m) * m * 4  # 4 bytes per float

            # Codes storage (compressed)
            code_size = (m * nbits + 7) // 8  # bytes per vector
            size_bytes += ntotal * code_size

        elif "IVF" in index_type:
            # IVF stores centroids and vector lists
            nlist = getattr(index, "nlist", 100)

            # Centroid storage
            size_bytes += nlist * d * 4  # 4 bytes per float for centroids

            # Vector storage depends on the type
            if "Flat" in index_type:
                size_bytes += ntotal * d * 4  # 4 bytes per float

        elif "HNSW" in index_type:
            # HNSW has graph structure overhead
            m = getattr(index, "m", 32)
            level_mult = 1 / math.log(m)  # According to HNSW paper

            # Vector storage
            size_bytes += ntotal * d * 4  # 4 bytes per float

            # Graph structure - this is an approximation
            avg_connections = m * (1 + level_mult)  # Include higher level links
            size_bytes += ntotal * avg_connections * 4  # 4 bytes per index

        elif "PQ" in index_type:
            # PQ uses codebooks and codes
            m = getattr(index, "m", 8)
            nbits = getattr(index, "nbits", 8)

            # Codebook storage
            size_bytes += (1 << nbits) * (d // m) * m * 4  # 4 bytes per float

            # Codes storage (compressed)
            code_size = (m * nbits + 7) // 8  # bytes per vector
            size_bytes += ntotal * code_size

        # Add ID mappings if applicable
        if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            size_bytes += ntotal * 8  # 8 bytes per ID mapping

        # Cache the result
        self._index_sizes[index_id] = size_bytes
        return size_bytes

    def unload_unused_indices(self, idle_seconds: int = 300) -> int:
        """
        Unload indices that haven't been used recently.

        Args:
            idle_seconds: Time in seconds since last access to consider an index idle

        Returns:
            Number of indices unloaded
        """
        import time

        now = time.time()
        unloaded = 0

        # Check for indices to unload
        with _registry_lock:
            for index_id in list(self._active_indices):
                last_access = self._last_accessed.get(index_id, 0)
                if now - last_access > idle_seconds:
                    # This index is idle - remove references to help with GC
                    self._active_indices.remove(index_id)
                    if index_id in self._last_accessed:
                        del self._last_accessed[index_id]
                    if index_id in self._index_sizes:
                        del self._index_sizes[index_id]
                    if index_id in _index_params:
                        del _index_params[index_id]

                    unloaded += 1

        # Suggest garbage collection if any indices were unloaded
        if unloaded > 0:
            import gc

            gc.collect()

        return unloaded

    def _start_monitoring(self) -> None:
        """Start background thread to monitor and unload unused indices."""
        import time
        import threading

        self._running = True

        def monitor_loop():
            try:
                while self._running and _memory_options["auto_unload_unused_indices"]:
                    # Sleep for a while
                    time.sleep(60)  # Check every minute

                    # Check memory usage and unload if needed
                    idle_time = 300  # 5 minutes of inactivity
                    self.unload_unused_indices(idle_seconds=idle_time)
            finally:
                self._running = False
                self._unload_thread = None

        self._unload_thread = threading.Thread(
            target=monitor_loop, name="FAISSx-MemoryMonitor", daemon=True
        )
        self._unload_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self._running = False

        if self._unload_thread and self._unload_thread.is_alive():
            self._unload_thread.join(timeout=1.0)
            self._unload_thread = None


# Initialize a global instance of the memory manager
memory_manager = MemoryManager()

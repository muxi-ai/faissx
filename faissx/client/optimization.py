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
FAISSx optimization controls module.

This module provides fine-grained parameter controls and memory management options
for FAISSx indices. It allows customizing performance characteristics and resource
usage for both local and remote modes.

Key features:
1. Fine-grained parameters: Control search quality, training, and other performance settings
   - Adjust search parameters like nprobe for IVF indices and efSearch for HNSW indices
   - Configure batch sizes for optimal performance
   - Set quality vs. speed tradeoffs for different index types

2. Memory management options: Configure memory usage, caching, and resource limits
   - Control memory consumption with configurable limits
   - Enable automatic unloading of idle indices
   - Use memory mapping for large indices
   - Configure vector caching for improved performance

Usage examples:
    # Adjust search parameters for an index
    index.set_parameter('nprobe', 20)  # Use more clusters for better recall in IVF indices
    index.set_parameter('batch_size', 5000)  # Process in larger batches for throughput

    # Configure memory management
    from faissx.client.optimization import memory_manager
    memory_manager.set_option('max_memory_usage_mb', 1024)  # Limit to 1GB
    memory_manager.set_option('auto_unload_unused_indices', True)  # Enable auto cleanup
"""

import gc
import logging
import math
import threading
import time
import weakref
from typing import Any, Dict, Optional, Set, TypeVar, Union

import faiss

# Set up module-level logger
logger = logging.getLogger(__name__)

# Type variable for the index instance - used in type hints
T = TypeVar('T')

# Type alias for FAISS index types
FaissIndex = Any  # Could be any FAISS index type

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

# Constants for memory estimation
BYTES_PER_FLOAT32 = 4  # Size of a float32 value in bytes
BYTES_PER_ID = 8  # Size of an ID value in bytes
BASELINE_OVERHEAD_KB = 100  # Base memory overhead per index in KB
MONITOR_INTERVAL_SECONDS = 60  # How often to check memory usage
DEFAULT_IDLE_SECONDS = 300  # Default time before unloading idle indices


class IndexParameters:
    """
    Manages fine-grained parameters for FAISS indices.

    This class provides a unified interface for setting and getting performance-related parameters
    for indices, regardless of their type. It handles proper parameter validation and application
    for both local and remote index modes.
    """

    # Parameter definitions with validation rules and descriptions
    PARAMETER_DEFINITIONS: Dict[str, Dict[str, Any]] = {
        # ----------------------------------------
        # SEARCH PERFORMANCE PARAMETERS
        # These parameters control how search operations behave
        # and represent the trade-off between speed and accuracy
        # ----------------------------------------

        # nprobe: Controls how many clusters are visited during search in IVF-based indices
        # Higher values = more accurate results but slower searches
        "nprobe": {
            "applicable_to": ["IVFFlat", "IVFPQ", "IVFScalarQuantizer"],
            "description": "Number of clusters to scan during search",
            "type": int,
            "min": 1,
            "max": 1024,
            "default": 1,
        },

        # efSearch: Controls search depth in HNSW indices
        # Higher values = more accurate results but slower searches
        "efSearch": {
            "applicable_to": ["HNSW"],
            "description": "Exploration factor for HNSW search",
            "type": int,
            "min": 1,
            "max": 1024,
            "default": 16,
        },

        # k_factor: Used for internal search multiplication factor
        # Helps with reranking by fetching more candidates than required
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

        # ----------------------------------------
        # TRAINING PARAMETERS
        # These parameters control how indices are trained
        # which affects index quality and performance
        # ----------------------------------------

        # n_iter: Number of iterations for IVF clustering
        # More iterations can produce better clustering but takes longer
        "n_iter": {
            "applicable_to": ["IVFFlat", "IVFPQ", "IVFScalarQuantizer"],
            "description": "Number of iterations when training the index",
            "type": int,
            "min": 1,
            "max": 1000,
            "default": 25,
        },

        # min_points_per_centroid: Controls cluster density in IVF indices
        # Prevents creation of nearly-empty clusters for better balance
        "min_points_per_centroid": {
            "applicable_to": ["IVFFlat", "IVFPQ", "IVFScalarQuantizer"],
            "description": "Minimum number of points per centroid during training",
            "type": int,
            "min": 5,
            "max": 1000,
            "default": 39,
        },

        # ----------------------------------------
        # INDEX CONSTRUCTION PARAMETERS
        # These parameters control how indices are built
        # which affects their structure and search capabilities
        # ----------------------------------------

        # efConstruction: Controls graph quality in HNSW indices
        # Higher values = better quality graph but slower construction
        "efConstruction": {
            "applicable_to": ["HNSW"],
            "description": "Construction time exploration factor for HNSW",
            "type": int,
            "min": 8,
            "max": 512,
            "default": 40,
        },

        # ----------------------------------------
        # OPERATION PARAMETERS
        # These parameters control how batch operations behave
        # affecting throughput and resource usage
        # ----------------------------------------

        # batch_size: Controls how many vectors are processed at once
        # Larger batches = better throughput but higher memory usage
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

        # ----------------------------------------
        # QUALITY PARAMETERS
        # These parameters control the quality vs. speed tradeoff
        # for specific index operations
        # ----------------------------------------

        # quantizer_effort: Controls encoding effort in scalar quantizers
        # Higher values = better quality encoding but slower processing
        "quantizer_effort": {
            "applicable_to": ["IVFScalarQuantizer"],
            "description": "Quantizer encoding effort (higher = better quality but slower)",
            "type": int,
            "min": 1,
            "max": 10,
            "default": 4,
        },
    }

    def __init__(self, index: FaissIndex) -> None:
        """
        Initialize parameter management for a FAISS index.

        Args:
            index: The FAISS index to manage parameters for
        """
        self._index_ref = weakref.ref(index)  # Use weakref to avoid circular references
        self._index_id = id(index)  # Unique identifier for the index
        self._index_type = self._get_index_type(index)  # Determine index type

        # Register the index in the global parameters registry if not already present
        with _registry_lock:
            if self._index_id not in _index_params:
                _index_params[self._index_id] = {}

    def _get_index_type(self, index: FaissIndex) -> str:
        """
        Determine the type of the index for parameter validation.

        Args:
            index: The FAISS index to check

        Returns:
            str: The index type as a string identifier
        """
        class_name = index.__class__.__name__

        # Map class names to simplified index type identifiers
        type_mapping = {
            "IndexFlatL2": "Flat",
            "IndexIVFFlat": "IVFFlat",
            "IndexIVFPQ": "IVFPQ",
            "IndexHNSW": "HNSW",
            "IndexPQ": "PQ",
            "IndexScalarQuantizer": "IVFScalarQuantizer"
        }

        # Look for a matching class name pattern
        for key, value in type_mapping.items():
            if key in class_name:
                return value

        # Default to Flat for unknown index types
        return "Flat"

    def set_parameter(self, name: str, value: Union[int, float, bool]) -> None:
        """
        Set a parameter value with validation.

        This method validates the parameter against defined constraints before
        applying it to the index.

        Args:
            name: Name of the parameter to set
            value: Value to set the parameter to

        Raises:
            ValueError: If parameter is invalid or not applicable to this index type
        """
        # Check that the parameter name is valid
        if name not in self.PARAMETER_DEFINITIONS:
            raise ValueError(f"Unknown parameter: {name}")

        param_def = self.PARAMETER_DEFINITIONS[name]

        # Check that the parameter applies to this index type
        if self._index_type not in param_def["applicable_to"]:
            raise ValueError(
                f"Parameter '{name}' not applicable to index type {self._index_type}"
            )

        # Validate parameter type
        expected_type = param_def["type"]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Param '{name}' expects {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

        # Validate parameter range
        if "min" in param_def and value < param_def["min"]:
            raise ValueError(f"Parameter '{name}' must be >= {param_def['min']}")

        if "max" in param_def and value > param_def["max"]:
            raise ValueError(f"Parameter '{name}' must be <= {param_def['max']}")

        # Store the parameter value in the global registry
        with _registry_lock:
            _index_params[self._index_id][name] = value

        # Apply the parameter to the actual index if it still exists
        index = self._index_ref()
        if index is not None:
            self._apply_parameter(index, name, value)

    def get_parameter(self, name: str) -> Union[int, float, bool]:
        """
        Get a parameter value.

        Returns the current value of the specified parameter, or its default
        value if not explicitly set.

        Args:
            name: Name of the parameter to get

        Returns:
            The parameter value

        Raises:
            ValueError: If parameter is invalid or not applicable to this index type
        """
        # Check that the parameter name is valid
        if name not in self.PARAMETER_DEFINITIONS:
            raise ValueError(f"Unknown parameter: {name}")

        param_def = self.PARAMETER_DEFINITIONS[name]

        # Check that the parameter applies to this index type
        if self._index_type not in param_def["applicable_to"]:
            raise ValueError(
                f"Parameter '{name}' not applicable to index type {self._index_type}"
            )

        # Get the parameter value from the registry or use the default
        with _registry_lock:
            if (
                self._index_id in _index_params
                and name in _index_params[self._index_id]
            ):
                return _index_params[self._index_id][name]
            return param_def["default"]

    def get_all_parameters(self) -> Dict[str, Union[int, float, bool]]:
        """
        Get all parameters applicable to this index.

        Returns:
            Dictionary of parameter names and their values, including defaults
            for parameters that haven't been explicitly set.
        """
        # Filter parameters by those applicable to this index type
        return {
            name: self.get_parameter(name)
            for name, param_def in self.PARAMETER_DEFINITIONS.items()
            if self._index_type in param_def["applicable_to"]
        }

    def reset_parameters(self) -> None:
        """
        Reset all parameters to their default values by clearing the index's parameter
        dictionary.
        """
        with _registry_lock:
            if self._index_id in _index_params:
                _index_params[self._index_id] = {}

    def _apply_parameter(
        self, index: FaissIndex, name: str, value: Union[int, float, bool]
    ) -> None:
        """
        Apply a parameter directly to the index if possible.

        This method handles special cases for common parameters like nprobe, efSearch,
        and efConstruction, which need direct application to the underlying index.

        Args:
            index: The FAISS index to modify
            name: Name of the parameter to set
            value: Value to set the parameter to
        """
        try:
            # Handle nprobe parameter for IVF indices
            if name == "nprobe" and hasattr(index, "_nprobe"):
                index._nprobe = value
                # Also update local index if it exists
                if hasattr(index, "_local_index") and index._local_index is not None:
                    index._local_index.nprobe = value

            # Handle efSearch parameter for HNSW indices
            elif name == "efSearch" and hasattr(index, "_ef"):
                index._ef = value
                # Also update local index if it exists
                if hasattr(index, "_local_index") and index._local_index is not None:
                    index._local_index.hnsw.efSearch = value

            # Handle efConstruction parameter for HNSW indices
            elif name == "efConstruction" and hasattr(index, "_efc"):
                index._efc = value

        except Exception as e:
            logger.warning(f"Failed to apply parameter {name}={value} to index: {e}")


class MemoryManager:
    """
    Manages memory usage for FAISS indices.

    This class tracks active indices, estimates their memory usage, and provides
    functionality to manage resource usage through automatic unloading of unused
    indices and other optimizations.

    Features:
    - Index usage tracking
    - Memory usage estimation
    - Automatic unloading of idle indices
    - Configurable memory management options
    """

    def __init__(self) -> None:
        """
        Initialize the memory manager with empty tracking collections and monitoring
        state.
        """
        self._active_indices: Set[int] = set()  # Set of active index IDs
        self._last_accessed: Dict[int, float] = {}  # Last access time for each index
        self._index_sizes: Dict[int, int] = {}  # Estimated size in bytes for each index
        self._unload_thread: Optional[threading.Thread] = None  # Background monitoring thread
        self._running: bool = False  # Monitoring thread state

    @staticmethod
    def set_option(name: str, value: Any) -> None:
        """
        Set a global memory management option with validation.

        This method configures how FAISSx manages memory for indices. The available options control
        various aspects of memory usage, caching behavior, and resource limits.

        Available options:
        - max_memory_usage_mb: Maximum memory usage in MB (None means no limit)
          Controls the total memory budget for all indices. When exceeded, least recently used
          indices may be unloaded.

        - use_memory_mapping: Whether to use memory mapping for large indices (boolean)
          When True, large indices are loaded using mmap, which can reduce memory pressure
          at the cost of potentially slower access.

        - index_cache_size: Maximum number of indices to keep in memory (integer)
          Controls how many indices can be active simultaneously before the least recently
          used ones are unloaded.

        - vector_cache_size_mb: Size of vector cache in MB (integer)
          Allocates memory for caching vectors to improve reconstruction performance.

        - auto_unload_unused_indices: Whether to automatically unload idle indices (boolean)
          When True, a background thread periodically checks for and unloads indices that
          haven't been accessed recently.

        - io_buffer_size_kb: Size of IO buffer in KB for index operations (integer)
          Controls the buffer size used when reading/writing indices to disk.

        Args:
            name: Name of the option to set
            value: Value to set the option to

        Raises:
            ValueError: If option name is invalid or value is invalid for the option

        Example:
            ```python
            # Limit total memory usage to 2GB
            MemoryManager.set_option('max_memory_usage_mb', 2048)

            # Enable automatic unloading of unused indices
            MemoryManager.set_option('auto_unload_unused_indices', True)
            ```
        """
        if name not in _memory_options:
            raise ValueError(f"Unknown memory option: {name}")

        # Validate option values
        if name == "max_memory_usage_mb" and value is not None and value <= 0:
            raise ValueError("max_memory_usage_mb must be positive or None")

        if name == "index_cache_size" and value <= 0:
            raise ValueError("index_cache_size must be positive")

        if name == "vector_cache_size_mb" and value <= 0:
            raise ValueError("vector_cache_size_mb must be positive")

        # Store validated option
        _memory_options[name] = value

    @staticmethod
    def get_option(name: str) -> Any:
        """
        Get the current value of a memory management option.

        Args:
            name: Name of the option to get

        Returns:
            Current value of the option

        Raises:
            ValueError: If option name is invalid
        """
        if name not in _memory_options:
            raise ValueError(f"Unknown memory option: {name}")

        return _memory_options[name]

    @staticmethod
    def get_all_options() -> Dict[str, Any]:
        """
        Get all current memory management options.

        Returns:
            Dictionary mapping option names to their current values
        """
        return dict(_memory_options)

    @staticmethod
    def reset_options() -> None:
        """
        Reset all memory management options to their default values.
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
        Get FAISS IO flags based on current memory options.

        Returns:
            Integer flags value for FAISS IO operations
        """
        flags = 0
        if _memory_options["use_memory_mapping"]:
            flags |= faiss.IO_FLAG_MMAP
        return flags

    def register_index_access(self, index: FaissIndex) -> None:
        """
        Register an index access to track its usage and potentially start monitoring.

        This method is called whenever an index is accessed to update its last access
        timestamp and ensure it's being tracked for memory management.

        Args:
            index: The index being accessed
        """
        index_id = id(index)
        now = time.time()

        with _registry_lock:
            # Update tracking information
            self._active_indices.add(index_id)
            self._last_accessed[index_id] = now

            # Start monitoring if auto-unload is enabled and not already running
            if (
                _memory_options["auto_unload_unused_indices"]
                and self._unload_thread is None
                and not self._running
            ):
                self._start_monitoring()

    def estimate_index_size(self, index: FaissIndex) -> int:
        """
        Estimate memory usage of an index based on its type and configuration.

        This method analyzes the index attributes and structure to determine
        its approximate memory footprint. The estimation includes:

        1. Base overhead for each index type
        2. Vector storage based on dimension and count
        3. Index-specific structures (centroids, codebooks, graphs)
        4. ID mapping overhead if applicable

        Memory estimation is essential for proper resource management, especially
        when working with large datasets or multiple indices.

        Args:
            index: The index to estimate size for

        Returns:
            Estimated size in bytes

        Note:
            This is an approximation and actual memory usage may vary, particularly for
            complex index types with dynamic allocations.
        """
        index_id = id(index)

        # Return cached size if available - avoids recalculating for frequent access
        if index_id in self._index_sizes:
            return self._index_sizes[index_id]

        # Initialize with baseline overhead - every index has management structures
        size_bytes = BASELINE_OVERHEAD_KB * 1024  # Convert KB to bytes

        # Extract common index properties
        d = getattr(index, "d", 0)  # Vector dimension
        ntotal = getattr(index, "ntotal", 0)  # Number of vectors

        # Calculate size based on index type
        index_type = index.__class__.__name__

        if "Flat" in index_type:
            # Flat indices store full vectors without compression
            # Memory usage is simple: dimension × vector_count × bytes_per_value
            size_bytes += ntotal * d * BYTES_PER_FLOAT32

        elif "IVFPQ" in index_type:
            # IVFPQ combines inverted file (IVF) with product quantization (PQ)
            nlist = getattr(index, "nlist", 100)  # Number of IVF clusters/cells
            m = getattr(index, "m", 8)  # Number of PQ subquantizers
            nbits = getattr(index, "nbits", 8)  # Bits per PQ code

            # Centroid storage - one centroid per IVF cell
            size_bytes += nlist * d * BYTES_PER_FLOAT32

            # PQ codebook storage - 2^nbits entries for each of the m subquantizers
            size_bytes += (1 << nbits) * (d // m) * m * BYTES_PER_FLOAT32

            # Compressed codes storage - each vector is stored as m codes of nbits bits
            code_size = (m * nbits + 7) // 8  # Rounded up to nearest byte
            size_bytes += ntotal * code_size  # Total space for all vector codes

        elif "IVF" in index_type:
            # IVF (Inverted File) stores centroids and vector lists
            nlist = getattr(index, "nlist", 100)  # Number of clusters

            # Centroids storage - one d-dimensional vector per cluster
            size_bytes += nlist * d * BYTES_PER_FLOAT32

            if "Flat" in index_type:
                # IVFFlat stores full uncompressed vectors within each cluster
                size_bytes += ntotal * d * BYTES_PER_FLOAT32

        elif "HNSW" in index_type:
            # HNSW (Hierarchical Navigable Small World) has complex graph structure
            m = getattr(index, "m", 32)  # Max number of connections per node
            level_mult = 1 / math.log(m)  # Level multiplier (from HNSW paper)

            # Vector storage - full vectors in original form
            size_bytes += ntotal * d * BYTES_PER_FLOAT32

            # Graph structure approximation - each node has connections
            # The number of connections varies by level, but we use an average here
            avg_connections = m * (1 + level_mult)  # Average across all levels
            size_bytes += ntotal * avg_connections * BYTES_PER_FLOAT32  # Pointers to neighbors

        elif "PQ" in index_type:
            # PQ (Product Quantization) compresses vectors using codebooks
            m = getattr(index, "m", 8)  # Number of subquantizers
            nbits = getattr(index, "nbits", 8)  # Bits per subquantizer code

            # Codebook storage - 2^nbits entries for each of the m subquantizers
            size_bytes += (1 << nbits) * (d // m) * m * BYTES_PER_FLOAT32

            # Compressed codes storage - each vector stored as m small codes
            code_size = (m * nbits + 7) // 8  # Round up to whole bytes
            size_bytes += ntotal * code_size  # Total space for all vector codes

        # Add ID mapping overhead if present - applies to IndexIDMap & IndexIDMap2
        if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            # Each vector gets an explicit ID mapping (typically 64-bit integers)
            size_bytes += ntotal * BYTES_PER_ID

        # Cache and return result
        self._index_sizes[index_id] = size_bytes
        return size_bytes

    def unload_unused_indices(self, idle_seconds: int = DEFAULT_IDLE_SECONDS) -> int:
        """
        Unload indices that haven't been accessed recently to free memory.

        This method implements a 'least recently used' (LRU) strategy to reclaim memory by
        removing references to indices that haven't been accessed for a specified period.
        When references are removed, the Python garbage collector can reclaim the memory
        if no other references to these indices exist elsewhere.

        The unloading process:
        1. Identifies indices that haven't been accessed in the specified time period
        2. Removes all internal references to those indices from tracking structures
        3. Triggers garbage collection to reclaim memory
        4. Returns the count of unloaded indices for monitoring

        This is particularly useful in long-running applications that create and use
        many indices over time, as it prevents memory usage from growing unbounded.

        Args:
            idle_seconds: Time in seconds since last access to consider an index idle
                         (default: 300 seconds / 5 minutes)

        Returns:
            Number of indices unloaded from memory

        Example:
            ```python
            # Unload indices not accessed in the last 10 minutes
            unloaded_count = memory_manager.unload_unused_indices(idle_seconds=600)
            print(f"Freed memory by unloading {unloaded_count} unused indices")
            ```
        """
        now = time.time()  # Current timestamp to compare against last access times
        unloaded = 0  # Counter for number of indices unloaded

        with _registry_lock:  # Thread-safe access to shared tracking structures
            # Iterate through a copy of the set to avoid modification during iteration
            # This prevents "collection modified during iteration" errors
            for index_id in list(self._active_indices):
                # Get the last access time or use 0 if not recorded (very old)
                last_access = self._last_accessed.get(index_id, 0)

                # Check if this index has been idle longer than the threshold
                if now - last_access > idle_seconds:
                    # Step 1: Remove the index from active set
                    self._active_indices.remove(index_id)

                    # Step 2: Clean up all references in tracking dictionaries
                    # Each cleanup step removes a reference that could prevent garbage collection
                    if index_id in self._last_accessed:
                        del self._last_accessed[index_id]
                    if index_id in self._index_sizes:
                        del self._index_sizes[index_id]
                    if index_id in _index_params:
                        del _index_params[index_id]

                    # Increment counter for reporting
                    unloaded += 1

        # Step 3: Trigger garbage collection if any indices were unloaded
        # This helps reclaim memory more quickly, especially for large indices
        if unloaded > 0:
            # Explicitly run the garbage collector to reclaim memory
            # Without this, memory might not be reclaimed until the next automatic GC cycle
            gc.collect()

        return unloaded

    def _start_monitoring(self) -> None:
        """
        Start a background thread to monitor and automatically unload unused indices.

        This method creates and starts a daemon thread that periodically checks for idle indices
        and unloads them to free memory. The monitoring process enables automatic memory management
        without requiring explicit calls to unload_unused_indices().

        Key features:
        - Uses a daemon thread that won't prevent application shutdown
        - Runs with low overhead (sleeps between monitoring cycles)
        - Self-terminates if auto_unload_unused_indices option is disabled
        - Ensures only one monitoring thread is active at a time
        - Cleans up resources properly when stopped

        The monitoring thread will:
        1. Sleep for MONITOR_INTERVAL_SECONDS (default: 60 seconds)
        2. Check if any indices have been idle for DEFAULT_IDLE_SECONDS (default: 300 seconds)
        3. Unload any idle indices to free memory
        4. Repeat until either the thread is signaled to stop or auto-unloading is disabled
        """
        # Set flag to indicate monitoring is active
        self._running = True

        def monitor_loop() -> None:
            """
            Main monitoring loop that periodically checks for and unloads idle indices.

            This function runs in a separate thread and implements the periodic monitoring logic.
            It balances memory usage optimization with minimal performance impact by:

            1. Sleeping most of the time to minimize CPU usage
            2. Checking configuration settings before taking action
            3. Performing memory cleanup operations only when needed
            4. Gracefully handling shutdown signals

            The loop continues until one of two conditions is met:
            - self._running is set to False (external stop signal)
            - auto_unload_unused_indices is disabled in memory options

            After exiting, it ensures thread state is properly cleaned up.
            """
            try:
                # Keep running while both conditions are true
                while self._running and _memory_options["auto_unload_unused_indices"]:
                    # Sleep for the monitoring interval
                    # This minimizes thread overhead between monitoring cycles
                    time.sleep(MONITOR_INTERVAL_SECONDS)

                    # Unload indices that haven't been accessed within the idle threshold
                    # This is the main memory reclamation mechanism
                    idle_count = self.unload_unused_indices(idle_seconds=DEFAULT_IDLE_SECONDS)

                    # Log the result if any indices were unloaded
                    if idle_count > 0:
                        logger.debug(f"Memory monitor unloaded {idle_count} idle indices")
            finally:
                # Clean up thread state when monitoring stops
                # This ensures resources are properly released regardless of how the loop exits
                self._running = False
                self._unload_thread = None

        # Create and start daemon thread for monitoring
        self._unload_thread = threading.Thread(
            target=monitor_loop, name="FAISSx-MemoryMonitor", daemon=True
        )
        self._unload_thread.start()

    def stop_monitoring(self) -> None:
        """
        Stop the background monitoring thread and clean up resources.

        This method provides a controlled way to stop the automatic index unloading process
        that was started by _start_monitoring(). It's important to call this method when
        auto-unloading is no longer needed to ensure proper resource cleanup.

        The shutdown process:
        1. Sets the _running flag to False, signaling the monitoring thread to exit
        2. Waits up to 1 second for the thread to terminate gracefully
        3. Clears the thread reference even if the join times out, allowing garbage collection

        This method is safe to call even if monitoring isn't active - it will simply
        return without taking any action if no monitoring thread exists.

        When to call:
        - Before application shutdown to ensure clean termination
        - When switching from automatic to manual memory management
        - When temporarily disabling monitoring during high-intensity operations

        Note:
            Even after stopping the monitor, you can still manually unload indices using
            the unload_unused_indices() method.
        """
        # Signal thread to stop
        self._running = False

        # Wait for thread to finish if it's still running
        if self._unload_thread and self._unload_thread.is_alive():
            self._unload_thread.join(timeout=1.0)
            self._unload_thread = None


# Initialize a global instance of the memory manager
memory_manager = MemoryManager()

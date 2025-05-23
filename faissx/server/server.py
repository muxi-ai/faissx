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
FAISSx Server - ZeroMQ Implementation

This module provides the core server implementation for the FAISSx vector search service.

It handles:
- ZeroMQ socket communication and message processing
- FAISS index management for vector operations
- Authentication and tenant isolation
- Binary protocol serialization/deserialization
- Request handling for create_index, add_vectors, search, and other operations

The server uses a REP socket pattern to provide synchronous request-response
communication and supports both in-memory and persistent storage of vector indices.
"""

import os
import time
import json
import zmq
import faiss
import logging
import numpy as np
import msgpack
import threading
import argparse
from queue import Queue

from .. import __version__ as faissx_version
from .core import DEFAULT_PORT
from .response import (
    error_response, success_response,
    format_search_results, format_vector_results,
    format_index_status
)
from .training import (
    is_trained_for_use, requires_training,
    get_training_requirements, estimate_training_vectors_needed
)
from .binary import (
    convert_to_binary, binary_to_float,
    is_binary_index_type, create_binary_index
)
from .logging import configure_logging
from . import specialized_operations
from . import transformations

# Server Configuration Constants
# These constants ensure consistent server behavior and easy configuration management

# Network and socket configuration
DEFAULT_BIND_ADDRESS = "0.0.0.0"  # Default bind address (all interfaces)
DEFAULT_SOCKET_TIMEOUT = 60000  # Socket timeout in milliseconds (60 seconds)
DEFAULT_HIGH_WATER_MARK = 1000  # High water mark for socket buffer
DEFAULT_LINGER = 1000  # Linger time for socket (1 second)
DEFAULT_OPERATION_TIMEOUT = 30  # Default timeout for operations in seconds

# Request processing constants
MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100MB maximum request size
MAX_RESPONSE_SIZE = 500 * 1024 * 1024  # 500MB maximum response size
MAX_CONCURRENT_REQUESTS = 100  # Maximum concurrent request handling

# Binary index class constants for type checking
BINARY_INDEX_CLASSES = [
    "IndexBinaryFlat", "IndexBinaryIVF", "IndexBinaryHash"
]

# Task worker configuration
DEFAULT_TASK_QUEUE_SIZE = 1000  # Maximum task queue size
DEFAULT_WORKER_THREAD_COUNT = 4  # Number of worker threads

# Authentication and security constants
AUTH_HEADER_FIELD = "api_key"
TENANT_HEADER_FIELD = "tenant_id"
MAX_API_KEY_LENGTH = 256
MAX_TENANT_ID_LENGTH = 128

# Index operation constants
MAX_VECTORS_PER_BATCH = 10000  # Maximum vectors per batch operation
MAX_SEARCH_K = 10000  # Maximum k value for search operations
MAX_DIMENSION = 10000  # Maximum vector dimension
MIN_DIMENSION = 1  # Minimum vector dimension

# Response status constants
RESPONSE_STATUS_SUCCESS = "success"
RESPONSE_STATUS_ERROR = "error"
RESPONSE_STATUS_TIMEOUT = "timeout"

# Operation timeout constants by operation type
OPERATION_TIMEOUTS = {
    "search": 30,
    "add_vectors": 60,
    "train_index": 300,
    "create_index": 60,
    "merge_indices": 180,
    "range_search": 45,
    "get_vectors": 120,
    "search_and_reconstruct": 60,
    "apply_transform": 30,
}

# Configure logging
logger = logging.getLogger("faissx.server")
# Don't set a default log level here, it will be set in run_server
# logger.setLevel(logging.DEBUG)


class RequestTimeoutError(Exception):
    """
    Exception raised when a request takes too long to process.

    This custom exception is used throughout the server to handle timeout scenarios
    where operations exceed their allocated processing time. It provides a clear
    indication that the failure was due to timeout rather than other errors.

    Usage:
        Used by TaskWorker and timeout-sensitive operations to signal when
        processing time limits are exceeded, allowing for graceful error handling
        and appropriate client responses.
    """
    pass


class TaskWorker:
    """
    Worker to handle long-running tasks asynchronously.

    This class provides a background task execution system that prevents long-running
    operations from blocking the main ZeroMQ request-response loop. It manages a
    separate worker thread and provides timeout capabilities for task execution.

    Architecture:
        - Uses a separate daemon thread for task processing
        - Maintains a task queue for pending operations
        - Stores results in memory with task ID mapping
        - Provides both fire-and-forget and wait-for-result patterns

    Attributes:
        timeout (int): Default timeout in seconds for task completion
        queue (Queue): Thread-safe queue for pending tasks
        results (dict): In-memory storage for completed task results
        worker_thread (Thread): Background thread for task processing

    Thread Safety:
        All operations are thread-safe using Queue and dict operations.
        Results are cleaned up after retrieval to prevent memory leaks.
    """

    def __init__(self, timeout=DEFAULT_OPERATION_TIMEOUT):
        """
        Initialize a worker for handling long-running tasks.

        Args:
            timeout (int): Default timeout in seconds for task completion.
                          Individual tasks can override this value.

        Implementation Notes:
            - Creates a daemon thread that will automatically terminate when
              the main process exits
            - Uses Queue for thread-safe communication between threads
            - Results dictionary is accessed by main thread only after completion
        """
        self.timeout = timeout
        self.queue = Queue()  # Thread-safe task queue
        self.results = {}     # Task results storage (task_id -> result)

        # Create and start the background worker thread
        # Daemon=True ensures thread dies when main process exits
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """
        Background worker loop to process tasks.

        This method runs in a separate thread and continuously processes tasks
        from the queue. It handles task execution, result storage, and error
        capture to ensure the main thread can retrieve results safely.

        Implementation Details:
            - Runs in infinite loop until process termination
            - Uses blocking queue.get() to wait for new tasks
            - Captures all exceptions to prevent thread termination
            - Stores both successful results and error information
            - Calls task_done() to signal queue completion
        """
        while True:
            try:
                # Block until a task is available in the queue
                # Returns tuple: (task_id, function, args, kwargs)
                task_id, func, args, kwargs = self.queue.get(block=True)

                try:
                    # Execute the task function with provided arguments
                    result = func(*args, **kwargs)

                    # Store successful result for main thread retrieval
                    self.results[task_id] = {"success": True, "result": result}
                except Exception as e:
                    # Capture any task execution errors
                    logger.error(f"Task {task_id} failed: {str(e)}")
                    self.results[task_id] = {"success": False, "error": str(e)}
                finally:
                    # Signal that this task is complete (required for Queue)
                    self.queue.task_done()
            except Exception as e:
                # Handle any unexpected worker thread errors
                # This should rarely happen, but prevents thread death
                logger.error(f"Worker error: {str(e)}")

    def submit_task(self, func, *args, **kwargs):
        """
        Submit a task to be executed in the background.

        Args:
            func: Function to execute in the background thread
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            str: Unique task ID that can be used to retrieve results later

        Implementation Notes:
            - Uses current timestamp as task ID for simplicity
            - Tasks are added to thread-safe queue for worker processing
            - Non-blocking operation - returns immediately
        """
        # Generate unique task ID using current timestamp
        # Simple but effective for this use case
        task_id = str(time.time())

        # Add task to queue - worker thread will pick it up
        self.queue.put((task_id, func, args, kwargs))

        return task_id

    def get_result(self, task_id):
        """
        Get the result of a task without waiting.

        Args:
            task_id (str): ID of the task to check

        Returns:
            dict: Result dictionary with 'success' and 'result'/'error' keys,
                  or None if task is not yet complete

        Usage:
            Non-blocking check for task completion. Use this when you want to
            poll for results without blocking the calling thread.
        """
        # Simple dictionary lookup - returns None if not found
        return self.results.get(task_id)

    def wait_for_result(self, task_id, timeout=None):
        """
        Wait for a task result with timeout.

        Args:
            task_id (str): ID of the task to wait for
            timeout (int, optional): Timeout in seconds. Uses instance default
                                    if not provided.

        Returns:
            dict: Result dictionary with 'success' and 'result'/'error' keys

        Raises:
            RequestTimeoutError: If the task doesn't complete within the timeout

        Implementation Strategy:
            - Uses polling with short sleep intervals rather than blocking wait
            - Cleans up results after retrieval to prevent memory leaks
            - Provides precise timeout control for different operation types
        """
        # Use instance timeout if none provided
        timeout = timeout or self.timeout
        start_time = time.time()

        # Poll for results with timeout
        while time.time() - start_time < timeout:
            if task_id in self.results:
                # Task completed - retrieve and clean up result
                result = self.results[task_id]
                del self.results[task_id]  # Prevent memory leaks
                return result

            # Short sleep to prevent busy waiting
            time.sleep(0.1)

        # Timeout reached without result completion
        raise RequestTimeoutError(
            "Task timed out after {} seconds".format(timeout)
        )


class FaissIndex:
    """
    FAISS index server implementation providing vector database operations.
    """

    def __init__(self, data_dir=None):
        """
        Initialize the FAISS server.

        Args:
            data_dir (str, optional): Directory to persist indices (not implemented yet)
        """
        self.indexes = {}
        self.dimensions = {}
        self.data_dir = data_dir
        self.base_indexes = {}  # Add initialization for base_indexes
        self.task_worker = TaskWorker()
        # logger.info("FAISSx server initialized (version %s)", faissx_version)

    def _run_with_timeout(self, func, *args, timeout=None, **kwargs):
        """
        Run a function with a timeout using the background task worker.

        This method provides a consistent timeout mechanism for long-running index
        operations. It prevents operations from blocking the main ZeroMQ loop and
        provides graceful timeout handling with appropriate error responses.

        Args:
            func: Function to execute with timeout protection
            *args: Positional arguments to pass to the function
            timeout: Timeout in seconds (uses operation-specific defaults if None)
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function execution or error response dict

        Implementation Strategy:
            - Delegates execution to TaskWorker for background processing
            - Provides consistent error response format for timeouts
            - Captures and logs all execution errors for debugging
            - Returns structured responses compatible with client expectations
        """
        try:
            # Submit task to background worker for timeout-controlled execution
            task_id = self.task_worker.submit_task(func, *args, **kwargs)

            # Wait for completion with timeout protection
            result = self.task_worker.wait_for_result(task_id, timeout)

            # Check if the background task succeeded
            if not result["success"]:
                # Task failed - return structured error response
                return error_response(result["error"])

            # Task succeeded - return the actual result
            return result["result"]
        except RequestTimeoutError as e:
            # Handle timeout specifically with appropriate logging and response
            logger.error(f"Timeout error: {str(e)}")
            return error_response("Operation timed out", code="TIMEOUT")
        except Exception as e:
            # Handle any unexpected errors during task execution
            logger.error(f"Unexpected error: {str(e)}")
            return error_response(f"Unexpected error: {str(e)}")

    def search(self, index_id, query_vectors, k=10, params=None):
        """
        Search for similar vectors in an index with timeout protection.

        This is the main entry point for vector similarity search operations.
        It provides timeout protection by delegating to the background task worker
        while maintaining the same interface as the internal search implementation.

        Args:
            index_id (str): ID of the target index to search in
            query_vectors (list): List of query vectors for similarity search
            k (int): Number of nearest neighbors to return per query (default: 10)
            params (dict, optional): Additional search parameters like nprobe for
                                    IVF indices or efSearch for HNSW indices

        Returns:
            dict: Response containing search results with the following structure:
                - success (bool): Whether the operation succeeded
                - results (list): List of result objects, one per query vector
                - num_queries (int): Number of query vectors processed
                - k (int): Number of neighbors requested per query
                - message (str, optional): Success/error message
                - error (str, optional): Error description if success is False
                - code (str, optional): Error code for programmatic handling

        Timeout Handling:
            Uses operation-specific timeout from OPERATION_TIMEOUTS configuration.
            Long-running searches will be terminated gracefully with timeout error.
        """
        return self._run_with_timeout(
            self._search, index_id, query_vectors, k, params
        )

    def _search(self, index_id, query_vectors, k=10, params=None):
        """
        Internal implementation of search that can be run with timeout.

        This method contains the core vector similarity search logic and supports
        both standard float vectors and binary vectors. It handles index validation,
        query preprocessing, parameter application, and result formatting.

        Args:
            index_id (str): ID of the target index
            query_vectors (list): List of query vectors
            k (int): Number of nearest neighbors to return
            params (dict, optional): Search parameters (nprobe, efSearch, etc.)

        Returns:
            dict: Formatted search results or error response

        Implementation Details:
            - Validates index existence and readiness for search
            - Detects binary vs float index types automatically
            - Applies runtime search parameters (nprobe, efSearch)
            - Handles dimension validation for query vectors
            - Converts results to JSON-serializable format
            - Provides detailed error messages for troubleshooting
        """
        # Validate that the target index exists
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} does not exist")

        try:
            index = self.indexes[index_id]

            # Check if this is a binary index by examining the class name
            # Binary indices require special handling for vector conversion
            is_binary = False
            for binary_class in [
                faiss.IndexBinaryFlat,
                faiss.IndexBinaryIVF,
                faiss.IndexBinaryHash,
            ]:
                if isinstance(index, binary_class):
                    is_binary = True
                    break

            # Verify that the index is properly trained and ready for search
            # Some index types (like IVF) require training before use
            is_ready, reason = is_trained_for_use(index)
            if not is_ready:
                return error_response(
                    f"Index is not ready: {reason}",
                    code="NOT_TRAINED"
                )

            if is_binary:
                # Handle binary index search path
                try:
                    # Convert float query vectors to binary format
                    # This involves quantizing to binary representation
                    query_binary = convert_to_binary(query_vectors)

                    # Log debug information for binary search operations
                    logger.debug(
                        f"Binary query shape: {query_binary.shape}, "
                        f"dtype: {query_binary.dtype}"
                    )

                    # Perform the binary similarity search
                    # Uses Hamming distance for binary vectors
                    distances, indices = index.search(query_binary, k)

                    # Convert numpy arrays to lists for JSON serialization
                    # Each query gets its own result set
                    results = []
                    for i in range(len(distances)):
                        results.append({
                            "distances": distances[i].tolist(),
                            "indices": indices[i].tolist()
                        })

                    return success_response(
                        {
                            "results": results,
                            "num_queries": len(query_vectors),
                            "k": k,
                            "is_binary": True  # Flag to indicate binary search
                        }
                    )
                except Exception as e:
                    # Handle any binary-specific processing errors
                    logger.error(f"Error in binary search: {str(e)}")
                    return error_response(f"Error in binary search: {str(e)}")
            else:
                # Handle standard float vector search path

                # Convert query vectors to numpy array with proper data type
                query_np = np.array(query_vectors, dtype=np.float32)

                # Validate that query dimensions match index dimensions
                if query_np.shape[1] != self.dimensions[index_id]:
                    return error_response(
                        f"Query dimension mismatch: expected {self.dimensions[index_id]}, "
                        f"got {query_np.shape[1]}"
                    )

                # Apply runtime search parameters if provided
                # These can optimize search quality vs speed tradeoffs
                if params:
                    for param_name, param_value in params.items():
                        # Set nprobe for IVF indices (number of clusters to search)
                        if hasattr(index, "set_" + param_name):
                            getattr(index, "set_" + param_name)(param_value)

                # Note: IndexPreTransform automatically applies transformations
                # during search, so no special handling is needed here

                # Perform the similarity search on float vectors
                # Uses L2 (Euclidean) or IP (inner product) distance as configured
                distances, indices = index.search(query_np, k)

                # Convert numpy arrays to lists for JSON serialization
                # Structure results as list of per-query result objects
                results = []
                for i in range(len(distances)):
                    results.append({
                        "distances": distances[i].tolist(),
                        "indices": indices[i].tolist()
                    })

                return success_response(
                    {
                        "results": results,
                        "num_queries": len(query_vectors),
                        "k": k
                    }
                )
        except Exception as e:
            # Handle any unexpected errors during search execution
            return error_response(f"Error searching index: {str(e)}")

    def get_vectors(self, index_id, start_idx=0, limit=None):
        """
        Retrieve all vectors in an index or a specified range of vectors.

        This method allows efficient retrieval of vectors stored in an index,
        with optional pagination using start_idx and limit parameters.

        Args:
            index_id (str): ID of the index
            start_idx (int, optional): Starting index for retrieval (default: 0)
            limit (int, optional): Maximum number of vectors to retrieve
                (default: None, retrieve all)

        Returns:
            dict: Response containing the vectors or error message
        """
        return self._run_with_timeout(self._get_vectors, index_id, start_idx, limit)

    def _get_vectors(self, index_id, start_idx=0, limit=None):
        """Internal implementation of get_vectors that can be run with timeout."""
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} not found")

        try:
            index = self.indexes[index_id]
            ntotal = index.ntotal

            if ntotal == 0:
                return success_response({"vectors": [], "ntotal": 0})

            # Validate parameters
            if start_idx < 0 or start_idx >= ntotal:
                return error_response(
                    f"Invalid start_idx: {start_idx}, should be between 0 and {ntotal-1}"
                )

            # If limit is None, retrieve all vectors from start_idx
            if limit is None:
                limit = ntotal - start_idx
            else:
                # Ensure we don't go beyond the total number of vectors
                limit = min(limit, ntotal - start_idx)

            # For indices with reconstruct method
            if hasattr(index, "reconstruct"):
                vectors = []
                for i in range(start_idx, start_idx + limit):
                    vectors.append(index.reconstruct(i).tolist())

                return success_response(
                    format_vector_results(vectors, start_idx, ntotal)
                )
            # For indices with reconstruct_n method
            elif hasattr(index, "reconstruct_n"):
                vectors = index.reconstruct_n(start_idx, limit)
                return success_response(
                    format_vector_results(vectors.tolist(), start_idx, ntotal)
                )
            else:
                return error_response(
                    f"Index type {type(index).__name__} does not support vector retrieval"
                )

        except Exception as e:
            logger.exception(f"Error retrieving vectors: {e}")
            return error_response(f"Error retrieving vectors: {str(e)}")

    def search_and_reconstruct(self, index_id, query_vectors, k=10, params=None):
        """
        Search for similar vectors and return both distances/indices and the reconstructed vectors.

        This method combines search and vector reconstruction in a single operation,
        which can be more efficient than separate calls, especially for remote operation.

        Args:
            index_id (str): ID of the target index
            query_vectors (list): List of query vectors
            k (int): Number of nearest neighbors to return
            params (dict, optional): Additional search parameters like nprobe for IVF indices

        Returns:
            dict: Response containing search results with reconstructed vectors or error message
        """
        return self._run_with_timeout(
            self._search_and_reconstruct, index_id, query_vectors, k, params
        )

    def _search_and_reconstruct(self, index_id, query_vectors, k=10, params=None):
        """Internal implementation of search_and_reconstruct that can be run with timeout."""
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} does not exist")

        try:
            # Convert query vectors to numpy array and validate dimensions
            query_np = np.array(query_vectors, dtype=np.float32)
            if query_np.shape[1] != self.dimensions[index_id]:
                return error_response(
                    f"Query dimension mismatch: expected {self.dimensions[index_id]}, "
                    f"got {query_np.shape[1]}"
                )

            index = self.indexes[index_id]

            # Check if the index is ready for use
            is_ready, reason = is_trained_for_use(index)
            if not is_ready:
                return error_response(
                    f"Index is not ready: {reason}",
                    code="NOT_TRAINED"
                )

            # Apply runtime parameters if provided
            if params:
                self._apply_search_params(index, params)

            # Check if index supports search_and_reconstruct directly
            if hasattr(index, "search_and_reconstruct"):
                distances, indices, vectors = index.search_and_reconstruct(query_np, k)
                vectors = vectors.tolist()
            else:
                # Fallback: do search, then reconstruct each result
                distances, indices = index.search(query_np, k)
                vectors = []

                # For each query result set
                for query_idx, idx_array in enumerate(indices):
                    query_vectors = []
                    # For each result in the set
                    for result_idx in idx_array:
                        if result_idx != -1:  # -1 indicates no result found
                            vector = index.reconstruct(int(result_idx)).tolist()
                            query_vectors.append(vector)
                        else:
                            # Add a placeholder for missing results
                            query_vectors.append([0.0] * self.dimensions[index_id])
                    vectors.append(query_vectors)

            # Format the results
            results = format_search_results(distances, indices, vectors)

            return success_response(
                {
                    "results": results,
                    "num_queries": len(query_vectors),
                    "k": k
                }
            )
        except Exception as e:
            logger.exception(f"Error in search_and_reconstruct: {e}")
            return error_response(f"Error in search_and_reconstruct: {str(e)}")

    def add_vectors(self, index_id, vectors, ids=None):
        """
        Add vectors to an index.

        Args:
            index_id (str): ID of the target index
            vectors (list): List of vectors to add
            ids (list, optional): List of IDs for the vectors

        Returns:
            dict: Response containing success status or error message
        """
        return self._run_with_timeout(self._add_vectors, index_id, vectors, ids)

    def _add_vectors(self, index_id, vectors, ids=None):
        """Internal implementation of add_vectors that can be run with timeout."""
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} does not exist")

        try:
            index = self.indexes[index_id]

            # Check if the index is ready for use
            is_ready, reason = is_trained_for_use(index)
            if not is_ready:
                return error_response(
                    f"Index is not ready: {reason}",
                    code="NOT_TRAINED"
                )

            # Check if this is a binary index
            is_binary = False
            for binary_class in [
                faiss.IndexBinaryFlat,
                faiss.IndexBinaryIVF,
                faiss.IndexBinaryHash,
            ]:
                if isinstance(index, binary_class):
                    is_binary = True
                    break

            if is_binary:
                # Convert vectors to binary format
                try:
                    binary_vectors = convert_to_binary(vectors)

                    # Debug information for binary operations
                    logger.debug(
                        f"Binary vectors shape: {binary_vectors.shape}, "
                        f"dtype: {binary_vectors.dtype}"
                    )

                    # Add vectors to the index (with or without IDs)
                    if ids is not None:
                        if isinstance(index, faiss.IndexBinaryIDMap):
                            # For IDMap binary indices
                            ids_np = np.array(ids, dtype=np.int64)
                            index.add_with_ids(binary_vectors, ids_np)
                        else:
                            # For standard binary indices, IDs are ignored
                            index.add(binary_vectors)
                    else:
                        index.add(binary_vectors)
                except Exception as e:
                    logger.error(f"Error in binary vector conversion: {str(e)}")
                    return error_response(f"Error in binary vector conversion: {str(e)}")
            else:
                # Convert vectors to numpy array
                vectors_np = np.array(vectors, dtype=np.float32)

                # Verify dimensions
                if vectors_np.shape[1] != self.dimensions[index_id]:
                    return error_response(
                        f"Vector dimension mismatch: expected {self.dimensions[index_id]}, "
                        f"got {vectors_np.shape[1]}"
                    )

                # Add vectors to the index (with or without IDs)
                if ids is not None:
                    if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                        ids_np = np.array(ids, dtype=np.int64)
                        index.add_with_ids(vectors_np, ids_np)
                    else:
                        # For non-IDMap indices, warn that IDs are ignored
                        index.add(vectors_np)
                else:
                    index.add(vectors_np)

            return success_response(
                {
                    "ntotal": index.ntotal,
                    "total": index.ntotal,
                    "count": len(vectors)
                },
                message=f"Added {len(vectors)} vectors to index {index_id}"
            )
        except Exception as e:
            return error_response(f"Error adding vectors: {str(e)}")

    def _apply_search_params(self, index, params):
        """Apply runtime parameters to an index."""
        for param, value in params.items():
            if param == "nprobe" and hasattr(index, "nprobe"):
                index.nprobe = int(value)
            elif param == "efSearch" and hasattr(index, "hnsw"):
                index.hnsw.efSearch = int(value)

    def create_index(self, index_id, dimension, index_type="L2", metadata=None):
        """
        Create a new FAISS index with specified parameters.

        Args:
            index_id (str): Unique identifier for the index
            dimension (int): Dimension of vectors to be stored
            index_type (str): Type of index:
                - "L2" - Flat L2 index (Euclidean distance)
                - "IP" - Flat IP index (inner product)
                - "IVF" - IVF index with L2 distance
                - "IVF_IP" - IVF index with inner product distance
                - "HNSW" - HNSW index with L2 distance
                - "HNSW_IP" - HNSW index with inner product distance
                - "PQ" - Product Quantization index with L2 distance
                - "PQ_IP" - Product Quantization index with inner product distance
                - "IDMap:{base_type}" - IDMap index with specified base type
                - "BINARY_FLAT" - Binary flat index (Hamming distance)
                - "BINARY_IVF{nlist}" - Binary IVF index with {nlist} clusters
                - "BINARY_HASH{bits}" - Binary hash index with {bits} bits per dimension
                - "PCA{dim},{base_type}" - PCA transformation followed by base index
                - "NORM,{base_type}" - L2 Normalization followed by base index
                - "OPQ{M}_{dim},{base_type}" - OPQ transformation followed by base index
            metadata (dict, optional): Additional metadata for the index

        Returns:
            dict: Response containing success status and index details
        """
        logger.info(f"Creating index: id={index_id}, type={index_type}, dimension={dimension}")
        logger.debug("DEBUG: create_index called - will return proper response structure")

        # Ensure dimension is an integer
        try:
            dimension = int(dimension)
        except (ValueError, TypeError):
            return error_response(f"Invalid dimension: {dimension}. Must be an integer.")

        # Ensure index_type is a string
        if not isinstance(index_type, str):
            return error_response(f"Invalid index_type: {index_type}. Must be a string.")

        if index_id in self.indexes:
            return error_response(f"Index {index_id} already exists")

        try:
            # Handle IDMap creation
            if index_type.startswith("IDMap:") or index_type.startswith("IDMap2:"):
                logger.info(f"Creating IDMap index: {index_type}")
                is_idmap2 = index_type.startswith("IDMap2:")
                prefix_len = 7 if is_idmap2 else 6

                # Extract base type
                base_type = index_type[prefix_len:]
                logger.info(f"Base type for IDMap: {base_type}")

                # Check if base index exists directly
                if base_type in self.indexes:
                    logger.info(f"Found existing base index: {base_type}")
                    base_index = self.indexes[base_type]
                    base_index_id = base_type
                else:
                    # Check if base index is already created (might be used for multiple IDMaps)
                    base_index_id = f"{index_id}_base"

                    if base_index_id not in self.indexes:
                        # Create base index
                        logger.info(f"Creating base index: id={base_index_id}, type=L2")
                        base_response = self.create_index(
                            base_index_id,
                            dimension,
                            "L2",  # Default to L2 index
                            metadata={"is_base_index": True, "parent_index": index_id}
                        )

                        # Ensure we check response properly
                        if not isinstance(base_response, dict
                                          ) or not base_response.get("success", False):
                            logger.error(f"Failed to create base index: {base_response}")
                            return error_response(
                                f"Failed to create base index: {base_response}"
                            )

                    base_index = self.indexes[base_index_id]

                # Store base index relationship
                self.base_indexes[index_id] = base_index_id

                # Create IDMap wrapper
                if is_idmap2:
                    logger.info(f"Creating IDMap2 wrapper for {base_index_id}")
                    index = faiss.IndexIDMap2(base_index)
                else:
                    logger.info(f"Creating IDMap wrapper for {base_index_id}")
                    index = faiss.IndexIDMap(base_index)

                self.indexes[index_id] = index
                self.dimensions[index_id] = dimension

                # Prepare response
                index_details = {
                    "index_id": index_id,
                    "dimension": dimension,
                    "type": index_type,
                    "base_index_id": base_index_id,
                    "is_trained": getattr(index, "is_trained", True),
                    "is_idmap": True,
                    "is_idmap2": is_idmap2
                }

                if metadata:
                    index_details["metadata"] = metadata

                logger.debug("DEBUG: IDMap creation returning success_response")
                return success_response(
                    index_details,
                    message=f"IDMap index {index_id} created successfully",
                )

            # Check for binary index types
            if is_binary_index_type(index_type):
                try:
                    # Create binary index using the binary module
                    index, index_info = create_binary_index(index_type, dimension)

                    # Store the index and dimension
                    self.indexes[index_id] = index
                    self.dimensions[index_id] = dimension

                    # Add metadata if provided
                    if metadata:
                        index_info["metadata"] = metadata

                    logger.debug("DEBUG: Binary index creation returning success_response")
                    return success_response(
                        index_info,
                        message=f"Binary index {index_id} created successfully",
                    )
                except Exception as e:
                    return error_response(f"Error creating binary index: {str(e)}")

            # Use the transformations module to create the index
            try:
                logger.info(f"Using transformations module to create index: {index_type}")
                index, index_info = transformations.create_index_from_type(
                    index_type, dimension, "L2", metadata
                )

                # Store the index and dimension
                self.indexes[index_id] = index
                self.dimensions[index_id] = dimension

                # Add index_id to the info
                index_info["index_id"] = index_id

                logger.debug("DEBUG: Transformations creation returning success_response")
                return success_response(
                    index_info,
                    message=f"Index {index_id} created successfully"
                )
            except Exception as e:
                # If the new method fails, fall back to the original method
                logger.warning(f"Error using new index creation: {str(e)}")

            # Handle standard FAISS index types (fallback method)
            logger.info(f"Using fallback method to create index: {index_type}")
            if index_type == "L2":
                index = faiss.IndexFlatL2(dimension)
            elif index_type == "IP":
                index = faiss.IndexFlatIP(dimension)
            elif index_type == "IVF":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 centroids by default
            elif index_type == "IVF_IP":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
            elif index_type == "HNSW":
                index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors by default
            elif index_type == "HNSW_IP":
                index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            elif index_type == "PQ":
                index = faiss.IndexPQ(dimension, 8, 8)  # 8 subquantizers x 8 bits each by default
            elif index_type == "PQ_IP":
                index = faiss.IndexPQ(dimension, 8, 8, faiss.METRIC_INNER_PRODUCT)
            elif index_type.startswith("IDMap:"):
                base_type = index_type[6:]
                base_index = self.create_index(f"{index_id}_base", dimension, base_type)
                # BUG FIX: Ensure we check response properly
                if (not isinstance(base_index, dict) or
                        not base_index.get("success", False)):
                    return error_response(
                        f"Failed to create base index: {base_index}"
                    )

                self.base_indexes[index_id] = f"{index_id}_base"
                index = faiss.IndexIDMap(self.indexes[f"{index_id}_base"])
            else:
                return error_response(f"Unsupported index type: {index_type}")

            self.indexes[index_id] = index
            self.dimensions[index_id] = dimension

            # Prepare response with index details
            index_details = {
                "index_id": index_id,
                "dimension": dimension,
                "type": index_type,
                "is_trained": getattr(index, "is_trained", True)
            }

            # Add metadata if provided
            if metadata:
                index_details["metadata"] = metadata

            # Add training requirements if needed
            if requires_training(index):
                training_reqs = get_training_requirements(index)
                index_details["requires_training"] = True
                index_details["training_info"] = training_reqs

            logger.debug("DEBUG: Fallback creation returning success_response")
            return success_response(index_details, message=f"Index {index_id} created successfully")

        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return error_response(f"Error creating index: {str(e)}")

    def add_with_ids(self, index_id, vectors, ids):
        """
        Add vectors with explicit IDs to an index.

        Args:
            index_id (str): ID of the target index
            vectors (list): List of vectors to add
            ids (list): List of IDs to associate with vectors

        Returns:
            dict: Response containing success status and count of added vectors or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            # Check if index is an IDMap type
            index = self.indexes[index_id]
            if not isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                return {
                    "success": False,
                    "error": f"Index {index_id} is not an IDMap type"
                }

            # Convert vectors and IDs to numpy arrays
            vectors_np = np.array(vectors, dtype=np.float32)
            ids_np = np.array(ids, dtype=np.int64)

            # Validate dimensions
            if vectors_np.shape[1] != self.dimensions[index_id]:
                return {
                    "success": False,
                    "error": (
                        f"Vector dimension mismatch. Expected {self.dimensions[index_id]}, "
                        f"got {vectors_np.shape[1]}"
                    )
                }

            # Validate matching lengths
            if len(vectors_np) != len(ids_np):
                error_msg = (
                    f"Number of vectors ({len(vectors_np)}) doesn't match "
                    f"number of IDs ({len(ids_np)})"
                )
                return {
                    "success": False,
                    "error": error_msg
                }

            # Initialize vector cache if needed
            if not hasattr(self, '_vector_cache'):
                self._vector_cache = {}
            if index_id not in self._vector_cache:
                self._vector_cache[index_id] = {}

            # Cache each vector with its ID as list to ensure serializability
            for i, id_val in enumerate(ids_np):
                id_int = int(id_val)
                self._vector_cache[index_id][id_int] = vectors_np[i].tolist()

            # Add vectors with IDs
            index.add_with_ids(vectors_np, ids_np)
            total = index.ntotal

            debug_msg = f"Added {len(vectors)} vectors w/ IDs to index {index_id}, total: {total}"
            logger.debug(debug_msg)
            return {"success": True, "count": len(vectors), "total": total}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def remove_ids(self, index_id, ids):
        """
        Remove vectors with the specified IDs from an index.

        Args:
            index_id (str): ID of the target index
            ids (list): List of IDs to remove

        Returns:
            dict: Response containing success status and count of removed vectors or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            # Check if index is an IDMap type
            index = self.indexes[index_id]
            if not isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                return {
                    "success": False,
                    "error": f"Index {index_id} is not an IDMap type"
                }

            # Convert IDs to numpy array
            ids_np = np.array(ids, dtype=np.int64)

            # Get current vector count
            before_count = index.ntotal

            # Remove IDs
            index.remove_ids(ids_np)

            # Calculate number of vectors removed
            after_count = index.ntotal
            removed_count = before_count - after_count

            debug_msg = (
                f"Removed {removed_count} vectors from index {index_id}, "
                f"remaining: {after_count}"
            )
            logger.debug(debug_msg)
            return {"success": True, "count": removed_count, "total": after_count}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def reconstruct(self, index_id, id_val):
        """
        Reconstruct a vector at the given index.

        Args:
            index_id (str): ID of the index
            id_val (int): Index of the vector to reconstruct

        Returns:
            dict: Response containing the reconstructed vector or error message
        """
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} not found")

        try:
            index = self.indexes[index_id]

            # Check if this is a valid vector index
            if id_val >= index.ntotal:
                return error_response(f"Vector index {id_val} out of range (0-{index.ntotal-1})")

            # For IDMap indices, we need to check if the ID exists
            if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                if not hasattr(index, "id_map") or id_val not in index.id_map:
                    return error_response(f"ID {id_val} not found in index")

            # Check if this is a binary index
            is_binary = False
            for binary_class in [
                faiss.IndexBinaryFlat,
                faiss.IndexBinaryIVF,
                faiss.IndexBinaryHash,
            ]:
                if isinstance(index, binary_class):
                    is_binary = True
                    break

            # Reconstruct the vector
            if is_binary:
                try:
                    # For binary indices
                    dimension = self.dimensions[index_id]
                    byte_dimension = (dimension + 7) // 8
                    binary_vector = np.zeros(byte_dimension, dtype=np.uint8)

                    # Reconstruct the binary vector
                    index.reconstruct(int(id_val), binary_vector)

                    # Print debug info
                    debug_msg = (
                        f"Reconstructed binary vector: shape={binary_vector.shape}, "
                        f"dtype={binary_vector.dtype}"
                    )
                    logger.debug(debug_msg)

                    # Convert to numpy array with correct shape for binary_to_float
                    binary_vector = binary_vector.reshape(1, -1)

                    # Convert binary vector to float vector for response
                    vector = binary_to_float(binary_vector, dimension)[0]

                    return success_response({"vector": vector})
                except Exception as e:
                    logger.debug(f"Error in binary reconstruction: {str(e)}")
                    return error_response(f"Error in binary reconstruction: {str(e)}")
            else:
                # For standard float indices
                vector = index.reconstruct(int(id_val)).tolist()
                return success_response({"vector": vector})
        except Exception as e:
            return error_response(f"Error reconstructing vector: {str(e)}")

    def reconstruct_n(self, index_id, start_idx, num_vectors):
        """
        Reconstruct a batch of vectors starting at the specified index.

        This method is more efficient than calling reconstruct() multiple times
        when reconstructing many vectors.

        Args:
            index_id (str): ID of the index
            start_idx (int): Starting index for reconstruction
            num_vectors (int): Number of vectors to reconstruct

        Returns:
            dict: Response containing the reconstructed vectors or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} not found"}

        try:
            index = self.indexes[index_id]

            # Validate parameters
            if start_idx < 0:
                return {"success": False, "error": "Starting index cannot be negative"}

            if num_vectors <= 0:
                return {"success": False, "error": "Number of vectors must be positive"}

            # Check if the range is valid
            if start_idx + num_vectors > index.ntotal:
                return {
                    "success": False,
                    "error": (
                        f"Range {start_idx}:{start_idx+num_vectors} exceeds "
                        f"index size {index.ntotal}"
                    ),
                }

            # Reconstruct vectors
            if hasattr(index, "reconstruct_n"):
                # Use native reconstruct_n method if available
                vectors = index.reconstruct_n(start_idx, num_vectors)
            else:
                # Fall back to individual reconstruction
                vectors = np.zeros((num_vectors, self.dimensions[index_id]), dtype=np.float32)
                for i in range(num_vectors):
                    vectors[i] = index.reconstruct(start_idx + i)

            # Convert to list of lists for serialization
            vectors_list = vectors.tolist()

            return {
                "success": True,
                "vectors": vectors_list,
                "start_idx": start_idx,
                "num_vectors": num_vectors
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_index_stats(self, index_id):
        """
        Get basic statistics about the specified index.

        Args:
            index_id (str): ID of the index to get stats for

        Returns:
            dict: Response containing index statistics or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} not found"}

        index = self.indexes[index_id]
        dimension = self.dimensions[index_id]
        ntotal = index.ntotal

        # Basic stats for all indices
        stats = {
            "success": True,
            "index_id": index_id,
            "dimension": dimension,
            "ntotal": ntotal,
        }

        return stats

    def get_index_status(self, index_id):
        """
        Get detailed status information about the specified index.

        Args:
            index_id (str): ID of the index to get status for

        Returns:
            dict: Response containing index status details or error message
        """
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} not found")

        index = self.indexes[index_id]
        dimension = self.dimensions[index_id]
        ntotal = index.ntotal

        # Basic status information common to all index types
        status = {
            "index_id": index_id,
            "dimension": dimension,
            "ntotal": ntotal,
            "index_type": type(index).__name__,
        }

        # Add training information
        training_requirements = get_training_requirements(index)
        is_ready, reason = is_trained_for_use(index)

        status.update({
            "is_trained": training_requirements["is_trained"],
            "requires_training": training_requirements["requires_training"],
            "is_ready_for_use": is_ready,
        })

        if not is_ready:
            status["ready_reason"] = reason

        if training_requirements["requires_training"]:
            status["training_info"] = training_requirements
            recommended_vectors = estimate_training_vectors_needed(index)
            if recommended_vectors:
                status["recommended_training_vectors"] = recommended_vectors

        # Add index-specific parameters based on the index type
        if isinstance(index, faiss.IndexIVF):
            status.update({
                "nlist": index.nlist,
                "nprobe": index.nprobe,
                "quantizer_type": type(index.quantizer).__name__
            })
        elif isinstance(index, faiss.IndexHNSW):
            status.update({
                "hnsw_m": index.hnsw.M,
                "ef_search": index.hnsw.efSearch,
                "ef_construction": index.hnsw.efConstruction
            })
        elif isinstance(index, faiss.IndexPQ):
            status.update({
                "pq_m": index.pq.M,
                "pq_nbits": index.pq.nbits,
            })
        elif isinstance(index, faiss.IndexScalarQuantizer):
            status.update({
                "sq_type": str(index.sq_type),
            })

        # Add base index relationship if applicable
        if index_id in self.base_indexes:
            status["base_index_id"] = self.base_indexes[index_id]

        return format_index_status(index_id, status)

    def get_index_info(self, index_id):
        """
        Get detailed metadata and configuration information about an index.

        This provides more comprehensive information than get_index_status,
        including details about the index structure, configuration parameters,
        and underlying storage characteristics.

        Args:
            index_id (str): ID of the index to get info for

        Returns:
            dict: Response containing detailed index information or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} not found"}

        index = self.indexes[index_id]

        # Start with basic status information
        status = self.get_index_status(index_id)
        if not status.get("success", False):
            return status

        # Remove success field before including in the info response
        if "success" in status:
            del status["success"]

        # Build detailed info
        info = {
            "success": True,
            "status": status,
            "metrics": {},
            "configuration": {},
            "storage": {},
        }

        # Add metrics information
        info["metrics"].update({
            "vectors_count": status.get("ntotal", 0),
            "dimension": status.get("dimension", 0),
            # Assuming float32 (4 bytes)
            "byte_size_per_vector": status.get("dimension", 0) * 4,
        })

        # Add estimated memory usage (approximate)
        vector_memory = status.get("ntotal", 0) * status.get("dimension", 0) * 4  # float32 vectors
        index_overhead = 0

        # Estimate index overhead based on type
        if isinstance(index, faiss.IndexFlat):
            # Flat indices store vectors directly with minimal overhead
            index_overhead = status.get("ntotal", 0) * 4  # Minimal overhead
        elif isinstance(index, faiss.IndexIVF):
            # IVF has inverted lists overhead
            index_overhead = status.get("nlist", 100) * 100 + status.get("ntotal", 0) * 8
        elif isinstance(index, faiss.IndexHNSW):
            # HNSW has graph structure overhead
            index_overhead = status.get("ntotal", 0) * status.get("hnsw_m", 32) * 8

        info["storage"].update({
            "estimated_memory_bytes": vector_memory + index_overhead,
            "persistent": self.data_dir is not None,
            "storage_path": str(self.data_dir / index_id) if self.data_dir else None,
        })

        # Add configuration details
        if isinstance(index, faiss.IndexFlat):
            info["configuration"]["metric"] = "L2" if isinstance(index, faiss.IndexFlatL2) else "IP"
        elif isinstance(index, faiss.IndexIVF):
            info["configuration"].update({
                "nlist": status.get("nlist", 100),
                "nprobe": status.get("nprobe", 1),
                "metric": "L2" if index.metric_type == faiss.METRIC_L2 else "IP",
            })
        elif isinstance(index, faiss.IndexHNSW):
            info["configuration"].update({
                "M": status.get("hnsw_m", 32),
                "efSearch": status.get("ef_search", 16),
                "efConstruction": status.get("ef_construction", 40),
                "metric": "L2" if index.metric_type == faiss.METRIC_L2 else "IP",
            })
        elif isinstance(index, faiss.IndexPQ):
            info["configuration"].update({
                "M": status.get("pq_m", 8),
                "nbits": status.get("pq_nbits", 8),
                "metric": "L2" if index.metric_type == faiss.METRIC_L2 else "IP",
            })

        # Add base index info if this is an IDMap
        if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            idmap_type = "IDMap2" if isinstance(index, faiss.IndexIDMap2) else "IDMap"
            info["configuration"]["idmap_type"] = idmap_type
            if index_id in self.base_indexes:
                info["configuration"]["base_index_id"] = self.base_indexes[index_id]

        return info

    def set_parameter(self, index_id, param_name, param_value):
        """
        Set a runtime parameter for the specified index.

        This method allows changing search parameters without recreating the index.
        Supported parameters vary by index type:

        - For IVF indices: nprobe, quantizer_type
        - For HNSW indices: efSearch, efConstruction, M
        - For PQ indices: M, nbits, use_precomputed_table
        - For SQ indices: qtype
        - For Binary indices: nprobe (for BINARY_IVF)
        - For IDMap indices: add_id_range, remove_ids
        - For multiple indices: metric_type

        Args:
            index_id (str): ID of the index to modify
            param_name (str): Name of the parameter to set
            param_value: Value to set for the parameter

        Returns:
            dict: Response indicating success or failure
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} not found"}

        index = self.indexes[index_id]

        try:
            # IVF index parameters
            if isinstance(index, faiss.IndexIVF):
                if param_name == "nprobe":
                    # Validate nprobe value
                    if not isinstance(param_value, int) or param_value <= 0:
                        return {"success": False, "error": "nprobe must be a positive integer"}

                    # Set nprobe parameter
                    index.nprobe = param_value
                    return {
                        "success": True,
                        "message": f"Set nprobe={param_value} for index {index_id}"
                    }

                # Additional IVF parameters
                elif param_name == "quantizer_type" and hasattr(index, "quantizer"):
                    return {
                        "success": False,
                        "error": "Changing quantizer_type is not supported after index creation",
                    }

            # HNSW index parameters
            elif isinstance(index, faiss.IndexHNSW):
                if param_name == "efSearch":
                    # Validate efSearch value
                    if not isinstance(param_value, int) or param_value <= 0:
                        return {"success": False, "error": "efSearch must be a positive integer"}

                    # Set efSearch parameter
                    index.hnsw.efSearch = param_value
                    return {
                        "success": True,
                        "message": f"Set efSearch={param_value} for index {index_id}"
                    }

                elif param_name == "efConstruction":
                    # Validate efConstruction value
                    if not isinstance(param_value, int) or param_value <= 0:
                        return {
                            "success": False,
                            "error": "efConstruction must be a positive integer",
                        }

                    # Set efConstruction parameter
                    index.hnsw.efConstruction = param_value
                    return {
                        "success": True,
                        "message": f"Set efConstruction={param_value} for index {index_id}"
                    }

                elif param_name == "M":
                    return {
                        "success": False,
                        "error": "Changing HNSW M parameter after index creation is not supported",
                    }

            # Product Quantization (PQ) index parameters
            elif isinstance(index, faiss.IndexPQ):
                if param_name == "use_precomputed_table":
                    index.use_precomputed_table = bool(param_value)
                    return {
                        "success": True,
                        "message": f"Set use_precomputed_table={param_value} for index {index_id}"
                    }

                elif param_name in ["M", "nbits"]:
                    return {
                        "success": False,
                        "error": f"Changing PQ {param_name} after index creation is not supported",
                    }

            # Scalar Quantization (SQ) index parameters
            elif isinstance(index, faiss.IndexScalarQuantizer):
                if param_name == "qtype":
                    return {
                        "success": False,
                        "error": "Changing SQ qtype after index creation is not supported",
                    }

            # Binary index parameters
            elif isinstance(index, faiss.IndexBinaryIVF):
                if param_name == "nprobe":
                    if not isinstance(param_value, int) or param_value <= 0:
                        return {"success": False, "error": "nprobe must be a positive integer"}

                    index.nprobe = param_value
                    return {
                        "success": True,
                        "message": f"Set nprobe={param_value} for binary index {index_id}"
                    }

            # IDMap index parameters
            elif isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                if param_name == "add_id_range":
                    return {
                        "success": False,
                        "error": "add_id_range should be used with add_with_ids method",
                    }

                elif param_name == "remove_ids":
                    return {
                        "success": False,
                        "error": "remove_ids should be used with the remove_ids method",
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
            return {"success": False, "error": f"Error setting parameter: {str(e)}"}

    def get_parameter(self, index_id, param_name):
        """
        Get the current value of a parameter for the specified index.

        Args:
            index_id (str): ID of the index
            param_name (str): Name of the parameter to retrieve

        Returns:
            dict: Response containing the parameter value or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} not found"}

        index = self.indexes[index_id]

        try:
            # Parameter handlers for different index types
            if param_name == "nprobe" and isinstance(index, faiss.IndexIVF):
                return {
                    "success": True,
                    "param_name": "nprobe",
                    "param_value": index.nprobe
                }

            elif param_name == "efSearch" and isinstance(index, faiss.IndexHNSW):
                return {
                    "success": True,
                    "param_name": "efSearch",
                    "param_value": index.hnsw.efSearch
                }

            elif param_name == "efConstruction" and isinstance(index, faiss.IndexHNSW):
                return {
                    "success": True,
                    "param_name": "efConstruction",
                    "param_value": index.hnsw.efConstruction
                }

            elif param_name == "is_trained":
                # This parameter is available for all index types
                is_trained = getattr(index, "is_trained", True)
                return {
                    "success": True,
                    "param_name": "is_trained",
                    "param_value": is_trained
                }

            elif param_name == "dimension":
                return {
                    "success": True,
                    "param_name": "dimension",
                    "param_value": self.dimensions[index_id]
                }

            elif param_name == "ntotal":
                return {
                    "success": True,
                    "param_name": "ntotal",
                    "param_value": index.ntotal
                }

            else:
                return {
                    "success": False,
                    "error": f"Parameter {param_name} not supported for this index type"
                }

        except Exception as e:
            return {"success": False, "error": f"Error getting parameter: {str(e)}"}

    def list_indexes(self):
        """
        List all available indexes.

        Returns:
            dict: Response containing list of indexes or error message
        """
        try:
            index_list = []
            for index_id in self.indexes:
                index_info = {
                    "index_id": index_id,
                    "dimension": self.dimensions[index_id],
                    "vector_count": self.indexes[index_id].ntotal,
                }

                # Add IDMap specific information if applicable
                if index_id in self.base_indexes:
                    index_info["base_index_id"] = self.base_indexes[index_id]
                    index_info["is_idmap"] = isinstance(self.indexes[index_id], faiss.IndexIDMap)
                    index_info["is_idmap2"] = isinstance(self.indexes[index_id], faiss.IndexIDMap2)

                index_list.append(index_info)

            return {"success": True, "indexes": index_list}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_indices(self):
        """
        Alias for list_indexes to maintain API compatibility.

        Returns:
            dict: Response containing list of indexes or error message
        """
        return self.list_indexes()

    def train_index(self, index_id, training_vectors):
        """
        Train an index with the provided vectors (required for IVF indices).

        Args:
            index_id (str): ID of the target index
            training_vectors (list): List of vectors to use for training

        Returns:
            dict: Response containing success status or error message
        """
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} does not exist")

        try:
            index = self.indexes[index_id]

            # Check if this is an IndexPreTransform
            is_pretransform = isinstance(index, faiss.IndexPreTransform)

            # Check if index requires training using our utility function
            if not requires_training(index) and not is_pretransform:
                # Get detailed info about why training is not needed
                training_info = get_training_requirements(index)

                return success_response({
                    "index_id": index_id,
                    "is_trained": True,
                    "training_skipped": True,
                    "training_info": training_info
                }, message="This index type does not require training")

            # Convert vectors to numpy array and validate dimensions
            vectors_np = np.array(training_vectors, dtype=np.float32)
            if vectors_np.shape[1] != self.dimensions[index_id]:
                return error_response(
                    f"Training vector dimension mismatch. Expected {self.dimensions[index_id]}, "
                    f"got {vectors_np.shape[1]}"
                )

            # Special handling for IndexPreTransform
            if is_pretransform:
                # Extract the transform and base index
                transform = index.chain.at(0)
                base_index = index.index

                # Get transform training requirements
                transform_requires_training = (
                    hasattr(transform, "is_trained") and not transform.is_trained
                )

                # Train the transform if needed
                if transform_requires_training:
                    if not train_transform(transform, vectors_np):
                        return error_response(
                            "Failed to train transformation component",
                            code="TRANSFORM_TRAINING_ERROR"
                        )

                # Check if the base index needs training
                base_index_requires_training = (
                    hasattr(base_index, "is_trained") and not base_index.is_trained
                )

                if base_index_requires_training:
                    # For indices that need training after transformation,
                    # we need to apply the transform to the training vectors
                    if hasattr(transform, "apply_py"):
                        transformed_vectors = transform.apply_py(vectors_np)
                    else:
                        # Fallback method using transform.apply()
                        output_dim = (
                            transform.d_out
                            if hasattr(transform, "d_out")
                            else vectors_np.shape[1]
                        )
                        transformed_vectors = np.zeros(
                            (len(vectors_np), output_dim), dtype=np.float32
                        )
                        for i, vec in enumerate(vectors_np):
                            transform.apply(
                                1,
                                faiss.swig_ptr(vec),
                                faiss.swig_ptr(transformed_vectors[i]),
                            )

                    # Now train the base index with transformed vectors
                    base_index.train(transformed_vectors)

                # Get training status after training
                transform_trained = not transform_requires_training or transform.is_trained
                base_index_trained = not base_index_requires_training or base_index.is_trained

                return success_response({
                    "index_id": index_id,
                    "trained_with": len(training_vectors),
                    "is_trained": transform_trained and base_index_trained,
                    "transform_trained": transform_trained,
                    "base_index_trained": base_index_trained,
                    "index_type": "IndexPreTransform"
                }, message="Transformed index successfully trained")

            # Check if we have enough training vectors
            recommended_vectors = estimate_training_vectors_needed(index)
            has_enough_vectors = True
            recommendation = None

            if recommended_vectors is not None and len(training_vectors) < recommended_vectors:
                has_enough_vectors = False
                recommendation = (
                    f"For optimal results, consider using at least "
                    f"{recommended_vectors} training vectors"
                )

            # Train the index
            index.train(vectors_np)

            # Get updated training requirements after training
            training_info = get_training_requirements(index)

            logger.debug(f"Trained index {index_id} with {len(training_vectors)} vectors")
            return success_response({
                "index_id": index_id,
                "trained_with": len(training_vectors),
                "is_trained": index.is_trained,
                "has_enough_vectors": has_enough_vectors,
                "recommendation": recommendation,
                "training_info": training_info
            }, message="Index successfully trained")

        except Exception as e:
            return error_response(
                f"Training error: {str(e)}",
                code="TRAINING_ERROR",
                details={
                    "index_id": index_id,
                    "vector_count": (
                        len(training_vectors)
                        if "training_vectors" in locals()
                        else 0
                    ),
                },
            )

    def get_transform_info(self, index_id):
        """
        Get information about the transformation component of an IndexPreTransform.

        Args:
            index_id (str): ID of the index

        Returns:
            dict: Response containing transformation information or error message
        """
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} not found")

        index = self.indexes[index_id]

        # Check if this is an IndexPreTransform
        if not isinstance(index, faiss.IndexPreTransform):
            return error_response(
                f"Index {index_id} is not an IndexPreTransform",
                code="NOT_TRANSFORM_INDEX"
            )

        try:
            # Extract information about the transformation chain
            info = {
                "index_id": index_id,
                "type": "IndexPreTransform",
                "input_dimension": self.dimensions[index_id],
                "output_dimension": index.index.d,
                "chain_size": index.chain.size(),
                "is_trained": True
            }

            # Get info for each transformation in the chain
            transforms_info = []
            for i in range(index.chain.size()):
                transform = index.chain.at(i)
                transform_type = type(transform).__name__

                transform_info = {
                    "type": transform_type,
                    "is_trained": is_transform_trained(transform)
                }

                # Add transform-specific information
                if isinstance(transform, faiss.PCAMatrix):
                    transform_info.update({
                        "input_dim": transform.d_in,
                        "output_dim": transform.d_out,
                        "do_whitening": transform.do_whitening
                    })

                elif isinstance(transform, faiss.NormalizationTransform):
                    transform_info.update({
                        "dimension": transform.d
                    })

                elif isinstance(transform, faiss.OPQMatrix):
                    transform_info.update({
                        "input_dim": transform.d_in,
                        "output_dim": transform.d_out,
                        "M": transform.M
                    })

                transforms_info.append(transform_info)

                # Update overall training status
                if hasattr(transform, "is_trained") and not transform.is_trained:
                    info["is_trained"] = False

            info["transforms"] = transforms_info

            # Get base index information
            base_index = index.index
            info["base_index"] = {
                "type": type(base_index).__name__,
                "dimension": base_index.d,
                "is_trained": getattr(base_index, "is_trained", True)
            }

            if not info["base_index"]["is_trained"]:
                info["is_trained"] = False

            return success_response(info)

        except Exception as e:
            return error_response(f"Error getting transform info: {str(e)}")

    def apply_transform(self, index_id, vectors):
        """
        Apply the transformation of an IndexPreTransform to the provided vectors.

        Args:
            index_id (str): ID of the index
            vectors (list): List of vectors to transform

        Returns:
            dict: Response containing transformed vectors or error message
        """
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} not found")

        index = self.indexes[index_id]

        # Check if this is an IndexPreTransform
        if not isinstance(index, faiss.IndexPreTransform):
            return error_response(
                f"Index {index_id} is not an IndexPreTransform",
                code="NOT_TRANSFORM_INDEX"
            )

        try:
            # Convert vectors to numpy array
            vectors_np = np.array(vectors, dtype=np.float32)

            # Verify input dimensions
            if vectors_np.shape[1] != self.dimensions[index_id]:
                return error_response(
                    f"Vector dimension mismatch: expected {self.dimensions[index_id]}, "
                    f"got {vectors_np.shape[1]}"
                )

            # Apply the transformation
            output_dim = index.index.d
            transformed_vectors = np.zeros((len(vectors), output_dim), dtype=np.float32)

            # Apply the transformation
            for i, vec in enumerate(vectors_np):
                transformed_vector = np.zeros(output_dim, dtype=np.float32)
                index.apply_chain(1, faiss.swig_ptr(vec), faiss.swig_ptr(transformed_vector))
                transformed_vectors[i] = transformed_vector

            return success_response({
                "index_id": index_id,
                "input_vectors": len(vectors),
                "input_dimension": self.dimensions[index_id],
                "output_dimension": output_dim,
                "transformed_vectors": transformed_vectors.tolist()
            })

        except Exception as e:
            return error_response(f"Error applying transformation: {str(e)}")

    def reset(self, index_id):
        """
        Reset an index by removing all vectors while preserving training.

        This is more efficient than deleting and recreating the index
        when the training data is large or expensive to recompute.

        Args:
            index_id (str): ID of the index to reset

        Returns:
            dict: Response indicating success or failure
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} not found"}

        index = self.indexes[index_id]

        try:
            # Handle different index types
            if isinstance(index, faiss.IndexFlat):
                # For flat indices, we can create a new index with the same parameters
                dimension = self.dimensions[index_id]
                if isinstance(index, faiss.IndexFlatL2):
                    self.indexes[index_id] = faiss.IndexFlatL2(dimension)
                elif isinstance(index, faiss.IndexFlatIP):
                    self.indexes[index_id] = faiss.IndexFlatIP(dimension)
                else:
                    # Generic flat index
                    self.indexes[index_id] = faiss.IndexFlat(dimension, index.metric_type)

            elif isinstance(index, faiss.IndexIVF):
                # For IVF indices, we need to preserve the trained quantizer
                if index.is_trained:
                    # Extract important parameters
                    dimension = self.dimensions[index_id]
                    nlist = index.nlist
                    metric_type = index.metric_type
                    quantizer = index.quantizer

                    # Create a new index with the same quantizer and parameters
                    if isinstance(index, faiss.IndexIVFFlat):
                        new_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
                    elif isinstance(index, faiss.IndexIVFPQ):
                        # Get PQ parameters
                        pq = index.pq
                        m = pq.M
                        nbits = pq.nbits
                        new_index = faiss.IndexIVFPQ(
                            quantizer,
                            dimension,
                            nlist,
                            m,
                            nbits,
                            metric_type,
                        )
                    elif isinstance(index, faiss.IndexIVFScalarQuantizer):
                        # Get scalar quantizer parameters
                        sq_type = index.sq_type
                        new_index = faiss.IndexIVFScalarQuantizer(
                            quantizer, dimension, nlist, sq_type, metric_type,
                        )
                    else:
                        return {
                            "success": False,
                            "error": "Unsupported IVF index type for reset"
                        }

                    # The new index is already trained because it uses the trained quantizer
                    new_index.is_trained = True

                    # Transfer other parameters
                    new_index.nprobe = index.nprobe

                    # Replace the old index
                    self.indexes[index_id] = new_index
                else:
                    # If not trained, we can just remove all vectors (none should exist)
                    return {
                        "success": False,
                        "error": "Index is not trained yet, no need to reset"
                    }

            elif isinstance(index, faiss.IndexHNSW):
                # HNSW indices can't be easily reset - we need to recreate
                dimension = self.dimensions[index_id]
                M = index.hnsw.M
                metric_type = index.metric_type

                # Create new HNSW index with same parameters
                new_index = faiss.IndexHNSW(dimension, M, metric_type)

                # Transfer other parameters
                new_index.hnsw.efConstruction = index.hnsw.efConstruction
                new_index.hnsw.efSearch = index.hnsw.efSearch

                # Replace the old index
                self.indexes[index_id] = new_index

            elif isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                # For IDMap indices, we need to reset the base index and the ID mapping
                if index_id in self.base_indexes:
                    base_index_id = self.base_indexes[index_id]

                    # Reset the base index first
                    base_result = self.reset(base_index_id)
                    if not base_result.get("success", False):
                        return base_result

                    # Re-create the IDMap layer
                    base_index = self.indexes[base_index_id]
                    is_idmap2 = isinstance(index, faiss.IndexIDMap2)

                    if is_idmap2:
                        self.indexes[index_id] = faiss.IndexIDMap2(base_index)
                    else:
                        self.indexes[index_id] = faiss.IndexIDMap(base_index)
                else:
                    # No base index relationship found
                    return {
                        "success": False,
                        "error": "Base index relationship not found for IDMap index"
                    }
            else:
                # For other index types, we might not be able to reset easily
                return {
                    "success": False,
                    "error": f"Reset not supported for index type {type(index).__name__}"
                }

            return {"success": True, "message": f"Index {index_id} has been reset"}

        except Exception as e:
            return {"success": False, "error": f"Error resetting index: {str(e)}"}

    def clear(self, index_id):
        """
        Completely clear an index, removing both vectors and training.

        This effectively recreates the index from scratch.

        Args:
            index_id (str): ID of the index to clear

        Returns:
            dict: Response indicating success or failure
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} not found"}

        try:
            # Get the original dimension
            dimension = self.dimensions[index_id]

            # Remove the index
            del self.indexes[index_id]

            # Check if this is a base index for any IDMap indices
            for idx_id, base_id in list(self.base_indexes.items()):
                if base_id == index_id:
                    # Remove dependent IDMap indices
                    if idx_id in self.indexes:
                        del self.indexes[idx_id]
                    if idx_id in self.dimensions:
                        del self.dimensions[idx_id]
                    del self.base_indexes[idx_id]

            # Recreate with the same type
            index_type = "L2"  # Default type

            # Create a new index with the same ID
            return self.create_index(index_id, dimension, index_type)

        except Exception as e:
            return {"success": False, "error": f"Error clearing index: {str(e)}"}

    def merge_indices(self, target_index_id, source_index_ids):
        """
        Merge multiple source indices into a target index.

        Args:
            target_index_id (str): ID of the target index to merge into
            source_index_ids (list): List of source index IDs to merge from

        Returns:
            dict: Response indicating success or failure
        """
        # Use the specialized implementation which handles more index types
        return specialized_operations.merge_indices(self, target_index_id, source_index_ids)

    def delete_index(self, index_id):
        """
        Delete an index from the server.

        Args:
            index_id (str): ID of the index to delete

        Returns:
            dict: Response indicating success or failure
        """
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} not found")

        try:
            # Delete the index
            del self.indexes[index_id]
            del self.dimensions[index_id]

            # Remove from base_indexes if present
            for idx_id, base_id in list(self.base_indexes.items()):
                if base_id == index_id or idx_id == index_id:
                    del self.base_indexes[idx_id]

            return success_response(
                {"index_id": index_id},
                message=f"Index {index_id} deleted successfully"
            )
        except Exception as e:
            logger.exception(f"Error deleting index: {e}")
            return error_response(f"Error deleting index: {str(e)}")

    def range_search(self, index_id, query_vectors, radius):
        """
        Search for vectors within a specified radius.

        Args:
            index_id (str): ID of the target index
            query_vectors (list): List of query vectors
            radius (float): Search radius (maximum distance)

        Returns:
            dict: Response containing search results or error message
        """
        return self._run_with_timeout(
            self._range_search, index_id, query_vectors, radius
        )

    def _range_search(self, index_id, query_vectors, radius):
        """Internal implementation of range_search that can be run with timeout."""
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} does not exist")

        try:
            # Convert query vectors to numpy array and validate dimensions
            query_np = np.array(query_vectors, dtype=np.float32)
            if query_np.shape[1] != self.dimensions[index_id]:
                return error_response(
                    f"Query dimension mismatch: expected {self.dimensions[index_id]}, "
                    f"got {query_np.shape[1]}"
                )

            index = self.indexes[index_id]

            # Check if the index is ready for use
            is_ready, reason = is_trained_for_use(index)
            if not is_ready:
                return error_response(
                    f"Index is not ready: {reason}",
                    code="NOT_TRAINED"
                )

            # Check if index supports range_search directly
            if hasattr(index, "range_search"):
                # Process each query vector separately
                all_results = []

                for i, query in enumerate(query_np):
                    # Range search returns (lims, distances, indices) - 3 values!
                    lims, distances, indices = index.range_search(query.reshape(1, -1), radius)

                    # Extract results for this query from the flattened arrays
                    # lims[0] to lims[1] gives the range for the first (and only) query
                    start_idx = lims[0]
                    end_idx = lims[1]

                    query_distances = distances[start_idx:end_idx]
                    query_indices = indices[start_idx:end_idx]

                    # Format result for this query
                    result = {
                        "query_index": i,
                        "num_results": len(query_indices),
                        "indices": (
                            query_indices.tolist()
                            if isinstance(query_indices, np.ndarray)
                            else query_indices
                        ),
                        "distances": (
                            query_distances.tolist()
                            if isinstance(query_distances, np.ndarray)
                            else query_distances
                        ),
                    }
                    all_results.append(result)

                return success_response(
                    {
                        "results": all_results,
                        "num_queries": len(query_vectors),
                        "radius": radius
                    }
                )
            else:
                return error_response(
                    f"Index type {type(index).__name__} does not support range search"
                )

        except Exception as e:
            logger.exception(f"Error in range_search: {e}")
            return error_response(f"Error in range_search: {str(e)}")

    def optimize_index(self, index_id, optimization_level=1):
        """
        Optimize an index for better performance.

        This applies various optimizations based on the index type and optimization level.

        Args:
            index_id (str): ID of the index to optimize
            optimization_level (int): Level of optimization (1-3, higher is more aggressive)

        Returns:
            dict: Response indicating success or failure
        """
        return specialized_operations.optimize_index(self, index_id, optimization_level)

    def compute_clustering(self, vectors, n_clusters, metric_type="L2", niter=25):
        """
        Compute k-means clustering on a set of vectors.

        Args:
            vectors (list): Input vectors
            n_clusters (int): Number of clusters to compute
            metric_type (str): Distance metric to use ("L2" or "IP")
            niter (int): Number of iterations for k-means

        Returns:
            dict: Response containing centroids and cluster assignments
        """
        # Convert metric type string to FAISS constant
        if metric_type.upper() == "IP":
            metric_type_const = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type_const = faiss.METRIC_L2

        return specialized_operations.compute_clustering(
            vectors, n_clusters, metric_type_const, niter
        )

    def recluster_index(self, index_id, n_clusters=None, sample_ratio=0.5):
        """
        Re-cluster an index to optimize its structure.

        Args:
            index_id (str): ID of the index to recluster
            n_clusters (int): Number of clusters (None to use existing)
            sample_ratio (float): Ratio of vectors to sample for clustering

        Returns:
            dict: Response indicating success or failure
        """
        return specialized_operations.recluster_index(
            self, index_id, n_clusters, sample_ratio
        )

    def batch_add_with_ids(self, index_id, vectors, ids, batch_size=1000):
        """
        Add vectors with IDs to an index in batches.

        Args:
            index_id (str): ID of the index
            vectors (list): Vectors to add
            ids (list): IDs to associate with vectors
            batch_size (int): Batch size for adding

        Returns:
            dict: Response indicating success or failure
        """
        return specialized_operations.batch_add_with_ids(
            self, index_id, vectors, ids, batch_size
        )

    def hybrid_search(self, index_id, query_vectors, vector_weight=0.5,
                      metadata_filter=None, k=10, params=None):
        """
        Perform hybrid search combining vector similarity with metadata filtering.

        Args:
            index_id (str): ID of the index to search in
            query_vectors (list): Query vectors for similarity search
            vector_weight (float): Weight given to vector similarity vs metadata (0.0-1.0)
            metadata_filter (dict): Filter expression for metadata
            k (int): Number of results to return
            params (dict): Additional search parameters

        Returns:
            dict: Response containing search results
        """
        return specialized_operations.hybrid_search(
            self, index_id, query_vectors, vector_weight, metadata_filter, k, params
        )


def serialize_message(data):
    """
    Serialize a message to binary format using msgpack.

    This function converts Python data structures to efficient binary format
    for transmission over ZeroMQ sockets. It handles both regular JSON-serializable
    data and complex search results containing numpy arrays.

    Args:
        data (dict): Data to serialize - typically response dictionaries with
                    success/error status and result data

    Returns:
        bytes: Serialized binary data ready for network transmission

    Implementation Notes:
        - Uses msgpack for efficient binary serialization
        - Handles numpy arrays in search results automatically
        - Sets use_bin_type=True for better binary data handling
        - Maintains compatibility with various Python data types

    Usage:
        Used by the ZeroMQ server to encode responses before sending them
        to clients. The binary format is more efficient than JSON for
        large result sets containing vector data.
    """
    if isinstance(data, dict) and "results" in data and data.get("success", False):
        # Special handling for search results with numpy arrays
        # These may contain large amounts of numerical data that benefit
        # from efficient binary encoding
        return msgpack.packb(data, use_bin_type=True)
    else:
        # Regular JSON-serializable data (errors, status responses, etc.)
        # Still use binary encoding for consistency and efficiency
        return msgpack.packb(data, use_bin_type=True)


def deserialize_message(data):
    """
    Deserialize a binary message using msgpack.

    This function converts binary msgpack data back to Python data structures.
    It includes error handling to gracefully manage corrupted or invalid messages.

    Args:
        data (bytes): Binary data to deserialize from client requests

    Returns:
        dict: Deserialized data or error message if deserialization fails

    Error Handling:
        If deserialization fails, returns a structured error response instead
        of raising an exception. This prevents server crashes from malformed
        client messages.

    Implementation Notes:
        - Uses raw=False to ensure strings are returned as str, not bytes
        - Captures all msgpack exceptions to prevent server instability
        - Returns consistent error format for failed deserialization
    """
    try:
        # Deserialize binary data to Python objects
        # raw=False ensures strings are decoded properly
        return msgpack.unpackb(data, raw=False)
    except Exception as e:
        # Handle any deserialization errors gracefully
        # Return structured error instead of crashing the server
        return {"success": False, "error": f"Failed to deserialize message: {str(e)}"}


def authenticate_request(request, auth_keys):
    """
    Authenticate a request using API keys and manage tenant isolation.

    This function provides security for multi-tenant deployments by validating
    API keys and associating requests with specific tenant IDs. It supports
    both authenticated and unauthenticated operation modes.

    Args:
        request (dict): Request data containing potential API key and operation data
        auth_keys (dict): Dictionary mapping API keys to tenant IDs, or None/empty
                         to disable authentication

    Returns:
        tuple: (is_authenticated, error_message) where:
            - is_authenticated (bool): True if request is authorized
            - error_message (str or None): Error description if auth failed

    Authentication Flow:
        1. Check if authentication is enabled (auth_keys provided)
        2. Extract API key from request
        3. Validate API key against known keys
        4. Add tenant_id to request for downstream processing
        5. Return success/failure status

    Tenant Isolation:
        When authentication succeeds, the tenant_id is added to the request.
        This enables downstream operations to isolate data by tenant,
        preventing cross-tenant data access.

    Security Considerations:
        - API keys should be kept secure and rotated regularly
        - Tenant IDs enable data isolation but require proper implementation
        - Failed authentication is logged for security monitoring
    """
    # Check if authentication is disabled
    if not auth_keys:
        # Authentication disabled - allow all requests
        # This mode is suitable for single-tenant or development environments
        return True, None

    # Extract API key from request headers/data
    api_key = request.get("api_key")
    if not api_key:
        # Request missing required API key
        return False, "API key required"

    # Validate API key against known tenant mappings
    tenant_id = auth_keys.get(api_key)
    if not tenant_id:
        # API key not found in authorized keys
        return False, "Invalid API key"

    # Authentication successful - add tenant context to request
    # This enables tenant isolation in downstream operations
    request["tenant_id"] = tenant_id
    return True, None


def register_specialized_handlers(server_instance):
    """
    Register specialized operation handlers from the specialized_operations module.

    This function extends the server's capabilities by registering additional
    operation handlers for advanced FAISS operations. It provides a plugin-like
    architecture for extending server functionality without modifying core code.

    Args:
        server_instance: The FaissIndex server instance to extend with additional
                        operation handlers

    Implementation Details:
        - Imports specialized_operations module dynamically
        - Creates action_handlers dictionary mapping operation names to functions
        - Merges new handlers with existing server capabilities
        - Provides graceful handling if handlers already exist

    Registered Operations:
        - merge_indices: Combine multiple indices into one
        - compute_clustering: Perform k-means clustering on vectors
        - recluster_index: Re-optimize index clustering structure
        - hybrid_search: Combined vector and metadata search
        - batch_add_with_ids: Efficient batch vector insertion
        - optimize_index: Apply performance optimizations

    Architecture Benefits:
        - Modular design allows optional advanced features
        - Clean separation of core vs specialized functionality
        - Easy to add new operations without core modifications
        - Consistent handler interface across all operations
    """
    # Import specialized operations module
    # This is done here to avoid circular imports and keep core lightweight
    from . import specialized_operations

    # Define mapping of operation names to handler functions
    # Each handler receives the server instance as first parameter
    action_handlers = {
        "merge_indices": specialized_operations.merge_indices,
        "compute_clustering": specialized_operations.compute_clustering,
        "recluster_index": specialized_operations.recluster_index,
        "hybrid_search": specialized_operations.hybrid_search,
        "batch_add_with_ids": specialized_operations.batch_add_with_ids,
        "optimize_index": specialized_operations.optimize_index,
    }

    # Add handlers to server instance
    # Create action_handlers dictionary if it doesn't exist
    if hasattr(server_instance, "action_handlers"):
        # Merge with existing handlers (specialized handlers take precedence)
        server_instance.action_handlers.update(action_handlers)
    else:
        # Create new action_handlers dictionary
        server_instance.action_handlers = action_handlers

    # Log successful registration for operational visibility
    # Note: Logging line commented out to avoid verbose output
    # logger.info(f"Registered {len(action_handlers)} specialized operation handlers")


def run_server(
    port=DEFAULT_PORT,
    bind_address=DEFAULT_BIND_ADDRESS,
    auth_keys=None,
    enable_auth=False,
    data_dir=None,
    socket_timeout=DEFAULT_SOCKET_TIMEOUT,
    high_water_mark=DEFAULT_HIGH_WATER_MARK,
    linger=DEFAULT_LINGER,
    log_level="WARNING",
):
    """
    Run the FAISSx ZeroMQ server on the specified port.

    This is the main entry point for starting the FAISSx server. It handles all
    aspects of server initialization, configuration, and the main request-response
    loop. The server uses ZeroMQ REP (reply) sockets for synchronous communication.

    Args:
        port (int): Port to run the server on (default from DEFAULT_PORT)
        bind_address (str): Address to bind the server to (default: all interfaces)
        auth_keys (list, optional): List of authentication keys for multi-tenant auth
        enable_auth (bool): Whether to enable API key authentication
        data_dir (str, optional): Directory to store persistent data (future feature)
        socket_timeout (int): Socket timeout in milliseconds for ZeroMQ operations
        high_water_mark (int): ZeroMQ high water mark for message buffering
        linger (int): ZeroMQ linger period for graceful socket shutdown
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Server Architecture:
        - Uses ZeroMQ REP (reply) socket pattern for request-response communication
        - Single-threaded request processing with background task worker for timeouts
        - MessagePack binary serialization for efficient data transfer
        - Supports both authenticated and unauthenticated operation modes
        - Comprehensive error handling with graceful degradation

    Request Processing Flow:
        1. Wait for incoming ZeroMQ message
        2. Deserialize request using MessagePack
        3. Authenticate request if auth is enabled (skip ping for health checks)
        4. Route request to appropriate handler based on action type
        5. Execute operation (potentially with timeout protection)
        6. Serialize and send response back to client

    Error Handling:
        - Socket errors are logged and handled gracefully
        - Authentication failures return structured error responses
        - Operation timeouts are handled with background task worker
        - Unexpected errors are captured and returned as error responses
        - Server continues running despite individual request failures

    Raises:
        Exception: If there's a critical error during server initialization
                  (port binding failure, invalid configuration, etc.)

    Example:
        >>> run_server(port=5432, enable_auth=True, auth_keys={"key123": "tenant1"})
    """
    # Configure logging system first
    # This sets up structured logging for operational visibility
    configure_logging(log_level)

    # Prepare concise startup information for operators
    storage_mode = f"Persistent ({data_dir})" if data_dir else "In-memory"
    auth_status = "Enabled" if enable_auth else "Disabled"

    # Display startup banner with configuration summary
    # This provides immediate feedback about server configuration
    print("\n---------------------------------------------\n")
    print("███████╗█████╗ ██╗███████╗███████╗")
    print("██╔════██╔══██╗██║██╔════╝██╔════╝")
    print("█████╗ ███████║██║███████╗███████╗ ██╗ ██╗")
    print("██╔══╝ ██╔══██║██║╚════██║╚════██║ ╚═██╔╝")
    print("██║    ██║  ██║██║███████║███████║ ██╔╝██╗")
    print("╚═╝    ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝ ╚═╝ ╚═╝")
    print("\n---------------------------------------------")
    print(f"* FAISSx {faissx_version} (c) Ran Aroussi")
    print(f"* FAISS {faiss.__version__} (c) Meta Platforms, Inc.")
    print("---------------------------------------------")
    print("\nConfiguration:\n")
    print(f" - Storage: {storage_mode}")
    print(f" - Authentication: {auth_status}")
    print(f" - Bind address: {bind_address}:{port}")
    print(f" - Log level: {log_level}")
    print("\n---------------------------------------------")

    # Log detailed startup information for debugging
    # Commented out to reduce verbosity, but useful for troubleshooting
    # logger.info("Starting FAISSx ZeroMQ server")
    # logger.info(f"Server version: {faissx_version}")
    # logger.info(f"Binding to {bind_address}:{port}")
    # logger.info(f"Data directory: {data_dir or 'None (in-memory only)'}")
    # logger.info(f"Authentication: {'Enabled' if enable_auth else 'Disabled'}")

    # Validate authentication configuration
    if enable_auth and not auth_keys:
        logger.warning("Authentication enabled but no keys provided")

    # Create the main server instance
    # This initializes all index storage and background workers
    server = FaissIndex(data_dir=data_dir)

    # Register specialized operation handlers for advanced features
    # This extends the server with optional advanced FAISS operations
    register_specialized_handlers(server)

    try:
        # Initialize ZeroMQ context and socket
        # Context manages ZeroMQ resources and should be created once
        context = zmq.Context()
        socket = context.socket(zmq.REP)  # REP = Reply socket for sync communication

        # Configure socket options for production stability
        # These settings optimize performance and reliability
        socket.setsockopt(zmq.LINGER, linger)                    # Graceful shutdown time
        socket.setsockopt(zmq.RCVTIMEO, socket_timeout)          # Receive timeout
        socket.setsockopt(zmq.SNDTIMEO, socket_timeout)          # Send timeout
        socket.setsockopt(zmq.RCVHWM, high_water_mark)          # Receive buffer limit
        socket.setsockopt(zmq.SNDHWM, high_water_mark)          # Send buffer limit

        # Bind the socket to the specified address and port
        try:
            socket.bind(f"tcp://{bind_address}:{port}")
        except zmq.error.ZMQError as e:
            if "Address already in use" in str(e):
                # Handle the common case of port conflicts gracefully
                logger.error(f"Port {port} is already in use")
                print(f"\nERROR: Port {port} is already in use!")
                print("This likely means another instance of the FAISSx server is already running.")
                print("\nPossible solutions:")
                print(f" 1. Stop the existing server instance using port {port}")
                print(" 2. Specify a different port using --port option")
                print(" 3. Check if any other application is using this port")
                print("\nYou can use the following command to see processes using this port:")
                print(f"    lsof -i :{port}")
                print("\nExiting...")
                print("\n---------------------------------------------\n")
                return  # Exit gracefully instead of raising the exception
            else:
                # Handle other binding errors (permissions, invalid address, etc.)
                print(f"\nZMQ ERROR during socket binding: {e}")
                error_details = (
                    f"Error details: {str(e)}, type: {type(e).__name__}, "
                    f"code: {e.errno if hasattr(e, 'errno') else 'N/A'}"
                )
                print(error_details)
                print("\n---------------------------------------------\n")
                logger.error(f"Failed to bind socket: {e}")
                raise  # Re-raise other types of exceptions

        # Create a worker thread pool for handling requests
        # This provides timeout protection for long-running operations
        task_worker = TaskWorker()

        # Display successful startup message
        print(f"Started. Listening on {bind_address}:{port}...")
        print("---------------------------------------------\n")

        # Main server loop - processes requests until shutdown
        while True:
            try:
                # Wait for a message from a client
                logger.info("Waiting for a message...")
                try:
                    # Blocking receive with timeout protection
                    message = socket.recv()
                    logger.info(f"Received message of length {len(message)}")
                except zmq.error.Again:
                    # Socket timeout - this is normal for long polling scenarios
                    logger.debug(
                        "Socket timeout while waiting for a message "
                        "(normal for long polling)"
                    )
                    continue
                except zmq.error.ZMQError as e:
                    # Handle other ZeroMQ errors during message reception
                    logger.error(f"Error receiving message: {e}")
                    continue

                # Deserialize the incoming request
                # This converts binary MessagePack data to Python objects
                request = msgpack.unpackb(message, raw=False)

                # Extract the requested operation type
                action = request.get("action", "")

                logger.debug(f"Received request: {action}")
                if action == "create_index":
                    logger.debug(f"Create index request details: {json.dumps(request)}")

                # Authentication check (skip for ping to allow health checks)
                # Ping operations should work without auth for monitoring systems
                if enable_auth and action != "ping":
                    is_authenticated, auth_error = authenticate_request(request, auth_keys)
                    if not is_authenticated:
                        logger.warning(f"Authentication failed for action '{action}': {auth_error}")
                        response = error_response(auth_error, code="AUTH_ERROR")
                        socket.send(msgpack.packb(response, use_bin_type=True))
                        continue

                # Initialize default response for unknown actions
                response = {"success": False, "error": "Unknown action"}

                # Process request based on action type
                # Each action maps to a specific server method
                if action == "create_index":
                    # Create a new FAISS index with specified parameters
                    response = server.create_index(
                        request.get("index_id", ""),
                        request.get("dimension", 0),
                        request.get("index_type", "L2"),
                        request.get("metadata", None)
                    )

                elif action == "add_vectors":
                    response = server.add_vectors(
                        request.get("index_id", ""),
                        request.get("vectors", []),
                        request.get("ids", None)
                    )

                elif action == "search":
                    # Use task worker for search operations to prevent timeouts
                    task_id = task_worker.submit_task(
                        server.search,
                        request.get("index_id", ""),
                        request.get("query_vectors", []),
                        request.get("k", 10),
                        request.get("params", None)
                    )
                    try:
                        result = task_worker.wait_for_result(task_id)
                        if result["success"]:
                            response = result["result"]
                        else:
                            response = error_response(result["error"])
                    except RequestTimeoutError:
                        response = error_response("Search operation timed out", code="TIMEOUT")

                elif action == "train_index":
                    response = server.train_index(
                        request.get("index_id", ""),
                        request.get("training_vectors", [])
                    )

                elif action == "get_index_status":
                    response = server.get_index_status(
                        request.get("index_id", "")
                    )

                elif action == "get_index_info":
                    response = server.get_index_info(
                        request.get("index_id", "")
                    )

                elif action == "list_indices":
                    response = server.list_indices()

                elif action == "delete_index":
                    response = server.delete_index(
                        request.get("index_id", "")
                    )

                elif action == "reset":
                    response = server.reset(
                        request.get("index_id", "")
                    )

                elif action == "clear":
                    response = server.clear(
                        request.get("index_id", "")
                    )

                elif action == "set_parameter":
                    response = server.set_parameter(
                        request.get("index_id", ""),
                        request.get("parameter", ""),
                        request.get("value", None)
                    )

                elif action == "get_parameter":
                    response = server.get_parameter(
                        request.get("index_id", ""),
                        request.get("parameter", "")
                    )

                elif action == "reconstruct":
                    response = server.reconstruct(
                        request.get("index_id", ""),
                        request.get("idx", 0)
                    )

                elif action == "reconstruct_n":
                    response = server.reconstruct_n(
                        request.get("index_id", ""),
                        request.get("start_idx", 0),
                        request.get("num_vectors", 10)
                    )

                elif action == "range_search":
                    # Use task worker for range search operations to prevent timeouts
                    task_id = task_worker.submit_task(
                        server.range_search,
                        request.get("index_id", ""),
                        request.get("query_vectors", []),
                        request.get("radius", 0.0)
                    )
                    try:
                        result = task_worker.wait_for_result(task_id)
                        if result["success"]:
                            response = result["result"]
                        else:
                            response = error_response(result["error"])
                    except RequestTimeoutError:
                        response = error_response(
                            "Range search operation timed out",
                            code="TIMEOUT",
                        )

                elif action == "merge_indices":
                    response = server.merge_indices(
                        request.get("target_index_id", ""),
                        request.get("source_index_ids", [])
                    )

                elif action == "ping":
                    response = {"success": True, "message": "pong", "time": time.time()}

                elif action == "get_vectors":
                    # Use task worker for get_vectors operations to prevent timeouts
                    task_id = task_worker.submit_task(
                        server.get_vectors,
                        request.get("index_id", ""),
                        request.get("start_idx", 0),
                        request.get("limit", None)
                    )
                    try:
                        result = task_worker.wait_for_result(task_id)
                        if result["success"]:
                            response = result["result"]
                        else:
                            response = error_response(result["error"])
                    except RequestTimeoutError:
                        response = error_response("Get vectors operation timed out", code="TIMEOUT")

                elif action == "search_and_reconstruct":
                    # Use task worker for search_and_reconstruct operations to prevent timeouts
                    task_id = task_worker.submit_task(
                        server.search_and_reconstruct,
                        request.get("index_id", ""),
                        request.get("query_vectors", []),
                        request.get("k", 10),
                        request.get("params", None)
                    )
                    try:
                        result = task_worker.wait_for_result(task_id)
                        if result["success"]:
                            response = result["result"]
                        else:
                            response = error_response(result["error"])
                    except RequestTimeoutError:
                        response = error_response(
                            "Search and reconstruct operation timed out",
                            code="TIMEOUT"
                        )

                elif action == "apply_transform":
                    # Use task worker for transformation operations to prevent timeouts
                    task_id = task_worker.submit_task(
                        server.apply_transform,
                        request.get("index_id", ""),
                        request.get("vectors", [])
                    )
                    try:
                        result = task_worker.wait_for_result(task_id)
                        if result["success"]:
                            response = result["result"]
                        else:
                            response = error_response(result["error"])
                    except RequestTimeoutError:
                        response = error_response(
                            "Apply transform operation timed out",
                            code="TIMEOUT"
                        )

                elif action == "get_transform_info":
                    response = server.get_transform_info(
                        request.get("index_id", "")
                    )

                elif action == "add_with_ids":
                    # Process add_with_ids request
                    logger.info(
                        f"Processing add_with_ids request for index {request.get('index_id', '')}")
                    response = server.add_with_ids(
                        request.get("index_id", ""),
                        request.get("vectors", []),
                        request.get("ids", [])
                    )

                # Check for specialized operations in action_handlers dictionary
                elif hasattr(server, "action_handlers") and action in server.action_handlers:
                    try:
                        # Call the specialized operation handler
                        handler_func = server.action_handlers[action]
                        # Pass the server instance as first argument
                        response = handler_func(
                            server,
                            **{k: v for k, v in request.items() if k != "action"},
                        )
                    except Exception as e:
                        logger.exception(f"Error in specialized operation handler: {e}")
                        response = error_response(f"Server error: {str(e)}")

                # Add version and timestamp to all responses for debugging
                # This helps with troubleshooting and request tracking
                if "version" not in response:
                    response["version"] = faissx_version

                if "timestamp" not in response:
                    response["timestamp"] = time.time()

                # Send the response back to the client
                # Serialize to binary format for efficient transmission
                socket.send(msgpack.packb(response, use_bin_type=True))

            except zmq.error.Again:
                # Socket timeout - just continue and wait for the next message
                # This is normal behavior and not an error condition
                logger.debug("Socket timeout waiting for request")
                continue

            except zmq.error.ZMQError as e:
                # Handle ZeroMQ-specific errors during request processing
                logger.error(f"ZMQ error: {e}")
                # Try to send an error response if possible
                try:
                    socket.send(msgpack.packb(
                        error_response(f"Server error: {str(e)}"),
                        use_bin_type=True
                    ))
                except Exception:
                    # If we can't send error response, just log and continue
                    pass

            except Exception as e:
                # Handle any unexpected errors during request processing
                # This prevents the server from crashing on individual request errors
                logger.exception(f"Error handling request: {e}")
                # Try to send an error response if possible
                try:
                    socket.send(msgpack.packb(
                        error_response(f"Server error: {str(e)}"),
                        use_bin_type=True
                    ))
                except Exception:
                    # If we can't send error response, just log and continue
                    pass

    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        logger.info("Server shutting down")
    except Exception as e:
        # Handle any critical server errors
        logger.exception(f"Server error: {e}")
    finally:
        # Cleanup ZeroMQ resources on shutdown
        # This ensures proper resource cleanup even if errors occur
        if 'socket' in locals():
            socket.close()
        if 'context' in locals():
            context.term()


def is_transform_trained(transform):
    """
    Check if a transformation component is properly trained.

    This utility function determines whether a FAISS transformation object
    has been properly trained and is ready for use. Different transform types
    have different training requirements.

    Args:
        transform: FAISS transform object (PCAMatrix, NormalizationTransform, etc.)

    Returns:
        bool: True if the transform is trained or doesn't require training,
              False if training is required but not completed

    Implementation Notes:
        - Some transforms (like NormalizationTransform) don't require training
        - Others (like PCAMatrix) must be trained before use
        - If is_trained attribute doesn't exist, assumes no training needed
        - Used by IndexPreTransform validation logic
    """
    # If it doesn't have is_trained attribute, assume it's always trained
    # This handles transforms that don't require explicit training
    if not hasattr(transform, "is_trained"):
        return True

    # Return the actual training status
    return transform.is_trained


def train_transform(transform, training_vectors):
    """
    Train a transformation component with provided vectors.

    This function handles the training process for FAISS transformation objects
    that require training data (like PCA, OPQ, etc.). It provides error handling
    and logging for the training process.

    Args:
        transform: FAISS transform object that needs training
        training_vectors: NumPy array of vectors to use for training

    Returns:
        bool: True if training was successful, False if it failed or
              the transform doesn't support training

    Training Process:
        1. Check if transform has a train() method
        2. Call train() with the provided vectors
        3. Handle any training errors gracefully
        4. Log training progress and results

    Error Handling:
        - Training failures are logged but don't crash the server
        - Returns False for transforms that don't support training
        - Captures and logs all training exceptions for debugging
    """
    try:
        # Check if the transform supports training
        if hasattr(transform, "train"):
            # Perform the training with provided vectors
            # This may take significant time for large datasets
            transform.train(training_vectors)
            return True
        # Transform doesn't support training - this is okay
        return False
    except Exception as e:
        # Handle any training errors gracefully
        # Log the error but don't crash the server
        logger.error(f"Error training transform: {e}")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FAISSx Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--bind-address", default=DEFAULT_BIND_ADDRESS, help="Address to bind to")
    parser.add_argument("--data-dir", help="Directory for persistent storage")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: WARNING)")
    args = parser.parse_args()

    # Use environment variables as fallback, but prioritize command-line arguments
    port = int(os.environ.get("FAISSX_PORT", args.port))
    bind_address = os.environ.get("FAISSX_BIND_ADDRESS", args.bind_address)
    data_dir = args.data_dir or os.environ.get("FAISSX_DATA_DIR")
    log_level = os.environ.get("FAISSX_LOG_LEVEL", args.log_level)

    # Default to no authentication when run directly
    run_server(
        port,
        bind_address,
        auth_keys={},
        enable_auth=False,
        data_dir=data_dir,
        log_level=log_level,
    )

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
import zmq
import numpy as np
import faiss
import msgpack
import argparse
import logging
import threading
from queue import Queue
from typing import Dict, List, Any, Tuple, Optional, Union
from faissx import __version__ as faissx_version
from faissx.server.response import (
    success_response, error_response, format_search_results,
    format_vector_results, format_index_status
)
from faissx.server.training import (
    requires_training, get_training_requirements,
    is_trained_for_use, estimate_training_vectors_needed
)
from faissx.server.binary import (
    is_binary_index_type, create_binary_index, convert_to_binary,
    binary_to_float
)

# Import our modules
from . import persistence
from . import hnsw
from . import hybrid
from . import transformations
from .transformations import (
    parse_transform_type, create_transformation, create_pretransform_index,
    is_transform_trained, get_transform_training_requirements, train_transform
)

# Constants for server configuration
DEFAULT_PORT = 45678  # Default port for ZeroMQ server
DEFAULT_BIND_ADDRESS = "0.0.0.0"  # Default bind address (all interfaces)
DEFAULT_SOCKET_TIMEOUT = 60000  # Socket timeout in milliseconds (60 seconds)
DEFAULT_HIGH_WATER_MARK = 1000  # High water mark for socket buffer
DEFAULT_LINGER = 1000  # Linger time for socket (1 second)
DEFAULT_OPERATION_TIMEOUT = 30  # Default timeout for operations in seconds

# Configure logging
logger = logging.getLogger("faissx.server")


class RequestTimeoutError(Exception):
    """Exception raised when a request takes too long to process."""
    pass


class TaskWorker:
    """Worker to handle long-running tasks asynchronously."""

    def __init__(self, timeout=DEFAULT_OPERATION_TIMEOUT):
        """
        Initialize a worker for handling long-running tasks.

        Args:
            timeout (int): Default timeout in seconds for task completion
        """
        self.timeout = timeout
        self.queue = Queue()
        self.results = {}
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Background worker loop to process tasks."""
        while True:
            try:
                task_id, func, args, kwargs = self.queue.get(block=True)
                try:
                    result = func(*args, **kwargs)
                    self.results[task_id] = {"success": True, "result": result}
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {str(e)}")
                    self.results[task_id] = {"success": False, "error": str(e)}
                finally:
                    self.queue.task_done()
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")

    def submit_task(self, func, *args, **kwargs):
        """
        Submit a task to be executed in the background.

        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            str: Task ID that can be used to retrieve results
        """
        task_id = str(time.time())
        self.queue.put((task_id, func, args, kwargs))
        return task_id

    def get_result(self, task_id):
        """
        Get the result of a task without waiting.

        Args:
            task_id (str): ID of the task

        Returns:
            dict: Result dictionary or None if not available
        """
        return self.results.get(task_id)

    def wait_for_result(self, task_id, timeout=None):
        """
        Wait for a task result with timeout.

        Args:
            task_id (str): ID of the task
            timeout (int, optional): Timeout in seconds

        Returns:
            dict: Result dictionary

        Raises:
            RequestTimeoutError: If the task doesn't complete within the timeout
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            if task_id in self.results:
                result = self.results[task_id]
                del self.results[task_id]  # Clean up
                return result
            time.sleep(0.1)

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
        logger.info("FAISSx server initialized (version %s)", faissx_version)

    def _run_with_timeout(self, func, *args, timeout=None, **kwargs):
        """
        Run a function with a timeout.

        Args:
            func: Function to run
            *args: Arguments to pass to the function
            timeout: Timeout in seconds
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function or error response
        """
        try:
            task_id = self.task_worker.submit_task(func, *args, **kwargs)
            result = self.task_worker.wait_for_result(task_id, timeout)

            if not result["success"]:
                return error_response(result["error"])

            return result["result"]
        except RequestTimeoutError as e:
            logger.error(f"Timeout error: {str(e)}")
            return error_response(f"Operation timed out", code="TIMEOUT")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return error_response(f"Unexpected error: {str(e)}")

    def search(self, index_id, query_vectors, k=10, params=None):
        """
        Search for similar vectors in an index.

        Args:
            index_id (str): ID of the target index
            query_vectors (list): List of query vectors
            k (int): Number of nearest neighbors to return
            params (dict, optional): Additional search parameters like nprobe for IVF indices

        Returns:
            dict: Response containing search results or error message
        """
        return self._run_with_timeout(
            self._search, index_id, query_vectors, k, params
        )

    def _search(self, index_id, query_vectors, k=10, params=None):
        """Internal implementation of search that can be run with timeout."""
        if index_id not in self.indexes:
            return error_response(f"Index {index_id} does not exist")

        try:
            index = self.indexes[index_id]

            # Check if this is a binary index
            is_binary = False
            for binary_class in [faiss.IndexBinaryFlat, faiss.IndexBinaryIVF, faiss.IndexBinaryHash]:
                if isinstance(index, binary_class):
                    is_binary = True
                    break

            # Check if the index is ready for use
            is_ready, reason = is_trained_for_use(index)
            if not is_ready:
                return error_response(
                    f"Index is not ready: {reason}",
                    code="NOT_TRAINED"
                )

            if is_binary:
                try:
                    # Convert query vectors to binary format
                    query_binary = convert_to_binary(query_vectors)

                    # Print debug info
                    print(f"Binary query shape: {query_binary.shape}, dtype: {query_binary.dtype}")

                    # Perform the search
                    distances, indices = index.search(query_binary, k)

                    # Convert numpy arrays to lists for serialization
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
                            "is_binary": True
                        }
                    )
                except Exception as e:
                    print(f"Error in binary search: {str(e)}")
                    return error_response(f"Error in binary search: {str(e)}")
            else:
                # Standard float vector search
                # Convert query vectors to numpy array and validate dimensions
                query_np = np.array(query_vectors, dtype=np.float32)
                if query_np.shape[1] != self.dimensions[index_id]:
                    return error_response(
                        f"Query dimension mismatch: expected {self.dimensions[index_id]}, "
                        f"got {query_np.shape[1]}"
                    )

                # Apply search parameters if provided
                if params:
                    for param_name, param_value in params.items():
                        if hasattr(index, "set_" + param_name):
                            getattr(index, "set_" + param_name)(param_value)

                # Special handling for IndexPreTransform is not needed here,
                # as the transformation is applied automatically during search

                # Perform the search
                distances, indices = index.search(query_np, k)

                # Convert numpy arrays to lists for serialization
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
            return error_response(f"Error searching index: {str(e)}")

    def get_vectors(self, index_id, start_idx=0, limit=None):
        """
        Retrieve all vectors in an index or a specified range of vectors.

        This method allows efficient retrieval of vectors stored in an index,
        with optional pagination using start_idx and limit parameters.

        Args:
            index_id (str): ID of the index
            start_idx (int, optional): Starting index for retrieval (default: 0)
            limit (int, optional): Maximum number of vectors to retrieve (default: None, retrieve all)

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
            for binary_class in [faiss.IndexBinaryFlat, faiss.IndexBinaryIVF, faiss.IndexBinaryHash]:
                if isinstance(index, binary_class):
                    is_binary = True
                    break

            if is_binary:
                # Convert vectors to binary format
                try:
                    binary_vectors = convert_to_binary(vectors)

                    # Print debug info
                    print(f"Binary vectors shape: {binary_vectors.shape}, dtype: {binary_vectors.dtype}")

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
                    print(f"Error in binary vector conversion: {str(e)}")
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
                    "ntotal": index.ntotal
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
        if index_id in self.indexes:
            return error_response(f"Index {index_id} already exists")

        try:
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

                    return success_response(index_info, message=f"Binary index {index_id} created successfully")
                except Exception as e:
                    return error_response(f"Error creating binary index: {str(e)}")

            # Check for pre-transform index types
            transform_type, base_index_type, transform_params = parse_transform_type(index_type)
            if transform_type is not None:
                try:
                    # First create the transformation
                    output_dim = transform_params.get("output_dim", dimension)
                    transform, transform_info = create_transformation(
                        transform_type,
                        dimension,
                        output_dim,
                        **transform_params
                    )

                    # Create the base index using the output dimension from the transform
                    base_index_id = f"{index_id}_base"
                    base_response = self.create_index(
                        base_index_id,
                        output_dim,
                        base_index_type,
                        metadata={"is_base_index": True, "parent_index": index_id}
                    )

                    if not base_response["success"]:
                        return base_response

                    # Get the base index
                    base_index = self.indexes[base_index_id]
                    self.base_indexes[index_id] = base_index_id

                    # Create the pretransform index
                    pretransform_index, index_info = create_pretransform_index(
                        base_index, transform, transform_info
                    )

                    # Store the index
                    self.indexes[index_id] = pretransform_index
                    self.dimensions[index_id] = dimension

                    # Add metadata if provided
                    if metadata:
                        index_info["metadata"] = metadata

                    return success_response(
                        index_info,
                        message=f"Transformed index {index_id} created successfully"
                    )

                except Exception as e:
                    # Clean up base index if created
                    if f"{index_id}_base" in self.indexes:
                        del self.indexes[f"{index_id}_base"]
                        del self.dimensions[f"{index_id}_base"]
                        if index_id in self.base_indexes:
                            del self.base_indexes[index_id]

                    return error_response(f"Error creating transformed index: {str(e)}")

            # Handle standard FAISS index types
            elif index_type == "L2":
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
                index = faiss.IndexPQ(dimension, 8, 8)  # 8 subquantizers with 8 bits each by default
            elif index_type == "PQ_IP":
                index = faiss.IndexPQ(dimension, 8, 8, faiss.METRIC_INNER_PRODUCT)
            elif index_type.startswith("IDMap:"):
                base_type = index_type[6:]
                base_index = self.create_index(f"{index_id}_base", dimension, base_type)
                if not base_index["success"]:
                    return base_index

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

            return success_response(index_details, message=f"Index {index_id} created successfully")

        except Exception as e:
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
                return {
                    "success": False,
                    "error": f"Number of vectors ({len(vectors_np)}) doesn't match number of IDs ({len(ids_np)})"
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

            print(f"Added {len(vectors)} vectors with IDs to index {index_id}, total: {total}")
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

            print(f"Removed {removed_count} vectors from index {index_id}, remaining: {after_count}")
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
            for binary_class in [faiss.IndexBinaryFlat, faiss.IndexBinaryIVF, faiss.IndexBinaryHash]:
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
                    print(f"Reconstructed binary vector: shape={binary_vector.shape}, dtype={binary_vector.dtype}")

                    # Convert to numpy array with correct shape for binary_to_float
                    binary_vector = binary_vector.reshape(1, -1)

                    # Convert binary vector to float vector for response
                    vector = binary_to_float(binary_vector, dimension)[0]

                    return success_response({"vector": vector})
                except Exception as e:
                    print(f"Error in binary reconstruction: {str(e)}")
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
                    "error": f"Range {start_idx}:{start_idx+num_vectors} exceeds index size {index.ntotal}"
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
            "vectors_count": status["ntotal"],
            "dimension": status["dimension"],
            # Assuming float32 (4 bytes)
            "byte_size_per_vector": status["dimension"] * 4,
        })

        # Add estimated memory usage (approximate)
        vector_memory = status["ntotal"] * status["dimension"] * 4  # float32 vectors
        index_overhead = 0

        # Estimate index overhead based on type
        if isinstance(index, faiss.IndexFlat):
            # Flat indices store vectors directly with minimal overhead
            index_overhead = status["ntotal"] * 4  # Minimal overhead
        elif isinstance(index, faiss.IndexIVF):
            # IVF has inverted lists overhead
            index_overhead = status["nlist"] * 100 + status["ntotal"] * 8
        elif isinstance(index, faiss.IndexHNSW):
            # HNSW has graph structure overhead
            index_overhead = status["ntotal"] * status["hnsw_m"] * 8

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
                "nlist": status["nlist"],
                "nprobe": status["nprobe"],
                "metric": "L2" if index.metric_type == faiss.METRIC_L2 else "IP",
            })
        elif isinstance(index, faiss.IndexHNSW):
            info["configuration"].update({
                "M": status["hnsw_m"],
                "efSearch": status["ef_search"],
                "efConstruction": status["ef_construction"],
                "metric": "L2" if index.metric_type == faiss.METRIC_L2 else "IP",
            })
        elif isinstance(index, faiss.IndexPQ):
            info["configuration"].update({
                "M": status["pq_m"],
                "nbits": status["pq_nbits"],
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

        - For IVF indices: nprobe
        - For HNSW indices: efSearch
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
            # Parameter handlers for different index types
            if param_name == "nprobe" and isinstance(index, faiss.IndexIVF):
                # Validate nprobe value
                if not isinstance(param_value, int) or param_value <= 0:
                    return {
                        "success": False,
                        "error": "nprobe must be a positive integer"
                    }

                # Set nprobe parameter
                index.nprobe = param_value
                return {
                    "success": True,
                    "message": f"Set nprobe={param_value} for index {index_id}"
                }

            elif param_name == "efSearch" and isinstance(index, faiss.IndexHNSW):
                # Validate efSearch value
                if not isinstance(param_value, int) or param_value <= 0:
                    return {
                        "success": False,
                        "error": "efSearch must be a positive integer"
                    }

                # Set efSearch parameter
                index.hnsw.efSearch = param_value
                return {
                    "success": True,
                    "message": f"Set efSearch={param_value} for index {index_id}"
                }

            elif param_name == "efConstruction" and isinstance(index, faiss.IndexHNSW):
                # Validate efConstruction value
                if not isinstance(param_value, int) or param_value <= 0:
                    return {
                        "success": False,
                        "error": "efConstruction must be a positive integer"
                    }

                # Set efConstruction parameter
                index.hnsw.efConstruction = param_value
                return {
                    "success": True,
                    "message": f"Set efConstruction={param_value} for index {index_id}"
                }

            else:
                return {
                    "success": False,
                    "error": f"Parameter {param_name} not supported for this index type"
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
                        output_dim = transform.d_out if hasattr(transform, "d_out") else vectors_np.shape[1]
                        transformed_vectors = np.zeros((len(vectors_np), output_dim), dtype=np.float32)
                        for i, vec in enumerate(vectors_np):
                            transform.apply(1, faiss.swig_ptr(vec), faiss.swig_ptr(transformed_vectors[i]))

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
                recommendation = f"For optimal results, consider using at least {recommended_vectors} training vectors"

            # Train the index
            index.train(vectors_np)

            # Get updated training requirements after training
            training_info = get_training_requirements(index)

            print(f"Trained index {index_id} with {len(training_vectors)} vectors")
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
                details={"index_id": index_id, "vector_count": len(training_vectors) if 'training_vectors' in locals() else 0}
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
                        new_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits, metric_type)
                    elif isinstance(index, faiss.IndexIVFScalarQuantizer):
                        # Get scalar quantizer parameters
                        sq_type = index.sq_type
                        new_index = faiss.IndexIVFScalarQuantizer(quantizer, dimension, nlist, sq_type, metric_type)
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

        All indices must be of compatible types and dimensions.

        Args:
            target_index_id (str): ID of the target index to merge into
            source_index_ids (list): List of source index IDs to merge from

        Returns:
            dict: Response indicating success or failure
        """
        if not source_index_ids:
            return {
                "success": False,
                "error": "No source indices provided for merging"
            }

        # Check if the target index exists
        if target_index_id not in self.indexes:
            return {
                "success": False,
                "error": f"Target index {target_index_id} not found"
            }

        # Check if all source indices exist
        for src_id in source_index_ids:
            if src_id not in self.indexes:
                return {
                    "success": False,
                    "error": f"Source index {src_id} not found"
                }

        target_index = self.indexes[target_index_id]
        target_dim = self.dimensions[target_index_id]

        try:
            # Process each source index
            for src_id in source_index_ids:
                src_index = self.indexes[src_id]
                src_dim = self.dimensions[src_id]

                # Check dimension compatibility
                if src_dim != target_dim:
                    return {
                        "success": False,
                        "error": (
                            f"Dimension mismatch: target dimension is {target_dim}, "
                            f"but source index {src_id} has dimension {src_dim}"
                        )
                    }

                # Check index type compatibility
                if type(src_index) != type(target_index):
                    return {
                        "success": False,
                        "error": (
                            f"Index type mismatch: target is {type(target_index).__name__}, "
                            f"but source index {src_id} is {type(src_index).__name__}"
                        )
                    }

                # Additional compatibility checks for specific index types
                if isinstance(target_index, faiss.IndexIVF):
                    # IVF indices must have compatible quantizers
                    if target_index.nlist != src_index.nlist:
                        return {
                            "success": False,
                            "error": (
                                f"IVF nlist mismatch: target nlist is {target_index.nlist}, "
                                f"but source index {src_id} has nlist {src_index.nlist}"
                            )
                        }

                # Extract vectors from source index
                if src_index.ntotal > 0:
                    vectors = np.zeros((src_index.ntotal, src_dim), dtype=np.float32)
                    for i in range(src_index.ntotal):
                        vectors[i] = src_index.reconstruct(i)

                    # Add vectors to target index
                    target_index.add(vectors)

            return {
                "success": True,
                "message": (
                    f"Successfully merged {len(source_index_ids)} indices into {target_index_id}, "
                    f"new total: {target_index.ntotal} vectors"
                )
            }

        except Exception as e:
            return {"success": False, "error": f"Error merging indices: {str(e)}"}

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
                    # Range search returns distances and indices as lists
                    distances, indices = index.range_search(query.reshape(1, -1), radius)

                    # Format result for this query
                    result = {
                        "query_index": i,
                        "num_results": len(indices),
                        "indices": indices.tolist() if isinstance(indices, np.ndarray) else indices,
                        "distances": distances.tolist() if isinstance(distances, np.ndarray) else distances
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


def serialize_message(data):
    """
    Serialize a message to binary format using msgpack.

    Args:
        data (dict): Data to serialize

    Returns:
        bytes: Serialized binary data
    """
    if isinstance(data, dict) and "results" in data and data.get("success", False):
        # Special handling for search results with numpy arrays
        return msgpack.packb(data, use_bin_type=True)
    else:
        # Regular JSON-serializable data
        return msgpack.packb(data, use_bin_type=True)


def deserialize_message(data):
    """
    Deserialize a binary message using msgpack.

    Args:
        data (bytes): Binary data to deserialize

    Returns:
        dict: Deserialized data or error message
    """
    try:
        return msgpack.unpackb(data, raw=False)
    except Exception as e:
        return {"success": False, "error": f"Failed to deserialize message: {str(e)}"}


def authenticate_request(request, auth_keys):
    """
    Authenticate a request using API keys.

    Args:
        request (dict): Request data containing API key
        auth_keys (dict): Dictionary mapping API keys to tenant IDs

    Returns:
        tuple: (is_authenticated, error_message)
    """
    if not auth_keys:
        # Authentication disabled
        return True, None

    api_key = request.get("api_key")
    if not api_key:
        return False, "API key required"

    tenant_id = auth_keys.get(api_key)
    if not tenant_id:
        return False, "Invalid API key"

    # Add tenant_id to the request
    request["tenant_id"] = tenant_id
    return True, None


def run_server(
    port=DEFAULT_PORT,
    bind_address=DEFAULT_BIND_ADDRESS,
    auth_keys=None,
    enable_auth=False,
    data_dir=None,
    socket_timeout=DEFAULT_SOCKET_TIMEOUT,
    high_water_mark=DEFAULT_HIGH_WATER_MARK,
    linger=DEFAULT_LINGER,
):
    """
    Run the ZeroMQ server for handling vector operations.

    Args:
        port (int): Port number for the server
        bind_address (str): Address to bind the server to
        auth_keys (dict): Dictionary of API keys for authentication
        enable_auth (bool): Whether to enable authentication
        data_dir (str): Directory for persistent storage
        socket_timeout (int): Socket timeout in milliseconds
        high_water_mark (int): High water mark for socket buffer
        linger (int): Linger time in milliseconds
    """
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)

        # Configure socket options for better stability
        socket.setsockopt(zmq.LINGER, linger)
        socket.setsockopt(zmq.RCVTIMEO, socket_timeout)
        socket.setsockopt(zmq.SNDTIMEO, socket_timeout)
        socket.setsockopt(zmq.RCVHWM, high_water_mark)
        socket.setsockopt(zmq.SNDHWM, high_water_mark)

        # Bind the socket
        socket.bind(f"tcp://{bind_address}:{port}")

        # Initialize the index server
        faiss_index = FaissIndex(data_dir=data_dir)

        # Create a worker thread pool for handling requests
        task_worker = TaskWorker()

        logger.info(f"FAISSx server running on port {port}")

        while True:
            try:
                # Wait for a message
                message = socket.recv()

                # Unpack the message
                request = msgpack.unpackb(message, raw=False)

                # Check the action
                action = request.get("action", "")

                logger.debug(f"Received request: {action}")

                # Process the request
                response = {"success": False, "error": "Unknown action"}

                # Process request based on action
                if action == "create_index":
                    response = faiss_index.create_index(
                        request.get("index_id", ""),
                        request.get("dimension", 0),
                        request.get("index_type", "L2"),
                        request.get("metadata", None)
                    )

                elif action == "add_vectors":
                    response = faiss_index.add_vectors(
                        request.get("index_id", ""),
                        request.get("vectors", []),
                        request.get("ids", None)
                    )

                elif action == "search":
                    # Use task worker for search operations to prevent timeouts
                    task_id = task_worker.submit_task(
                        faiss_index.search,
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
                    response = faiss_index.train_index(
                        request.get("index_id", ""),
                        request.get("training_vectors", [])
                    )

                elif action == "get_index_status":
                    response = faiss_index.get_index_status(
                        request.get("index_id", "")
                    )

                elif action == "get_index_info":
                    response = faiss_index.get_index_info(
                        request.get("index_id", "")
                    )

                elif action == "list_indices":
                    response = faiss_index.list_indices()

                elif action == "delete_index":
                    response = faiss_index.delete_index(
                        request.get("index_id", "")
                    )

                elif action == "reset":
                    response = faiss_index.reset(
                        request.get("index_id", "")
                    )

                elif action == "clear":
                    response = faiss_index.clear(
                        request.get("index_id", "")
                    )

                elif action == "set_parameter":
                    response = faiss_index.set_parameter(
                        request.get("index_id", ""),
                        request.get("parameter", ""),
                        request.get("value", None)
                    )

                elif action == "get_parameter":
                    response = faiss_index.get_parameter(
                        request.get("index_id", ""),
                        request.get("parameter", "")
                    )

                elif action == "reconstruct":
                    response = faiss_index.reconstruct(
                        request.get("index_id", ""),
                        request.get("idx", 0)
                    )

                elif action == "reconstruct_n":
                    response = faiss_index.reconstruct_n(
                        request.get("index_id", ""),
                        request.get("start_idx", 0),
                        request.get("num_vectors", 10)
                    )

                elif action == "range_search":
                    # Use task worker for range search operations to prevent timeouts
                    task_id = task_worker.submit_task(
                        faiss_index.range_search,
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
                        response = error_response("Range search operation timed out", code="TIMEOUT")

                elif action == "merge_indices":
                    response = faiss_index.merge_indices(
                        request.get("target_index_id", ""),
                        request.get("source_index_ids", [])
                    )

                elif action == "ping":
                    response = {"success": True, "message": "pong", "time": time.time()}

                elif action == "get_vectors":
                    # Use task worker for get_vectors operations to prevent timeouts
                    task_id = task_worker.submit_task(
                        faiss_index.get_vectors,
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
                        faiss_index.search_and_reconstruct,
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
                        faiss_index.apply_transform,
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
                    response = faiss_index.get_transform_info(
                        request.get("index_id", "")
                    )

                # Add version information to all responses
                if "version" not in response:
                    response["version"] = faissx_version

                # Add timestamp to all responses
                if "timestamp" not in response:
                    response["timestamp"] = time.time()

                # Send the response
                socket.send(msgpack.packb(response, use_bin_type=True))

            except zmq.error.Again:
                # Socket timeout - just continue and wait for the next message
                logger.debug("Socket timeout waiting for request")
                continue

            except zmq.error.ZMQError as e:
                logger.error(f"ZMQ error: {e}")
                # Try to send an error response if possible
                try:
                    socket.send(msgpack.packb(
                        error_response(f"Server error: {str(e)}"),
                        use_bin_type=True
                    ))
                except:
                    pass

            except Exception as e:
                logger.exception(f"Error handling request: {e}")
                # Try to send an error response if possible
                try:
                    socket.send(msgpack.packb(
                        error_response(f"Server error: {str(e)}"),
                        use_bin_type=True
                    ))
                except:
                    pass

    except KeyboardInterrupt:
        logger.info("Server shutting down")
    except Exception as e:
        logger.exception(f"Server error: {e}")
    finally:
        # Clean up
        if 'socket' in locals():
            socket.close()
        if 'context' in locals():
            context.term()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FAISSx Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--bind-address", default=DEFAULT_BIND_ADDRESS, help="Address to bind to")
    parser.add_argument("--data-dir", help="Directory for persistent storage")
    args = parser.parse_args()

    # Use environment variables as fallback, but prioritize command-line arguments
    port = int(os.environ.get("FAISSX_PORT", args.port))
    bind_address = os.environ.get("FAISSX_BIND_ADDRESS", args.bind_address)
    data_dir = args.data_dir or os.environ.get("FAISSX_DATA_DIR")

    # Default to no authentication when run directly
    run_server(port, bind_address, auth_keys={}, enable_auth=False, data_dir=data_dir)

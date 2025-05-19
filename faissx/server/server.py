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

This module provides the core server implementation for the FAISSx vector
search service.

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
from faissx import __version__ as faissx_version

# Constants for server configuration
DEFAULT_PORT = 45678  # Default port for ZeroMQ server
DEFAULT_BIND_ADDRESS = "0.0.0.0"  # Default bind address (all interfaces)


class FaissIndex:
    """
    Manages FAISS indexes for vector storage and search.

    This class provides a high-level interface for creating and managing FAISS vector indexes,
    supporting both in-memory and persistent storage options.
    """

    def __init__(self, data_dir=None):
        """
        Initialize the FAISS index manager.

        Args:
            data_dir (str, optional): Directory to store FAISS indices.
                    If None, uses in-memory indices without persistence.
        """
        self.data_dir = data_dir
        self.indexes = {}  # Dictionary to store FAISS indexes
        self.dimensions = {}  # Dictionary to store dimensions for each index

    def create_index(self, index_id, dimension, index_type="L2"):
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
                Additional parameters can be specified:
                - "IVF100" - IVF with 100 centroids
                - "IVF100_IP" - IVF with 100 centroids and inner product distance
                - "HNSW32" - HNSW with M=32 (connections per node)
                - "HNSW32_IP" - HNSW with M=32 and inner product distance
                - "PQ8x8" - PQ with 8 subquantizers, each with 8 bits
                - "PQ8x8_IP" - PQ with 8 subquantizers, 8 bits, inner product

        Returns:
            dict: Response containing success status and index details or error message
        """
        if index_id in self.indexes:
            return {"success": False, "error": f"Index {index_id} already exists"}

        try:
            # Parse index parameters
            index_params = {}
            if "_" in index_type:
                parts = index_type.split("_")
                main_type = parts[0]

                # Handle numeric parameters in IVF
                if main_type.startswith("IVF"):
                    try:
                        nlist = int(main_type[3:])
                        index_params["nlist"] = nlist
                        main_type = "IVF"
                    except ValueError:
                        pass

                # Handle HNSW parameters
                elif main_type.startswith("HNSW"):
                    try:
                        M = int(main_type[4:])
                        index_params["M"] = M
                        main_type = "HNSW"
                    except ValueError:
                        index_params["M"] = 32  # Default M value

                # Handle PQ parameters
                elif main_type.startswith("PQ"):
                    try:
                        # Format is PQMxB where M is number of subquantizers and B is bits
                        params = main_type[2:]
                        if "x" in params:
                            M, B = params.split("x")
                            index_params["M"] = int(M)  # Number of subquantizers
                            index_params["nbits"] = int(B)  # Bits per subquantizer
                        else:
                            index_params["M"] = int(params)  # Just number of subquantizers
                            index_params["nbits"] = 8  # Default 8 bits
                        main_type = "PQ"
                    except ValueError:
                        # Default PQ parameters
                        index_params["M"] = 8  # Default 8 subquantizers
                        index_params["nbits"] = 8  # Default 8 bits

                # Check for distance metric
                if parts[-1] == "IP":
                    index_params["metric"] = "IP"
                else:
                    index_params["metric"] = "L2"
            else:
                main_type = index_type
                index_params["metric"] = "L2"

            # Create appropriate index type
            if main_type == "L2" or (main_type == index_type and index_params["metric"] == "L2"):
                index = faiss.IndexFlatL2(dimension)
            elif main_type == "IP" or (main_type == index_type and index_params["metric"] == "IP"):
                index = faiss.IndexFlatIP(dimension)
            elif main_type == "IVF":
                # Default number of centroids if not specified
                nlist = index_params.get("nlist", 100)

                # Create quantizer based on the metric
                if index_params["metric"] == "IP":
                    quantizer = faiss.IndexFlatIP(dimension)
                else:
                    quantizer = faiss.IndexFlatL2(dimension)

                # Create IVF index
                index = faiss.IndexIVFFlat(
                    quantizer, dimension, nlist,
                    faiss.METRIC_INNER_PRODUCT if index_params["metric"] == "IP"
                    else faiss.METRIC_L2
                )

                # IVF indexes need to be trained before use
                index.is_trained = False
            elif main_type == "HNSW":
                # HNSW parameters
                M = index_params.get("M", 32)  # Default connections per node

                # Create HNSW index
                if index_params["metric"] == "IP":
                    index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
                else:
                    index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2)

                # HNSW doesn't require training
                index.is_trained = True
            elif main_type == "PQ":
                # PQ parameters
                M = index_params.get("M", 8)  # Number of subquantizers
                nbits = index_params.get("nbits", 8)  # Bits per subquantizer

                # PQ dimension must be a multiple of M
                if dimension % M != 0:
                    return {
                        "success": False,
                        "error": f"PQ requires dimension ({dimension}) to be a multiple of M ({M})"
                    }

                # Create PQ index
                if index_params["metric"] == "IP":
                    index = faiss.IndexPQ(dimension, M, nbits, faiss.METRIC_INNER_PRODUCT)
                else:
                    index = faiss.IndexPQ(dimension, M, nbits, faiss.METRIC_L2)

                # PQ indexes need to be trained before use
                index.is_trained = False
            else:
                return {
                    "success": False,
                    "error": f"Unsupported index type: {index_type}",
                }

            # Store index and its dimension
            self.indexes[index_id] = index
            self.dimensions[index_id] = dimension

            index_details = {
                "index_id": index_id,
                "dimension": dimension,
                "type": index_type,
                "is_trained": getattr(index, "is_trained", True),
                "requires_training": main_type in ["IVF", "PQ"]
            }

            print(f"Created index {index_id} with dimension {dimension}, type {index_type}")
            return {
                "success": True,
                **index_details
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_vectors(self, index_id, vectors):
        """
        Add vectors to an existing index.

        Args:
            index_id (str): ID of the target index
            vectors (list): List of vectors to add

        Returns:
            dict: Response containing success status and count of added vectors or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            # Convert vectors to numpy array and validate dimensions
            vectors_np = np.array(vectors, dtype=np.float32)
            if vectors_np.shape[1] != self.dimensions[index_id]:
                return {
                    "success": False,
                    "error": (
                        f"Vector dimension mismatch. Expected {self.dimensions[index_id]}, "
                        f"got {vectors_np.shape[1]}"
                    ),
                }

            # Add vectors to index
            self.indexes[index_id].add(vectors_np)
            total = self.indexes[index_id].ntotal
            print(f"Added {len(vectors)} vectors to index {index_id}, total: {total}")
            return {"success": True, "count": len(vectors), "total": total}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search(self, index_id, query_vectors, k=10):
        """
        Search for similar vectors in an index.

        Args:
            index_id (str): ID of the target index
            query_vectors (list): List of query vectors
            k (int): Number of nearest neighbors to return

        Returns:
            dict: Response containing search results or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            # Convert query vectors to numpy array and validate dimensions
            query_np = np.array(query_vectors, dtype=np.float32)
            if query_np.shape[1] != self.dimensions[index_id]:
                return {
                    "success": False,
                    "error": (
                        f"Query vector dimension mismatch. "
                        f"Expected {self.dimensions[index_id]}, got {query_np.shape[1]}"
                    ),
                }

            # Perform search
            distances, indices = self.indexes[index_id].search(query_np, k)

            # Convert results to Python lists for serialization
            results = []
            for i in range(len(query_vectors)):
                results.append(
                    {"distances": distances[i].tolist(), "indices": indices[i].tolist()}
                )

            print(f"Searched index {index_id} with {len(query_vectors)} queries, k={k}")
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def range_search(self, index_id, query_vectors, radius):
        """
        Search for vectors within a specified radius.

        Args:
            index_id (str): ID of the target index
            query_vectors (list): List of query vectors
            radius (float): Maximum distance threshold

        Returns:
            dict: Response containing search results or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            # Check if the index supports range_search
            if not hasattr(self.indexes[index_id], 'range_search'):
                return {
                    "success": False,
                    "error": (
                        f"Index type {type(self.indexes[index_id]).__name__} "
                        f"does not support range search"
                    )
                }

            # Convert query vectors to numpy array and validate dimensions
            query_np = np.array(query_vectors, dtype=np.float32)
            if query_np.shape[1] != self.dimensions[index_id]:
                return {
                    "success": False,
                    "error": (
                        f"Query vector dimension mismatch. "
                        f"Expected {self.dimensions[index_id]}, got {query_np.shape[1]}"
                    ),
                }

            # Perform range search
            results = []
            for i in range(query_np.shape[0]):
                # Range search one query at a time to avoid memory issues
                lims, distances, indices = self.indexes[index_id].range_search(
                    query_np[i:i+1], radius
                )

                # Extract results for this query
                # lims[0] is the start, lims[1] is the end of results for the first query
                query_distances = distances[lims[0]:lims[1]].tolist()
                query_indices = indices[lims[0]:lims[1]].tolist()

                results.append({
                    "distances": query_distances,
                    "indices": query_indices,
                    "count": len(query_distances)
                })

            print(
                f"Range searched index {index_id} with {len(query_vectors)} "
                f"queries, radius={radius}"
            )
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_index_stats(self, index_id):
        """
        Get statistics for an index.

        Args:
            index_id (str): ID of the target index

        Returns:
            dict: Response containing index statistics or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            index = self.indexes[index_id]
            stats = {
                "index_id": index_id,
                "dimension": self.dimensions[index_id],
                "vector_count": index.ntotal,
                "type": "L2" if isinstance(index, faiss.IndexFlatL2) else "IP",
            }
            return {"success": True, "stats": stats}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_indexes(self):
        """
        List all available indexes.

        Returns:
            dict: Response containing list of indexes or error message
        """
        try:
            index_list = []
            for index_id in self.indexes:
                index_list.append(
                    {
                        "index_id": index_id,
                        "dimension": self.dimensions[index_id],
                        "vector_count": self.indexes[index_id].ntotal,
                    }
                )
            return {"success": True, "indexes": index_list}
        except Exception as e:
            return {"success": False, "error": str(e)}

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
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            index = self.indexes[index_id]

            # Check if index requires training
            if not hasattr(index, 'is_trained') or index.is_trained:
                return {"success": False, "error": "This index type does not require training"}

            # Convert vectors to numpy array and validate dimensions
            vectors_np = np.array(training_vectors, dtype=np.float32)
            if vectors_np.shape[1] != self.dimensions[index_id]:
                return {
                    "success": False,
                    "error": (
                        f"Training vector dimension mismatch. "
                        f"Expected {self.dimensions[index_id]}, got {vectors_np.shape[1]}"
                    ),
                }

            # Train the index
            index.train(vectors_np)

            print(f"Trained index {index_id} with {len(training_vectors)} vectors")
            return {
                "success": True,
                "index_id": index_id,
                "trained_with": len(training_vectors),
                "is_trained": index.is_trained
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


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
):
    """
    Run the ZeroMQ server for handling vector operations.

    Args:
        port (int): Port number for the server
        bind_address (str): Address to bind the server to
        auth_keys (dict): Dictionary of API keys for authentication
        enable_auth (bool): Whether to enable authentication
        data_dir (str): Directory for persistent storage
    """
    # Initialize ZeroMQ context and socket
    """Run the ZeroMQ server"""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{bind_address}:{port}")

    faiss_index = FaissIndex(data_dir=data_dir)

    # Print concise startup message
    storage_mode = f"persistent ({data_dir})" if data_dir else "in-memory"
    auth_status = "enabled" if enable_auth else "disabled"

    print("\n---------------------------------------------\n")
    print("███████╗█████╗ ██╗███████╗███████╗ ██╗  ██╗")
    print("██╔════██╔══██╗██║██╔════╝██╔════╝ ╚██╗██╔╝")
    print("█████╗ ███████║██║███████╗███████╗  ╚███╔╝")
    print("██╔══╝ ██╔══██║██║╚════██║╚════██║  ██╔██╗")
    print("██║    ██║  ██║██║███████║███████║ ██╔╝ ██╗")
    print("╚═╝    ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝ ╚═╝  ╚═╝")
    print("\n---------------------------------------------")
    print(f"FAISSx Server v{faissx_version} (c) 2025 Ran Aroussi")
    print("---------------------------------------------")
    print("\nStarting using:")
    print(f"  - FAISS version: {faiss.__version__}")
    print(f"  - Storage: {storage_mode}")
    print(f"  - Authentication: {auth_status}")
    # print("---------------------------------------------")
    print(f"\nStarted. Listening on {bind_address}:{port}...")
    print("\n---------------------------------------------\n")

    while True:
        try:
            # Wait for next request from client
            message = socket.recv()
            try:
                request = deserialize_message(message)
                action = request.get("action", "")

                # Handle authentication if enabled
                if enable_auth:
                    is_authenticated, error = authenticate_request(request, auth_keys)
                    if not is_authenticated:
                        response = {"success": False, "error": error}
                        socket.send(serialize_message(response))
                        continue

                if action == "create_index":
                    response = faiss_index.create_index(
                        request.get("index_id", ""),
                        request.get("dimension", 0),
                        request.get("index_type", "L2"),
                    )

                elif action == "add_vectors":
                    response = faiss_index.add_vectors(
                        request.get("index_id", ""), request.get("vectors", [])
                    )

                elif action == "search":
                    response = faiss_index.search(
                        request.get("index_id", ""),
                        request.get("query_vectors", []),
                        request.get("k", 10),
                    )

                elif action == "range_search":
                    response = faiss_index.range_search(
                        request.get("index_id", ""),
                        request.get("query_vectors", []),
                        request.get("radius", 1.0),
                    )

                elif action == "get_index_stats":
                    response = faiss_index.get_index_stats(request.get("index_id", ""))

                elif action == "list_indexes":
                    response = faiss_index.list_indexes()

                elif action == "train_index":
                    response = faiss_index.train_index(
                        request.get("index_id", ""),
                        request.get("training_vectors", []),
                    )

                elif action == "ping":
                    response = {"success": True, "message": "pong", "time": time.time()}

                else:
                    response = {"success": False, "error": f"Unknown action: {action}"}

            except Exception as e:
                response = {"success": False, "error": str(e)}

            # Send reply back to client
            socket.send(serialize_message(response))

        except KeyboardInterrupt:
            print("Server shutting down...")
            break
        except Exception as e:
            print(f"Error: {e}")

    socket.close()
    context.term()


if __name__ == "__main__":
    port = int(os.environ.get("faissx_PORT", DEFAULT_PORT))
    data_dir = os.environ.get("FAISS_DATA_DIR", None)
    bind_address = os.environ.get("FAISS_BIND_ADDRESS", DEFAULT_BIND_ADDRESS)

    # Default to no authentication when run directly
    run_server(port, bind_address, auth_keys={}, enable_auth=False, data_dir=data_dir)

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
        self.base_indexes = {}  # Dictionary to store relationships between ID maps and base indices

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
                - "IDMap:{base_index_id}" - IDMap wrapper for an existing index
                - "IDMap2:{base_index_id}" - IDMap2 wrapper for an existing index
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
            # Check for IDMap index types
            if index_type.startswith("IDMap"):
                # Handle both IDMap and IDMap2 types
                is_idmap2 = index_type.startswith("IDMap2:")

                # Extract the base index ID
                if ":" not in index_type:
                    return {
                        "success": False,
                        "error": "Invalid IDMap format. Use 'IDMap:base_index_id' or 'IDMap2:base_index_id'"
                    }

                base_index_id = index_type.split(":", 1)[1]

                # Check if base index exists
                if base_index_id not in self.indexes:
                    return {
                        "success": False,
                        "error": f"Base index {base_index_id} does not exist"
                    }

                base_index = self.indexes[base_index_id]

                # Special handling for non-empty base indices
                if base_index.ntotal > 0:
                    # For non-empty indices, we need to create a new empty index of the same type
                    # and then copy the vectors over to the IDMap wrapper

                    # Store vectors and IDs to add later
                    if hasattr(base_index, "reconstruct_n") and hasattr(base_index, "id_map"):
                        # For another IDMap as base, get IDs and vectors
                        ids = np.array(list(base_index.id_map.keys()), dtype=np.int64)
                        vectors = np.zeros((len(ids), self.dimensions[base_index_id]), dtype=np.float32)

                        for i, id_val in enumerate(ids):
                            vectors[i] = base_index.reconstruct(id_val)
                    else:
                        # For regular indices, create sequential IDs
                        vectors = np.zeros((base_index.ntotal, self.dimensions[base_index_id]), dtype=np.float32)
                        for i in range(base_index.ntotal):
                            vectors[i] = base_index.reconstruct(i)

                        ids = np.arange(base_index.ntotal, dtype=np.int64)

                    # Get the underlying index type by creating a fresh empty base index
                    # Create a copy of the same type as the original base index
                    base_copy = None

                    if isinstance(base_index, faiss.IndexFlatL2):
                        base_copy = faiss.IndexFlatL2(self.dimensions[base_index_id])
                    elif isinstance(base_index, faiss.IndexFlatIP):
                        base_copy = faiss.IndexFlatIP(self.dimensions[base_index_id])
                    elif isinstance(base_index, faiss.IndexIVFFlat):
                        # Create a similar IVF index
                        if base_index.metric_type == faiss.METRIC_L2:
                            quantizer = faiss.IndexFlatL2(self.dimensions[base_index_id])
                            base_copy = faiss.IndexIVFFlat(
                                quantizer, self.dimensions[base_index_id],
                                base_index.nlist, faiss.METRIC_L2
                            )
                        else:
                            quantizer = faiss.IndexFlatIP(self.dimensions[base_index_id])
                            base_copy = faiss.IndexIVFFlat(
                                quantizer, self.dimensions[base_index_id],
                                base_index.nlist, faiss.METRIC_INNER_PRODUCT
                            )

                        # Train the index if needed
                        if not base_copy.is_trained:
                            if base_index.is_trained:
                                base_copy.train(vectors)
                            else:
                                return {
                                    "success": False,
                                    "error": "Base index is not trained and cannot be used as template"
                                }
                    else:
                        # For other index types, attempt to determine basic parameters
                        # This is a simplification and might not handle all index types
                        return {
                            "success": False,
                            "error": "Complex index types with existing vectors are not supported for IDMap wrapping"
                        }

                    # Create appropriate IDMap index with the empty base
                    if is_idmap2:
                        index = faiss.IndexIDMap2(base_copy)
                    else:
                        index = faiss.IndexIDMap(base_copy)

                    # Store index and metadata
                    self.indexes[index_id] = index
                    self.dimensions[index_id] = self.dimensions[base_index_id]

                    # Store relationship for future reference
                    self.base_indexes[index_id] = base_index_id

                    # Add the vectors with IDs
                    index.add_with_ids(vectors, ids)

                    # Return success
                    return {
                        "success": True,
                        "index_id": index_id,
                        "dimension": self.dimensions[index_id],
                        "type": index_type,
                        "is_trained": getattr(index, "is_trained", True),
                        "base_index_id": base_index_id,
                        "vector_count": index.ntotal
                    }
                else:
                    # For empty base indices, proceed normally
                    # Create appropriate IDMap index
                    if is_idmap2:
                        index = faiss.IndexIDMap2(base_index)
                    else:
                        index = faiss.IndexIDMap(base_index)

                    # Store index and metadata
                    self.indexes[index_id] = index
                    self.dimensions[index_id] = self.dimensions[base_index_id]

                    # Store relationship for future reference
                    self.base_indexes[index_id] = base_index_id

                    # Return success
                    return {
                        "success": True,
                        "index_id": index_id,
                        "dimension": self.dimensions[index_id],
                        "type": index_type,
                        "is_trained": getattr(index, "is_trained", True),
                        "base_index_id": base_index_id
                    }

            # Handle other index types (existing code)
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
                            index_params["M"] = int(
                                params
                            )  # Just number of subquantizers
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
            if main_type == "L2" or (
                main_type == index_type and index_params["metric"] == "L2"
            ):
                index = faiss.IndexFlatL2(dimension)
            elif main_type == "IP" or (
                main_type == index_type and index_params["metric"] == "IP"
            ):
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
                    quantizer,
                    dimension,
                    nlist,
                    (
                        faiss.METRIC_INNER_PRODUCT
                        if index_params["metric"] == "IP"
                        else faiss.METRIC_L2
                    ),
                )

                # IVF indexes need to be trained before use
                index.is_trained = False
            elif main_type == "HNSW":
                # HNSW parameters
                M = index_params.get("M", 32)  # Default connections per node

                # Create HNSW index
                if index_params["metric"] == "IP":
                    index = faiss.IndexHNSWFlat(
                        dimension, M, faiss.METRIC_INNER_PRODUCT
                    )
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
                        "error": f"PQ requires dimension ({dimension}) to be a multiple of M ({M})",
                    }

                # Create PQ index
                if index_params["metric"] == "IP":
                    index = faiss.IndexPQ(
                        dimension, M, nbits, faiss.METRIC_INNER_PRODUCT
                    )
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
                "requires_training": main_type in ["IVF", "PQ"],
            }

            print(
                f"Created index {index_id} with dimension {dimension}, type {index_type}"
            )
            return {"success": True, **index_details}
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
        Reconstruct a vector from its ID.

        Args:
            index_id (str): ID of the target index
            id_val (int): ID of the vector to reconstruct

        Returns:
            dict: Response containing the reconstructed vector or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            # Get the index
            index = self.indexes[index_id]
            id_int = int(id_val)  # Convert to integer for consistent key lookup

            # Special handling for IDMap indices
            if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                # First, check if we have this vector in our cache
                if hasattr(self, '_vector_cache') and index_id in self._vector_cache:
                    # Check cache first for efficiency
                    if id_int in self._vector_cache[index_id]:
                        cached_vector = self._vector_cache[index_id][id_int]
                        # Make sure we have a list (serializable) not a numpy array
                        vector_list = cached_vector if isinstance(cached_vector, list) else cached_vector.tolist()
                        return {
                            "success": True,
                            "vector": vector_list
                        }

                # If not in cache, try direct reconstruction if supported
                try:
                    # Attempt direct reconstruction
                    vector = index.reconstruct(id_int)
                    vector_list = vector.tolist()

                    # Cache the result for future use
                    if not hasattr(self, '_vector_cache'):
                        self._vector_cache = {}
                    if index_id not in self._vector_cache:
                        self._vector_cache[index_id] = {}
                    self._vector_cache[index_id][id_int] = vector_list

                    return {"success": True, "vector": vector_list}
                except Exception:
                    # If direct reconstruction fails, try search-based fallback
                    # This is a last resort and can be very inefficient for large indices

                    # Try the base index if this is an IDMap
                    if index_id in self.base_indexes:
                        base_id = self.base_indexes[index_id]

                        # Check if the base index supports reconstruction
                        if hasattr(self.indexes[base_id], "reconstruct"):
                            try:
                                # For IDMap indices, we need to find the internal index
                                if hasattr(index, 'id_map'):
                                    internal_idx = index.id_map.get(id_int)
                                    if internal_idx is not None:
                                        # Use the base index to reconstruct
                                        vector = self.indexes[base_id].reconstruct(internal_idx)
                                        vector_list = vector.tolist()

                                        # Cache for future use
                                        if not hasattr(self, '_vector_cache'):
                                            self._vector_cache = {}
                                        if index_id not in self._vector_cache:
                                            self._vector_cache[index_id] = {}

                                        self._vector_cache[index_id][id_int] = vector_list

                                        return {"success": True, "vector": vector_list}
                            except Exception:
                                # Fall through to error
                                pass

                    # If all else fails, return more detailed error message
                    return {
                        "success": False,
                        "error": (
                            "Reconstruction not supported for this IDMap index type. "
                            "The vector was not found in cache and direct reconstruction failed."
                        )
                    }
            else:
                # Check if the index supports reconstruction
                if not hasattr(index, "reconstruct"):
                    return {
                        "success": False,
                        "error": f"Index {index_id} does not support reconstruction"
                    }

                # Reconstruct the vector
                vector = index.reconstruct(id_int)
                vector_list = vector.tolist()

                print(f"Reconstructed vector with ID {id_val} from index {index_id}")
                return {"success": True, "vector": vector_list}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def replace_vector(self, index_id, id_val, vector):
        """
        Replace a vector while preserving its ID (for IndexIDMap2 only).

        Args:
            index_id (str): ID of the target index
            id_val (int): ID of the vector to replace
            vector (list): New vector data

        Returns:
            dict: Response containing success status or error message
        """
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            # Check if index is an IDMap2 type
            index = self.indexes[index_id]
            if not isinstance(index, faiss.IndexIDMap2):
                return {
                    "success": False,
                    "error": f"Index {index_id} is not an IDMap2 type"
                }

            # Convert vector to numpy array
            vector_np = np.array([vector], dtype=np.float32)
            id_np = np.array([id_val], dtype=np.int64)

            # Validate dimension
            if vector_np.shape[1] != self.dimensions[index_id]:
                return {
                    "success": False,
                    "error": (
                        f"Vector dimension mismatch. Expected {self.dimensions[index_id]}, "
                        f"got {vector_np.shape[1]}"
                    )
                }

            # Replace the vector
            index.add_with_ids(vector_np, id_np)  # For IDMap2, this replaces existing vectors

            print(f"Replaced vector with ID {id_val} in index {index_id}")
            return {"success": True}
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
            if not hasattr(self.indexes[index_id], "range_search"):
                return {
                    "success": False,
                    "error": (
                        f"Index type {type(self.indexes[index_id]).__name__} "
                        f"does not support range search"
                    ),
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
                    query_np[i:i + 1], radius
                )

                # Extract results for this query
                # lims[0] is the start, lims[1] is the end of results for the first query
                query_distances = distances[lims[0]:lims[1]].tolist()
                query_indices = indices[lims[0]:lims[1]].tolist()

                results.append(
                    {
                        "distances": query_distances,
                        "indices": query_indices,
                        "count": len(query_distances),
                    }
                )

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

            # Add IDMap specific information if applicable
            if index_id in self.base_indexes:
                stats["base_index_id"] = self.base_indexes[index_id]
                stats["is_idmap"] = isinstance(index, faiss.IndexIDMap)
                stats["is_idmap2"] = isinstance(index, faiss.IndexIDMap2)

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
            if not hasattr(index, "is_trained") or index.is_trained:
                return {
                    "success": False,
                    "error": "This index type does not require training",
                }

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
                "is_trained": index.is_trained,
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

                elif action == "add_with_ids":
                    response = faiss_index.add_with_ids(
                        request.get("index_id", ""),
                        request.get("vectors", []),
                        request.get("ids", [])
                    )

                elif action == "remove_ids":
                    response = faiss_index.remove_ids(
                        request.get("index_id", ""),
                        request.get("ids", [])
                    )

                elif action == "reconstruct":
                    response = faiss_index.reconstruct(
                        request.get("index_id", ""),
                        request.get("id", 0)
                    )

                elif action == "replace_vector":
                    response = faiss_index.replace_vector(
                        request.get("index_id", ""),
                        request.get("id", 0),
                        request.get("vector", [])
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

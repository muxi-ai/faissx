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
FAISSx Protocol - Message Serialization and Deserialization

This module defines the binary communication protocol used between FAISSx clients and servers.

It provides functions for:
- Serializing and deserializing messages with headers / vector data / metadata
- Creating standardized request and response formats for all operations
- Handling binary vector data efficiently using NumPy arrays
- Formatting operation-specific messages (create_index, add_vectors, search, etc.)
- Error handling and response formatting

The protocol uses MessagePack for efficient binary serialization of structured data and raw binary
encoding for vector data to minimize overhead and maximize performance.
"""

import msgpack
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Constants for message structure
# Required fields in message headers
HEADER_FIELDS = ["operation", "api_key", "tenant_id", "index_id", "request_id"]
VECTOR_DTYPE = np.float32  # Standard dtype for vector data


class ProtocolError(Exception):
    """Custom exception for protocol-related errors during message parsing or formatting"""

    pass


def serialize_message(
    header: Dict[str, Any],
    vectors: Optional[np.ndarray] = None,
    metadata: Optional[Any] = None,
) -> bytes:
    """
    Serialize a message for sending over ZeroMQ.

    The message format consists of:
    1. A sizes header containing lengths of each component
    2. The message header (operation, auth info, etc.)
    3. Optional vector data
    4. Optional metadata

    Args:
        header: Message header with operation, auth, etc.
        vectors: Optional numpy array of vectors
        metadata: Optional metadata to include

    Returns:
        bytes: Serialized message ready for transmission
    """
    # 1. Serialize header with msgpack
    header_bytes = msgpack.packb(header)
    header_size = len(header_bytes)

    # 2. Initialize message parts list with header
    parts = [header_bytes]

    # 3. Process vector data if present
    vector_size = 0
    if vectors is not None:
        # Convert vectors to float32 if needed for consistency
        if vectors.dtype != VECTOR_DTYPE:
            vectors = vectors.astype(VECTOR_DTYPE)
        vector_bytes = vectors.tobytes()
        vector_size = len(vector_bytes)
        parts.append(vector_bytes)

    # 4. Process metadata if present
    metadata_size = 0
    if metadata is not None:
        metadata_bytes = msgpack.packb(metadata)
        metadata_size = len(metadata_bytes)
        parts.append(metadata_bytes)

    # 5. Create sizes header containing lengths of all components
    sizes = msgpack.packb(
        {
            "header_size": header_size,
            "vector_size": vector_size,
            "metadata_size": metadata_size,
        }
    )

    # 6. Combine all parts into final message
    return sizes + b"".join(parts)


def deserialize_message(
    data: bytes,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[Any]]:
    """
    Deserialize a message received over ZeroMQ.

    The deserialization process:
    1. Extracts the sizes header to determine component lengths
    2. Parses the message header
    3. Reconstructs vector data if present
    4. Extracts metadata if present

    Args:
        data: Raw message bytes received from ZeroMQ

    Returns:
        Tuple containing:
        - header: Dictionary with message header
        - vectors: Numpy array of vectors (if present)
        - metadata: Metadata object (if present)

    Raises:
        ProtocolError: If message format is invalid or parsing fails
    """
    try:
        # 1. Extract sizes header by finding end of msgpack map
        # \xc0 is msgpack's nil value, marking end of map
        sizes_end = data.find(b"\xc0")
        if sizes_end == -1:
            raise ProtocolError("Invalid message format: can't find sizes header")

        sizes = msgpack.unpackb(data[: sizes_end + 1])

        # 2. Get component sizes from header
        header_size = sizes.get("header_size", 0)
        vector_size = sizes.get("vector_size", 0)
        metadata_size = sizes.get("metadata_size", 0)

        offset = sizes_end + 1

        # 3. Extract and parse message header
        header = msgpack.unpackb(data[offset:offset + header_size])
        offset += header_size

        # 4. Process vector data if present
        vectors = None
        if vector_size > 0:
            vector_data = data[offset:offset + vector_size]

            # Reconstruct array shape based on header information
            if "vector_shape" in header:
                # Use explicit shape from header
                shape = header["vector_shape"]
                vectors = np.frombuffer(vector_data, dtype=VECTOR_DTYPE).reshape(shape)
            else:
                # Infer shape from dimension information
                dimension = header.get("dimension", 0)
                if dimension > 0:
                    # Calculate number of vectors based on data size
                    # 4 bytes per float32
                    count = vector_size // (dimension * 4)
                    vectors = np.frombuffer(vector_data, dtype=VECTOR_DTYPE).reshape(
                        count, dimension
                    )
                else:
                    # Fallback to raw buffer if shape can't be determined
                    vectors = np.frombuffer(vector_data, dtype=VECTOR_DTYPE)

            offset += vector_size

        # 5. Extract metadata if present
        metadata = None
        if metadata_size > 0:
            metadata = msgpack.unpackb(data[offset:offset + metadata_size])

        return header, vectors, metadata

    except (msgpack.UnpackException, IndexError, ValueError) as e:
        raise ProtocolError(f"Failed to deserialize message: {str(e)}")


# --- Operation-specific serialization/deserialization ---


def prepare_create_index_request(
    api_key: str,
    tenant_id: str,
    name: str,
    dimension: int,
    index_type: str = "IndexFlatL2",
) -> bytes:
    """
    Prepare a create_index request message.

    Creates a new vector index with specified parameters.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID for multi-tenancy
        name: Index name
        dimension: Vector dimension
        index_type: FAISS index type (default: IndexFlatL2)

    Returns:
        bytes: Serialized create_index request message
    """
    header = {
        "operation": "create_index",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "name": name,
        "dimension": dimension,
        "index_type": index_type,
    }
    return serialize_message(header)


def prepare_add_vectors_request(
    api_key: str,
    tenant_id: str,
    index_id: str,
    vectors: np.ndarray,
    vector_ids: List[str],
    vector_metadata: List[Dict[str, Any]],
) -> bytes:
    """
    Prepare an add_vectors request message.

    Adds multiple vectors to an existing index with associated metadata.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Target index ID
        vectors: Numpy array of vectors to add (shape: N x D)
        vector_ids: List of unique IDs for each vector
        vector_metadata: List of metadata dictionaries for each vector

    Returns:
        bytes: Serialized add_vectors request message
    """
    header = {
        "operation": "add_vectors",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
        "vector_shape": vectors.shape,
    }

    # Combine vector IDs with their metadata
    metadata = []
    for i, vector_id in enumerate(vector_ids):
        metadata.append(
            {
                "id": vector_id,
                "metadata": vector_metadata[i] if i < len(vector_metadata) else {},
            }
        )

    return serialize_message(header, vectors, metadata)


def prepare_search_request(
    api_key: str,
    tenant_id: str,
    index_id: str,
    query_vector: np.ndarray,
    k: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Prepare a search request message.

    Searches for k nearest neighbors of the query vector.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Target index ID
        query_vector: Query vector (shape: D)
        k: Number of nearest neighbors to return
        filter_metadata: Optional metadata filter criteria

    Returns:
        bytes: Serialized search request message
    """
    header = {
        "operation": "search",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
        "k": k,
        "dimension": query_vector.shape[0],
    }

    # Reshape single vector to 2D array if needed
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)

    return serialize_message(header, query_vector, filter_metadata)


def prepare_delete_vector_request(
    api_key: str, tenant_id: str, index_id: str, vector_id: str
) -> bytes:
    """
    Prepare a delete_vector request message.

    Removes a single vector from the index.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Target index ID
        vector_id: ID of vector to delete

    Returns:
        bytes: Serialized delete_vector request message
    """
    header = {
        "operation": "delete_vector",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
        "vector_id": vector_id,
    }
    return serialize_message(header)


def prepare_get_index_info_request(
    api_key: str, tenant_id: str, index_id: str
) -> bytes:
    """
    Prepare a get_index_info request message.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Index ID

    Returns:
        bytes: Serialized message
    """
    header = {
        "operation": "get_index_info",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
    }
    return serialize_message(header)


def prepare_delete_index_request(api_key: str, tenant_id: str, index_id: str) -> bytes:
    """
    Prepare a delete_index request message.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Index ID

    Returns:
        bytes: Serialized message
    """
    header = {
        "operation": "delete_index",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
    }
    return serialize_message(header)


# --- Response formatting ---


def prepare_success_response(result: Any = None) -> bytes:
    """
    Prepare a success response message.

    Args:
        result: Result data to include in the response

    Returns:
        bytes: Serialized message
    """
    header = {"status": "ok"}
    return serialize_message(header, None, result)


def prepare_error_response(
    error_type: str, message: str, request_id: Optional[str] = None
) -> bytes:
    """
    Prepare an error response message.

    Args:
        error_type: Type of error
        message: Error message
        request_id: Optional request ID for tracing

    Returns:
        bytes: Serialized message
    """
    header = {"status": "error", "error_type": error_type, "message": message}
    if request_id:
        header["request_id"] = request_id

    return serialize_message(header)

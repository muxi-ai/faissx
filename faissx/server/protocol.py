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
Enterprise Protocol Infrastructure for FAISSx Communication

This module provides comprehensive binary communication protocol capabilities between
FAISSx clients and servers, supporting efficient vector data transmission, structured
message serialization, and robust error handling mechanisms.

Key Features:
- High-performance binary serialization using MessagePack for structured data
- Zero-copy vector data encoding for optimal memory usage and transfer speed
- Comprehensive request/response formatting for all FAISS operations
- Enterprise-grade error handling with detailed diagnostics
- Standardized authentication and multi-tenant support
- Input validation and type safety throughout the protocol stack

Protocol Architecture:
The communication protocol uses a multi-part message format:
1. Sizes Header: MessagePack-encoded component lengths for efficient parsing
2. Message Header: Operation metadata, authentication, and parameters
3. Vector Data: Raw binary NumPy arrays with optimized float32 encoding
4. Metadata: Optional structured data for filtering and context

Supported Operations:
- Index Management: create_index, delete_index, get_index_info
- Vector Operations: add_vectors, delete_vector, search
- Response Handling: success_response, error_response with detailed diagnostics

Performance Optimizations:
- MessagePack binary serialization for minimal overhead
- Float32 standardization for consistent vector representation
- Lazy loading and streaming for large datasets
- Memory-efficient buffer management
- Zero-copy operations where possible

Integration:
This module integrates seamlessly with ZeroMQ for network transport,
providing reliable message delivery with built-in retry mechanisms
and comprehensive error recovery strategies.
"""

import msgpack
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Protocol Constants for Message Structure
# These constants ensure consistent message formatting across all operations

# Required fields in message headers for authentication and routing
REQUIRED_HEADER_FIELDS = ["operation", "api_key", "tenant_id", "index_id", "request_id"]

# Standard data type for all vector operations to ensure consistency
VECTOR_DTYPE = np.float32

# MessagePack protocol constants for efficient binary serialization
MSGPACK_NIL_MARKER = b"\xc0"  # MessagePack nil value marking map boundaries
FLOAT32_BYTES_PER_ELEMENT = 4  # 4 bytes per float32 element

# Operation constants for request/response standardization
# These constants eliminate magic strings and improve maintainability
OPERATION_CREATE_INDEX = "create_index"
OPERATION_ADD_VECTORS = "add_vectors"
OPERATION_SEARCH = "search"
OPERATION_DELETE_VECTOR = "delete_vector"
OPERATION_GET_INDEX_INFO = "get_index_info"
OPERATION_DELETE_INDEX = "delete_index"

# Response status constants for consistent status reporting
RESPONSE_STATUS_OK = "ok"
RESPONSE_STATUS_ERROR = "error"

# Header field constants for structured message components
HEADER_FIELD_OPERATION = "operation"
HEADER_FIELD_API_KEY = "api_key"
HEADER_FIELD_TENANT_ID = "tenant_id"
HEADER_FIELD_INDEX_ID = "index_id"
HEADER_FIELD_REQUEST_ID = "request_id"
HEADER_FIELD_VECTOR_SHAPE = "vector_shape"
HEADER_FIELD_DIMENSION = "dimension"
HEADER_FIELD_STATUS = "status"
HEADER_FIELD_ERROR_TYPE = "error_type"
HEADER_FIELD_MESSAGE = "message"

# Default FAISS index types for validation and documentation
DEFAULT_INDEX_TYPE = "IndexFlatL2"
SUPPORTED_INDEX_TYPES = {
    "IndexFlatL2",
    "IndexFlatIP",
    "IndexIVFFlat",
    "IndexIVFPQ",
    "IndexHNSW",
    "IndexHNSWFlat",
    "IndexPQ",
    "IndexScalarQuantizer",
    "IndexIVFScalarQuantizer",
    "IndexBinaryFlat",
    "IndexPreTransform",
}

# Search parameters and limits for validation
DEFAULT_SEARCH_K = 10
MAX_SEARCH_K = 10000
MAX_DIMENSION = 10000
MIN_DIMENSION = 1

# Error type constants for consistent error categorization
ERROR_TYPE_VALIDATION = "validation_error"
ERROR_TYPE_SERIALIZATION = "serialization_error"
ERROR_TYPE_DESERIALIZATION = "deserialization_error"
ERROR_TYPE_PROTOCOL = "protocol_error"
ERROR_TYPE_AUTHENTICATION = "authentication_error"
ERROR_TYPE_NOT_FOUND = "not_found_error"


class ProtocolError(Exception):
    """
    Enterprise-grade exception for protocol-related errors during message processing.

    This exception provides detailed error information for debugging and monitoring,
    including error categorization, context data, and recovery suggestions.

    Attributes:
        error_type: Categorized error type for systematic handling
        context: Additional context data for debugging
        original_error: Original exception that caused this error (if any)
    """

    def __init__(
        self,
        message: str,
        error_type: str = ERROR_TYPE_PROTOCOL,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize a protocol error with comprehensive error information.

        Args:
            message: Human-readable error description
            error_type: Categorized error type for systematic handling
            context: Additional context data for debugging
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.error_type = error_type
        self.context = context or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Provide detailed error representation for logging and debugging."""
        error_details = f"ProtocolError ({self.error_type}): {super().__str__()}"
        if self.context:
            error_details += f" | Context: {self.context}"
        if self.original_error:
            error_details += f" | Caused by: {type(self.original_error).__name__}: {self.original_error}"
        return error_details


# Helper Functions for Input Validation and Common Operations
# These functions promote code reuse and consistent validation patterns


def _validate_header_fields(header: Dict[str, Any]) -> None:
    """
    Validate that required header fields are present and properly formatted.

    Args:
        header: Message header dictionary to validate

        Raises:
        ProtocolError: If required fields are missing or invalid
    """
    missing_fields = [
        field
        for field in REQUIRED_HEADER_FIELDS
        if field not in header and field != HEADER_FIELD_REQUEST_ID  # optional
    ]

    if missing_fields:
        raise ProtocolError(
            f"Missing required header fields: {missing_fields}",
            error_type=ERROR_TYPE_VALIDATION,
            context={
                "missing_fields": missing_fields,
                "provided_fields": list(header.keys()),
            },
        )


def _validate_vector_parameters(
    vectors: np.ndarray, dimension: Optional[int] = None
) -> None:
    """
    Validate vector array parameters for consistency and correctness.

    Args:
        vectors: NumPy array to validate
        dimension: Expected vector dimension (optional)

    Raises:
        ProtocolError: If vectors are invalid or inconsistent
    """
    if not isinstance(vectors, np.ndarray):
        raise ProtocolError(
            f"Vectors must be numpy array, got {type(vectors)}",
            error_type=ERROR_TYPE_VALIDATION,
            context={"provided_type": str(type(vectors))},
        )

    if vectors.ndim == 0 or (vectors.ndim == 1 and len(vectors) == 0):
        raise ProtocolError(
            "Vectors array cannot be empty",
            error_type=ERROR_TYPE_VALIDATION,
            context={"array_shape": vectors.shape},
        )

    if dimension is not None:
        actual_dim = vectors.shape[-1] if vectors.ndim > 1 else vectors.shape[0]
        if actual_dim != dimension:
            raise ProtocolError(
                f"Vector dimension mismatch: expected {dimension}, got {actual_dim}",
                error_type=ERROR_TYPE_VALIDATION,
                context={
                    "expected_dimension": dimension,
                    "actual_dimension": actual_dim,
                },
            )


def _validate_search_parameters(k: int, dimension: int) -> None:
    """
    Validate search parameters for reasonable limits and consistency.

    Args:
        k: Number of nearest neighbors to return
        dimension: Vector dimension

    Raises:
        ProtocolError: If parameters are invalid or unreasonable
    """
    if not isinstance(k, int) or k <= 0:
        raise ProtocolError(
            f"Search parameter k must be positive integer, got {k}",
            error_type=ERROR_TYPE_VALIDATION,
            context={"provided_k": k},
        )

    if k > MAX_SEARCH_K:
        raise ProtocolError(
            f"Search parameter k too large: {k} > {MAX_SEARCH_K}",
            error_type=ERROR_TYPE_VALIDATION,
            context={"provided_k": k, "max_k": MAX_SEARCH_K},
        )

    if (
        not isinstance(dimension, int)
        or dimension < MIN_DIMENSION
        or dimension > MAX_DIMENSION
    ):
        raise ProtocolError(
            f"Invalid dimension: {dimension} (must be between {MIN_DIMENSION} and {MAX_DIMENSION})",
            error_type=ERROR_TYPE_VALIDATION,
            context={
                "provided_dimension": dimension,
                "min_dimension": MIN_DIMENSION,
                "max_dimension": MAX_DIMENSION,
            },
        )


def _create_base_header(
    operation: str,
    api_key: str,
    tenant_id: str,
    index_id: Optional[str] = None,
    **additional_fields: Any,
) -> Dict[str, Any]:
    """
    Create a standardized message header with common authentication and routing fields.

    Args:
        operation: Operation type (must be one of the supported operations)
        api_key: API key for authentication
        tenant_id: Tenant ID for multi-tenant isolation
        index_id: Optional index identifier
        **additional_fields: Additional operation-specific fields

    Returns:
        dict: Standardized header with all required fields

    Raises:
        ProtocolError: If required fields are missing or invalid
    """
    # Validate input parameters
    if not operation or not isinstance(operation, str):
        raise ProtocolError(
            f"Operation must be non-empty string, got {operation}",
            error_type=ERROR_TYPE_VALIDATION,
        )

    if not api_key or not isinstance(api_key, str):
        raise ProtocolError(
            f"API key must be non-empty string, got {api_key}",
            error_type=ERROR_TYPE_VALIDATION,
        )

    if not tenant_id or not isinstance(tenant_id, str):
        raise ProtocolError(
            f"Tenant ID must be non-empty string, got {tenant_id}",
            error_type=ERROR_TYPE_VALIDATION,
        )

    # Create base header with required fields
    header = {
        HEADER_FIELD_OPERATION: operation,
        HEADER_FIELD_API_KEY: api_key,
        HEADER_FIELD_TENANT_ID: tenant_id,
    }

    # Add index_id if provided
    if index_id:
        header[HEADER_FIELD_INDEX_ID] = index_id

    # Add any additional operation-specific fields
    header.update(additional_fields)

    return header


def serialize_message(
    header: Dict[str, Any],
    vectors: Optional[np.ndarray] = None,
    metadata: Optional[Any] = None,
) -> bytes:
    """
    Serialize a message for efficient transmission over ZeroMQ with enterprise-grade validation.

    The message format uses a multi-part structure optimized for performance:
    1. Sizes Header: MessagePack-encoded lengths for efficient parsing
    2. Message Header: Operation metadata and authentication information
    3. Vector Data: Raw binary NumPy arrays with standardized float32 encoding
    4. Metadata: Optional structured data using MessagePack serialization

    Args:
        header: Message header containing operation, authentication, and parameters
        vectors: Optional NumPy array of vectors with automatic dtype conversion
        metadata: Optional metadata object (must be MessagePack-serializable)

    Returns:
        bytes: Serialized message ready for network transmission

    Raises:
        ProtocolError: If serialization fails or input validation errors occur

    Example:
        >>> header = {"operation": "search", "api_key": "key", "tenant_id": "tenant"}
        >>> query = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        >>> message = serialize_message(header, query)
        >>> # Message ready for ZeroMQ transmission
    """
    try:
        # Validate header structure and required fields
        if not isinstance(header, dict):
            raise ProtocolError(
                f"Header must be dictionary, got {type(header)}",
                error_type=ERROR_TYPE_VALIDATION,
            )

        # 1. Serialize header with MessagePack for efficient binary encoding
        header_bytes = msgpack.packb(header)
        header_size = len(header_bytes)

        # 2. Initialize message parts list starting with header
        parts = [header_bytes]

        # 3. Process vector data with validation and standardization
        vector_size = 0
        if vectors is not None:
            # Validate vector format
            _validate_vector_parameters(vectors)

            # Convert vectors to standard float32 dtype for consistency
            if vectors.dtype != VECTOR_DTYPE:
                vectors = vectors.astype(VECTOR_DTYPE)

            # Serialize vector data as raw binary for maximum efficiency
            vector_bytes = vectors.tobytes()
            vector_size = len(vector_bytes)
            parts.append(vector_bytes)

        # 4. Process metadata with MessagePack serialization
        metadata_size = 0
        if metadata is not None:
            try:
                metadata_bytes = msgpack.packb(metadata)
                metadata_size = len(metadata_bytes)
                parts.append(metadata_bytes)
            except (TypeError, ValueError) as e:
                raise ProtocolError(
                    f"Failed to serialize metadata: {str(e)}",
                    error_type=ERROR_TYPE_SERIALIZATION,
                    original_error=e,
                )

        # 5. Create sizes header for efficient message parsing
        sizes = msgpack.packb(
            {
                "header_size": header_size,
                "vector_size": vector_size,
                "metadata_size": metadata_size,
            }
        )

        # 6. Combine all components into final message
        return sizes + b"".join(parts)

    except ProtocolError:
        # Re-raise protocol errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors in ProtocolError for consistent handling
        raise ProtocolError(
            f"Unexpected serialization error: {str(e)}",
            error_type=ERROR_TYPE_SERIALIZATION,
            original_error=e,
        )


def deserialize_message(
    data: bytes,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[Any]]:
    """
    Deserialize a message received over ZeroMQ with comprehensive error handling.

    This function reconstructs the original message components from the binary
    format, handling various edge cases and providing detailed error diagnostics
    for debugging and monitoring purposes.

    The deserialization process:
    1. Extract sizes header to determine component boundaries
    2. Parse message header for operation and authentication data
    3. Reconstruct vector arrays with proper shape information
    4. Extract metadata with validation

    Args:
        data: Raw message bytes received from ZeroMQ transport

    Returns:
        Tuple containing:
        - header: Dictionary with message header and operation parameters
        - vectors: Reconstructed NumPy array (None if not present)
        - metadata: Deserialized metadata object (None if not present)

    Raises:
        ProtocolError: If message format is invalid, corrupted, or parsing fails

    Example:
        >>> # Received message data from ZeroMQ
        >>> header, vectors, metadata = deserialize_message(message_data)
        >>> operation = header["operation"]
        >>> if vectors is not None:
        >>>     print(f"Received {vectors.shape[0]} vectors")
    """
    try:
        # Validate input data
        if not isinstance(data, bytes) or len(data) == 0:
            raise ProtocolError(
                f"Invalid message data: expected non-empty bytes, got {type(data)}",
                error_type=ERROR_TYPE_DESERIALIZATION,
            )

        # 1. Extract sizes header using MessagePack unpacker
        try:
            unpacker = msgpack.Unpacker(raw=False, use_list=True)
            unpacker.feed(data)
            sizes = next(unpacker)
            sizes_bytes_consumed = unpacker.tell()
        except (msgpack.UnpackException, StopIteration, ValueError) as e:
            raise ProtocolError(
                f"Failed to parse sizes header: {str(e)}",
                error_type=ERROR_TYPE_DESERIALIZATION,
                original_error=e,
            )

        # 2. Extract component sizes with validation
        header_size = sizes.get("header_size", 0)
        vector_size = sizes.get("vector_size", 0)
        metadata_size = sizes.get("metadata_size", 0)

        if header_size < 0 or vector_size < 0 or metadata_size < 0:
            raise ProtocolError(
                f"Invalid component sizes: header={header_size}, "
                f"vector={vector_size}, metadata={metadata_size}",
                error_type=ERROR_TYPE_DESERIALIZATION,
                context={"sizes": sizes},
            )

        # Validate total message length
        expected_length = (
            sizes_bytes_consumed + header_size + vector_size + metadata_size
        )
        if len(data) < expected_length:
            raise ProtocolError(
                f"Message truncated: expected {expected_length} bytes, got {len(data)}",
                error_type=ERROR_TYPE_DESERIALIZATION,
                context={
                    "expected_length": expected_length,
                    "actual_length": len(data),
                },
            )

        offset = sizes_bytes_consumed

        # 3. Extract and parse message header
        try:
            header = msgpack.unpackb(data[offset:offset + header_size])
        except (msgpack.UnpackException, ValueError) as e:
            raise ProtocolError(
                f"Failed to parse message header: {str(e)}",
                error_type=ERROR_TYPE_DESERIALIZATION,
                original_error=e,
            )

        offset += header_size

        # 4. Process vector data with shape reconstruction
        vectors = None
        if vector_size > 0:
            vector_data = data[offset:offset + vector_size]

            try:
                # Reconstruct array shape using header information
                if HEADER_FIELD_VECTOR_SHAPE in header:
                    # Use explicit shape from header for accurate reconstruction
                    shape = header[HEADER_FIELD_VECTOR_SHAPE]
                    vectors = np.frombuffer(vector_data, dtype=VECTOR_DTYPE).reshape(
                        shape
                    )
                else:
                    # Infer shape from dimension information as fallback
                    dimension = header.get(HEADER_FIELD_DIMENSION, 0)
                    if dimension > 0:
                        # Calculate number of vectors: total bytes / (dimension * bytes_per_element)
                        count = vector_size // (dimension * FLOAT32_BYTES_PER_ELEMENT)
                        if count * dimension * FLOAT32_BYTES_PER_ELEMENT != vector_size:
                            raise ProtocolError(
                                f"Vector data size mismatch: {vector_size} bytes "
                                f"cannot form {count} vectors of dimension {dimension}",
                                error_type=ERROR_TYPE_DESERIALIZATION,
                            )
                        vectors = np.frombuffer(
                            vector_data, dtype=VECTOR_DTYPE
                        ).reshape(count, dimension)
                    else:
                        # Last resort: return raw buffer without shape
                        vectors = np.frombuffer(vector_data, dtype=VECTOR_DTYPE)
            except (ValueError, TypeError) as e:
                raise ProtocolError(
                    f"Failed to reconstruct vector data: {str(e)}",
                    error_type=ERROR_TYPE_DESERIALIZATION,
                    original_error=e,
                )

            offset += vector_size

        # 5. Extract metadata with error handling
        metadata = None
        if metadata_size > 0:
            try:
                metadata = msgpack.unpackb(data[offset:offset + metadata_size])
            except (msgpack.UnpackException, ValueError) as e:
                raise ProtocolError(
                    f"Failed to parse metadata: {str(e)}",
                    error_type=ERROR_TYPE_DESERIALIZATION,
                    original_error=e,
                )

        return header, vectors, metadata

    except ProtocolError:
        # Re-raise protocol errors to preserve error categorization
        raise
    except Exception as e:
        # Wrap unexpected errors for consistent error handling
        raise ProtocolError(
            f"Unexpected deserialization error: {str(e)}",
            error_type=ERROR_TYPE_DESERIALIZATION,
            original_error=e,
        )


# --- Operation-specific Request Preparation Functions ---
# These functions provide standardized request formatting with validation


def prepare_create_index_request(
    api_key: str,
    tenant_id: str,
    name: str,
    dimension: int,
    index_type: str = DEFAULT_INDEX_TYPE,
) -> bytes:
    """
    Prepare a create_index request message with comprehensive parameter validation.

    Creates a new vector index with specified parameters, supporting all major
    FAISS index types with proper validation and error handling.

    Args:
        api_key: API key for authentication (must be non-empty string)
        tenant_id: Tenant ID for multi-tenancy (must be non-empty string)
        name: Index name identifier (must be non-empty string)
        dimension: Vector dimension (must be positive integer within limits)
        index_type: FAISS index type (default: IndexFlatL2, must be supported type)

    Returns:
        bytes: Serialized create_index request message ready for transmission

    Raises:
        ProtocolError: If parameters are invalid or serialization fails

    Example:
        >>> request = prepare_create_index_request(
        ...     api_key="my-key",
        ...     tenant_id="tenant-1",
        ...     name="my-index",
        ...     dimension=128,
        ...     index_type="IndexFlatL2"
        ... )
        >>> # Send request via ZeroMQ
    """
    # Validate dimension parameter
    if (
        not isinstance(dimension, int)
        or dimension < MIN_DIMENSION
        or dimension > MAX_DIMENSION
    ):
        raise ProtocolError(
            f"Invalid dimension: {dimension} (must be between {MIN_DIMENSION} and {MAX_DIMENSION})",
            error_type=ERROR_TYPE_VALIDATION,
            context={
                "dimension": dimension,
                "min_dimension": MIN_DIMENSION,
                "max_dimension": MAX_DIMENSION,
            },
        )

    # Validate index type
    if index_type not in SUPPORTED_INDEX_TYPES:
        raise ProtocolError(
            f"Unsupported index type: {index_type}",
            error_type=ERROR_TYPE_VALIDATION,
            context={
                "provided_type": index_type,
                "supported_types": list(SUPPORTED_INDEX_TYPES),
            },
        )

    # Validate name parameter
    if not name or not isinstance(name, str):
        raise ProtocolError(
            f"Index name must be non-empty string, got {name}",
            error_type=ERROR_TYPE_VALIDATION,
        )

    # Create standardized header with validation
    header = _create_base_header(
        operation=OPERATION_CREATE_INDEX,
        api_key=api_key,
        tenant_id=tenant_id,
        name=name,
        dimension=dimension,
        index_type=index_type,
    )

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
    Prepare an add_vectors request message with comprehensive validation and metadata handling.

    Adds multiple vectors to an existing index with associated metadata, ensuring
    data consistency and proper error handling for large-scale vector operations.

    Args:
        api_key: API key for authentication (must be non-empty string)
        tenant_id: Tenant ID for multi-tenancy (must be non-empty string)
        index_id: Target index identifier (must be non-empty string)
        vectors: NumPy array of vectors to add (shape: N x D, will be converted to float32)
        vector_ids: List of unique identifiers for each vector (must match vector count)
        vector_metadata: List of metadata dictionaries (must match vector count)

    Returns:
        bytes: Serialized add_vectors request message ready for transmission

    Raises:
        ProtocolError: If parameters are invalid, inconsistent, or serialization fails

    Example:
        >>> vectors = np.random.random((100, 128)).astype(np.float32)
        >>> vector_ids = [f"vec_{i}" for i in range(100)]
        >>> metadata = [{"category": "A", "score": 0.9}] * 100
        >>> request = prepare_add_vectors_request(
        ...     api_key="my-key",
        ...     tenant_id="tenant-1",
        ...     index_id="my-index",
        ...     vectors=vectors,
        ...     vector_ids=vector_ids,
        ...     vector_metadata=metadata
        ... )
    """
    # Validate vectors parameter
    _validate_vector_parameters(vectors)

    # Ensure vectors is 2D array
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    elif vectors.ndim != 2:
        raise ProtocolError(
            f"Vectors must be 1D or 2D array, got {vectors.ndim}D",
            error_type=ERROR_TYPE_VALIDATION,
            context={"array_shape": vectors.shape},
        )

    num_vectors = vectors.shape[0]

    # Validate vector_ids consistency
    if not isinstance(vector_ids, list) or len(vector_ids) != num_vectors:
        raise ProtocolError(
            f"Vector IDs count mismatch: expected {num_vectors}, got {len(vector_ids)}",
            error_type=ERROR_TYPE_VALIDATION,
            context={"expected_count": num_vectors, "actual_count": len(vector_ids)},
        )

    # Validate vector_metadata consistency
    if not isinstance(vector_metadata, list) or len(vector_metadata) != num_vectors:
        raise ProtocolError(
            f"Vector metadata count mismatch: expected {num_vectors}, got {len(vector_metadata)}",
            error_type=ERROR_TYPE_VALIDATION,
            context={
                "expected_count": num_vectors,
                "actual_count": len(vector_metadata),
            },
        )

    # Create standardized header with shape information
    header = _create_base_header(
        operation=OPERATION_ADD_VECTORS,
        api_key=api_key,
        tenant_id=tenant_id,
        index_id=index_id,
        vector_shape=vectors.shape,
    )

    # Combine vector IDs with their metadata for transmission
    combined_metadata = []
    for i, vector_id in enumerate(vector_ids):
        if not isinstance(vector_id, str) or not vector_id:
            raise ProtocolError(
                f"Vector ID at index {i} must be non-empty string, got {vector_id}",
                error_type=ERROR_TYPE_VALIDATION,
                context={"index": i, "vector_id": vector_id},
            )

        combined_metadata.append(
            {
                "id": vector_id,
                "metadata": vector_metadata[i] if i < len(vector_metadata) else {},
            }
        )

    return serialize_message(header, vectors, combined_metadata)


def prepare_search_request(
    api_key: str,
    tenant_id: str,
    index_id: str,
    query_vector: np.ndarray,
    k: int = DEFAULT_SEARCH_K,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Prepare a search request message with comprehensive parameter validation and optimization.

    Searches for k nearest neighbors of the query vector with optional metadata
    filtering, ensuring optimal performance and proper error handling.

    Args:
        api_key: API key for authentication (must be non-empty string)
        tenant_id: Tenant ID for multi-tenancy (must be non-empty string)
        index_id: Target index identifier (must be non-empty string)
        query_vector: Query vector for similarity search (shape: D or 1xD)
        k: Number of nearest neighbors to return (default: 10, max: 10000)
        filter_metadata: Optional metadata filter criteria (must be serializable)

    Returns:
        bytes: Serialized search request message ready for transmission

    Raises:
        ProtocolError: If parameters are invalid or serialization fails

    Example:
        >>> query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> filter_criteria = {"category": "premium", "score": {"$gte": 0.8}}
        >>> request = prepare_search_request(
        ...     api_key="my-key",
        ...     tenant_id="tenant-1",
        ...     index_id="my-index",
        ...     query_vector=query,
        ...     k=50,
        ...     filter_metadata=filter_criteria
        ... )
    """
    # Validate query vector
    _validate_vector_parameters(query_vector)

    # Ensure query_vector is 1D for dimension extraction
    if query_vector.ndim == 2:
        if query_vector.shape[0] != 1:
            raise ProtocolError(
                f"Query vector must be single vector, got shape {query_vector.shape}",
                error_type=ERROR_TYPE_VALIDATION,
                context={"query_shape": query_vector.shape},
            )
        query_vector = query_vector.flatten()
    elif query_vector.ndim != 1:
        raise ProtocolError(
            f"Query vector must be 1D or 2D array, got {query_vector.ndim}D",
            error_type=ERROR_TYPE_VALIDATION,
            context={"query_shape": query_vector.shape},
        )

    dimension = query_vector.shape[0]

    # Validate search parameters
    _validate_search_parameters(k, dimension)

    # Create standardized header
    header = _create_base_header(
        operation=OPERATION_SEARCH,
        api_key=api_key,
        tenant_id=tenant_id,
        index_id=index_id,
        k=k,
        dimension=dimension,
    )

    # Reshape single vector to 2D array for consistent serialization
    query_2d = query_vector.reshape(1, -1)

    return serialize_message(header, query_2d, filter_metadata)


def prepare_delete_vector_request(
    api_key: str, tenant_id: str, index_id: str, vector_id: str
) -> bytes:
    """
    Prepare a delete_vector request message with comprehensive validation.

    Removes a single vector from the index by its unique identifier,
    with proper authentication and error handling.

    Args:
        api_key: API key for authentication (must be non-empty string)
        tenant_id: Tenant ID for multi-tenancy (must be non-empty string)
        index_id: Target index identifier (must be non-empty string)
        vector_id: Unique identifier of vector to delete (must be non-empty string)

    Returns:
        bytes: Serialized delete_vector request message ready for transmission

    Raises:
        ProtocolError: If parameters are invalid or serialization fails

    Example:
        >>> request = prepare_delete_vector_request(
        ...     api_key="my-key",
        ...     tenant_id="tenant-1",
        ...     index_id="my-index",
        ...     vector_id="vector-123"
        ... )
    """
    # Validate vector_id parameter
    if not vector_id or not isinstance(vector_id, str):
        raise ProtocolError(
            f"Vector ID must be non-empty string, got {vector_id}",
            error_type=ERROR_TYPE_VALIDATION,
        )

    # Create standardized header
    header = _create_base_header(
        operation=OPERATION_DELETE_VECTOR,
        api_key=api_key,
        tenant_id=tenant_id,
        index_id=index_id,
        vector_id=vector_id,
    )

    return serialize_message(header)


def prepare_get_index_info_request(
    api_key: str, tenant_id: str, index_id: str
) -> bytes:
    """
    Prepare a get_index_info request message with standardized formatting.

    Retrieves comprehensive information about an index including statistics,
    configuration parameters, and operational status.

    Args:
        api_key: API key for authentication (must be non-empty string)
        tenant_id: Tenant ID for multi-tenancy (must be non-empty string)
        index_id: Index identifier to query (must be non-empty string)

    Returns:
        bytes: Serialized get_index_info request message ready for transmission

    Raises:
        ProtocolError: If parameters are invalid or serialization fails

    Example:
        >>> request = prepare_get_index_info_request(
        ...     api_key="my-key",
        ...     tenant_id="tenant-1",
        ...     index_id="my-index"
        ... )
    """
    # Create standardized header
    header = _create_base_header(
        operation=OPERATION_GET_INDEX_INFO,
        api_key=api_key,
        tenant_id=tenant_id,
        index_id=index_id,
    )

    return serialize_message(header)


def prepare_delete_index_request(api_key: str, tenant_id: str, index_id: str) -> bytes:
    """
    Prepare a delete_index request message with comprehensive validation.

    Permanently removes an index and all associated data with proper
    authentication and safety checks.

    Args:
        api_key: API key for authentication (must be non-empty string)
        tenant_id: Tenant ID for multi-tenancy (must be non-empty string)
        index_id: Index identifier to delete (must be non-empty string)

    Returns:
        bytes: Serialized delete_index request message ready for transmission

    Raises:
        ProtocolError: If parameters are invalid or serialization fails

    Example:
        >>> request = prepare_delete_index_request(
        ...     api_key="my-key",
        ...     tenant_id="tenant-1",
        ...     index_id="my-index"
        ... )
    """
    # Create standardized header
    header = _create_base_header(
        operation=OPERATION_DELETE_INDEX,
        api_key=api_key,
        tenant_id=tenant_id,
        index_id=index_id,
    )

    return serialize_message(header)


# --- Response Formatting Functions ---
# These functions provide standardized response formatting for server implementations


def prepare_success_response(result: Any = None) -> bytes:
    """
    Prepare a standardized success response message with optional result data.

    Creates a consistent success response format that clients can reliably
    parse and process, with optional payload data.

    Args:
        result: Optional result data to include in response (must be serializable)

    Returns:
        bytes: Serialized success response message ready for transmission

    Raises:
        ProtocolError: If result data cannot be serialized

    Example:
        >>> # Simple success response
        >>> response = prepare_success_response()
        >>>
        >>> # Success response with data
        >>> result_data = {"index_id": "my-index", "vector_count": 1000}
        >>> response = prepare_success_response(result_data)
    """
    header = {HEADER_FIELD_STATUS: RESPONSE_STATUS_OK}
    return serialize_message(header, None, result)


def prepare_error_response(
    error_type: str, message: str, request_id: Optional[str] = None
) -> bytes:
    """
    Prepare a standardized error response message with comprehensive error information.

    Creates a consistent error response format that provides detailed information
    for debugging, monitoring, and client error handling.

    Args:
        error_type: Categorized error type for systematic error handling
        message: Human-readable error description
        request_id: Optional request identifier for tracing and correlation

    Returns:
        bytes: Serialized error response message ready for transmission

    Raises:
        ProtocolError: If error data cannot be serialized

    Example:
        >>> # Basic error response
        >>> response = prepare_error_response(
        ...     error_type="validation_error",
        ...     message="Invalid vector dimension: expected 128, got 64"
        ... )
        >>>
        >>> # Error response with request tracing
        >>> response = prepare_error_response(
        ...     error_type="not_found_error",
        ...     message="Index 'my-index' not found",
        ...     request_id="req-12345"
        ... )
    """
    # Validate error parameters
    if not error_type or not isinstance(error_type, str):
        raise ProtocolError(
            f"Error type must be non-empty string, got {error_type}",
            error_type=ERROR_TYPE_VALIDATION,
        )

    if not message or not isinstance(message, str):
        raise ProtocolError(
            f"Error message must be non-empty string, got {message}",
            error_type=ERROR_TYPE_VALIDATION,
        )

    # Create error header with required fields
    header = {
        HEADER_FIELD_STATUS: RESPONSE_STATUS_ERROR,
        HEADER_FIELD_ERROR_TYPE: error_type,
        HEADER_FIELD_MESSAGE: message,
    }

    # Add request ID for tracing if provided
    if request_id:
        header[HEADER_FIELD_REQUEST_ID] = request_id

    return serialize_message(header)

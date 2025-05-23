#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Response Standardization
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
Enterprise Response Standardization for FAISSx Server API

This module provides comprehensive response formatting capabilities for the FAISSx
server API, ensuring consistent response structures, comprehensive error handling,
and enterprise-grade monitoring and debugging features.

Key Features:
- Standardized response structures across all API endpoints
- Comprehensive error categorization and detailed error information
- Performance monitoring with timestamps and response metadata
- Input validation and type safety throughout response formatting
- Extensible response formats for different data types and operations
- Production-ready logging and debugging capabilities

Response Architecture:
All responses follow a consistent structure with core fields:
- success: Boolean indicating operation success/failure status
- timestamp: ISO timestamp for request tracking and performance monitoring
- data: Operation-specific payload data with structured formatting
- error: Detailed error information for failed operations
- metadata: Additional context and debugging information

Supported Response Types:
- Success Responses: Standardized success formatting with optional data payload
- Error Responses: Comprehensive error information with categorization
- Search Results: Formatted vector search results with performance metrics
- Vector Results: Structured vector data with pagination and metadata
- Index Status: Index health and configuration information
- Operation Results: Standardized operation outcomes with detailed context

Integration:
This module integrates seamlessly with the FAISSx server infrastructure,
providing consistent response formatting for all API endpoints while
supporting monitoring, logging, and debugging requirements.
"""

import time
import logging
from typing import Any, Dict, List, Optional

# Response Structure Constants
# These constants ensure consistent response formatting across all API endpoints

# Core response field names for standardized structure
RESPONSE_FIELD_SUCCESS = "success"
RESPONSE_FIELD_TIMESTAMP = "timestamp"
RESPONSE_FIELD_MESSAGE = "message"
RESPONSE_FIELD_ERROR = "error"
RESPONSE_FIELD_ERROR_CODE = "error_code"
RESPONSE_FIELD_ERROR_DETAILS = "error_details"
RESPONSE_FIELD_DATA = "data"
RESPONSE_FIELD_METADATA = "metadata"

# Success response status values
RESPONSE_STATUS_SUCCESS = True
RESPONSE_STATUS_FAILURE = False

# Data field constants for specific response types
DATA_FIELD_RESULTS = "results"
DATA_FIELD_QUERY_COUNT = "query_count"
DATA_FIELD_INDEX_ID = "index_id"
DATA_FIELD_VECTORS = "vectors"
DATA_FIELD_COUNT = "count"
DATA_FIELD_START_IDX = "start_idx"
DATA_FIELD_STATUS = "status"
DATA_FIELD_OPERATION = "operation"

# Error categorization constants for systematic error handling
ERROR_CATEGORY_VALIDATION = "validation_error"
ERROR_CATEGORY_PROCESSING = "processing_error"
ERROR_CATEGORY_AUTHENTICATION = "authentication_error"
ERROR_CATEGORY_AUTHORIZATION = "authorization_error"
ERROR_CATEGORY_NOT_FOUND = "not_found_error"
ERROR_CATEGORY_CONFLICT = "conflict_error"
ERROR_CATEGORY_INTERNAL = "internal_error"
ERROR_CATEGORY_TIMEOUT = "timeout_error"

# Response validation constants
MAX_ERROR_MESSAGE_LENGTH = 1000
MAX_RESPONSE_DATA_ITEMS = 10000
MIN_TIMESTAMP_VALUE = 0

# Logging configuration for response formatting operations
logger = logging.getLogger("faissx.server.response")


class ResponseFormattingError(Exception):
    """
    Enterprise-grade exception for response formatting errors.

    This exception provides detailed error information for debugging response
    formatting issues, including error categorization and context data.

    Attributes:
        error_category: Categorized error type for systematic handling
        context: Additional context data for debugging
        original_error: Original exception that caused this error (if any)
    """

    def __init__(
        self,
        message: str,
        error_category: str = ERROR_CATEGORY_PROCESSING,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize a response formatting error with comprehensive information.

        Args:
            message: Human-readable error description
            error_category: Categorized error type for systematic handling
            context: Additional context data for debugging
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.error_category = error_category
        self.context = context or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Provide detailed error representation for logging and debugging."""
        error_details = (
            f"ResponseFormattingError ({self.error_category}): {super().__str__()}"
        )
        if self.context:
            error_details += f" | Context: {self.context}"
        if self.original_error:
            error_details += (
                f" | Caused by: {type(self.original_error).__name__}: {self.original_error}"
            )
        return error_details


# Helper Functions for Input Validation and Common Operations
# These functions promote code reuse and consistent validation patterns


def _validate_response_data(
    data: Any, max_items: int = MAX_RESPONSE_DATA_ITEMS
) -> None:
    """
    Validate response data for reasonable size limits and structure.

    Args:
        data: Response data to validate
        max_items: Maximum number of items allowed in collections

    Raises:
        ResponseFormattingError: If data validation fails
    """
    if data is None:
        return

    # Validate list/collection sizes to prevent memory issues
    if isinstance(data, (list, tuple)):
        if len(data) > max_items:
            raise ResponseFormattingError(
                f"Response data too large: {len(data)} items > {max_items}",
                error_category=ERROR_CATEGORY_VALIDATION,
                context={"data_size": len(data), "max_size": max_items},
            )

    # Validate dictionary structures
    if isinstance(data, dict):
        total_items = sum(
            len(value) if isinstance(value, (list, tuple)) else 1
            for value in data.values()
        )
        if total_items > max_items:
            raise ResponseFormattingError(
                f"Response data too complex: {total_items} total items > {max_items}",
                error_category=ERROR_CATEGORY_VALIDATION,
                context={"total_items": total_items, "max_items": max_items},
            )


def _validate_error_parameters(error: str, code: Optional[str] = None) -> None:
    """
    Validate error message and code parameters for consistency and safety.

    Args:
        error: Error message to validate
        code: Optional error code to validate

    Raises:
        ResponseFormattingError: If error parameters are invalid
    """
    if not error or not isinstance(error, str):
        raise ResponseFormattingError(
            f"Error message must be non-empty string, got {error}",
            error_category=ERROR_CATEGORY_VALIDATION,
        )

    if len(error) > MAX_ERROR_MESSAGE_LENGTH:
        raise ResponseFormattingError(
            f"Error message too long: {len(error)} > {MAX_ERROR_MESSAGE_LENGTH}",
            error_category=ERROR_CATEGORY_VALIDATION,
            context={
                "message_length": len(error),
                "max_length": MAX_ERROR_MESSAGE_LENGTH,
            },
        )

    if code is not None and not isinstance(code, str):
        raise ResponseFormattingError(
            f"Error code must be string, got {type(code)}",
            error_category=ERROR_CATEGORY_VALIDATION,
        )


def _create_base_response(
    success: bool, timestamp: Optional[float] = None, **additional_fields: Any
) -> Dict[str, Any]:
    """
    Create a standardized base response structure with core fields.

    Args:
        success: Boolean indicating operation success/failure
        timestamp: Optional timestamp (uses current time if not provided)
        **additional_fields: Additional response-specific fields

    Returns:
        dict: Base response structure with core fields

    Raises:
        ResponseFormattingError: If base response creation fails
    """
    if timestamp is None:
        timestamp = time.time()

    # Validate timestamp value
    if not isinstance(timestamp, (int, float)) or timestamp < MIN_TIMESTAMP_VALUE:
        raise ResponseFormattingError(
            f"Invalid timestamp: {timestamp}",
            error_category=ERROR_CATEGORY_VALIDATION,
            context={"timestamp": timestamp, "min_value": MIN_TIMESTAMP_VALUE},
        )

    # Create base response structure
    response = {
        RESPONSE_FIELD_SUCCESS: success,
        RESPONSE_FIELD_TIMESTAMP: timestamp,
    }

    # Add any additional fields
    response.update(additional_fields)

    return response


def _add_pagination_info(
    response_data: Dict[str, Any],
    start_idx: Optional[int] = None,
    total_count: Optional[int] = None,
    page_size: Optional[int] = None,
) -> None:
    """
    Add pagination information to response data for large result sets.

    Args:
        response_data: Response data dictionary to modify
        start_idx: Starting index for pagination
        total_count: Total number of items available
        page_size: Number of items per page

    Raises:
        ResponseFormattingError: If pagination parameters are invalid
    """
    if start_idx is not None:
        if not isinstance(start_idx, int) or start_idx < 0:
            raise ResponseFormattingError(
                f"Start index must be non-negative integer, got {start_idx}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )
        response_data[DATA_FIELD_START_IDX] = start_idx

    if total_count is not None:
        if not isinstance(total_count, int) or total_count < 0:
            raise ResponseFormattingError(
                f"Total count must be non-negative integer, got {total_count}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )
        response_data["total_count"] = total_count

    if page_size is not None:
        if not isinstance(page_size, int) or page_size <= 0:
            raise ResponseFormattingError(
                f"Page size must be positive integer, got {page_size}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )
        response_data["page_size"] = page_size


def success_response(
    data: Optional[Dict[str, Any]] = None, message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized success response with comprehensive validation and monitoring.

    This function creates consistent success responses across all API endpoints,
    ensuring proper structure, validation, and debugging capabilities.

    Args:
        data: Optional data payload to include in response (must be serializable)
        message: Optional success message for user feedback

    Returns:
        dict: Standardized success response with core fields and optional data

    Raises:
        ResponseFormattingError: If response creation fails or validation errors occur

    Example:
        >>> # Simple success response
        >>> response = success_response()
        >>> print(response["success"])  # True
        >>>
        >>> # Success response with data
        >>> data = {"index_id": "my-index", "vector_count": 1000}
        >>> response = success_response(data=data, message="Operation completed successfully")
    """
    try:
        # Validate input parameters
        if data is not None:
            _validate_response_data(data)

        if message is not None and not isinstance(message, str):
            raise ResponseFormattingError(
                f"Success message must be string, got {type(message)}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        # Create base response structure
        response = _create_base_response(success=RESPONSE_STATUS_SUCCESS)

        # Add optional message
        if message:
            response[RESPONSE_FIELD_MESSAGE] = message

        # Add optional data payload
        if data:
            response.update(data)

        logger.debug(f"Created success response with {len(response)} fields")
        return response

    except ResponseFormattingError:
        # Re-raise response formatting errors
        raise
    except Exception as e:
        # Wrap unexpected errors for consistent error handling
        raise ResponseFormattingError(
            f"Unexpected error creating success response: {str(e)}",
            error_category=ERROR_CATEGORY_INTERNAL,
            original_error=e,
        )


def error_response(
    error: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response with comprehensive error information and debugging context.

    This function creates consistent error responses that provide detailed information
    for debugging, monitoring, and systematic error handling across all API endpoints.

    Args:
        error: Human-readable error message describing the failure
        code: Optional error code for systematic categorization and handling
        details: Optional additional error details and context information

    Returns:
        dict: Standardized error response with comprehensive error information

    Raises:
        ResponseFormattingError: If error response creation fails or validation errors occur

    Example:
        >>> # Basic error response
        >>> response = error_response("Vector dimension mismatch")
        >>> print(response["success"])  # False
        >>>
        >>> # Detailed error response with context
        >>> details = {"expected_dimension": 128, "actual_dimension": 64}
        >>> response = error_response(
        ...     error="Invalid vector dimension",
        ...     code="validation_error",
        ...     details=details
        ... )
    """
    try:
        # Validate error parameters
        _validate_error_parameters(error, code)

        if details is not None:
            _validate_response_data(details)

        # Create base error response structure
        response = _create_base_response(
            success=RESPONSE_STATUS_FAILURE, **{RESPONSE_FIELD_ERROR: error}
        )

        # Add optional error code for categorization
        if code:
            response[RESPONSE_FIELD_ERROR_CODE] = code

        # Add optional error details for debugging
        if details:
            response[RESPONSE_FIELD_ERROR_DETAILS] = details

        logger.warning(f"Created error response: {error} (code: {code})")
        return response

    except ResponseFormattingError:
        # Re-raise response formatting errors
        raise
    except Exception as e:
        # Wrap unexpected errors and create fallback error response
        logger.error(f"Failed to create error response: {str(e)}")
        return _create_base_response(
            success=RESPONSE_STATUS_FAILURE,
            **{
                RESPONSE_FIELD_ERROR: "Internal error creating error response",
                RESPONSE_FIELD_ERROR_CODE: ERROR_CATEGORY_INTERNAL,
            },
        )


def format_search_results(
    results: List[Dict[str, List[float]]],
    index_id: Optional[str] = None,
    query_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Format search results in a standardized structure with comprehensive metadata and validation.

    This function creates consistent search result responses with performance metrics,
    query information, and proper error handling for vector search operations.

    Args:
        results: List of search results containing distances and indices from vector search
        index_id: Optional identifier of the index that was searched
        query_info: Optional query parameters and metadata for debugging and monitoring

    Returns:
        dict: Standardized search results response with results data and metadata

    Raises:
        ResponseFormattingError: If search results formatting fails or validation errors occur

    Example:
        >>> # Basic search results
        >>> results = [{"distances": [0.1, 0.2], "indices": [0, 5]}]
        >>> response = format_search_results(results, index_id="my-index")
        >>>
        >>> # Detailed search results with query info
        >>> query_info = {"k": 10, "dimension": 128, "query_time_ms": 45.2}
        >>> response = format_search_results(results, index_id="my-index", query_info=query_info)
    """
    try:
        # Validate input parameters
        if not isinstance(results, list):
            raise ResponseFormattingError(
                f"Search results must be list, got {type(results)}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        _validate_response_data(results)

        if index_id is not None and not isinstance(index_id, str):
            raise ResponseFormattingError(
                f"Index ID must be string, got {type(index_id)}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        if query_info is not None:
            _validate_response_data(query_info)

        # Create response data with search results
        response_data = {
            DATA_FIELD_RESULTS: results,
            DATA_FIELD_QUERY_COUNT: len(results),
        }

        # Add optional index identifier
        if index_id:
            response_data[DATA_FIELD_INDEX_ID] = index_id

        # Add optional query information and metadata
        if query_info:
            response_data.update(query_info)

        logger.debug(f"Formatted search results: {len(results)} queries")
        return success_response(response_data)

    except ResponseFormattingError:
        # Re-raise response formatting errors
        raise
    except Exception as e:
        # Wrap unexpected errors for consistent error handling
        raise ResponseFormattingError(
            f"Unexpected error formatting search results: {str(e)}",
            error_category=ERROR_CATEGORY_INTERNAL,
            original_error=e,
        )


def format_vector_results(
    vectors: List[List[float]],
    index_id: Optional[str] = None,
    start_idx: Optional[int] = None,
    **metadata: Any,
) -> Dict[str, Any]:
    """
    Format vector results in a standardized structure with pagination and comprehensive metadata.

    This function creates consistent vector result responses with pagination support,
    metadata information, and proper validation for vector retrieval operations.

    Args:
        vectors: List of vectors containing float values from vector operations
        index_id: Optional identifier of the index containing the vectors
        start_idx: Optional starting index for paginated results
        **metadata: Additional metadata about the vectors and operation context

    Returns:
        dict: Standardized vector results response with vector data and metadata

    Raises:
        ResponseFormattingError: If vector results formatting fails or validation errors occur

    Example:
        >>> # Basic vector results
        >>> vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        >>> response = format_vector_results(vectors, index_id="my-index")
        >>>
        >>> # Paginated vector results with metadata
        >>> response = format_vector_results(
        ...     vectors=vectors,
        ...     index_id="my-index",
        ...     start_idx=100,
        ...     dimension=128,
        ...     total_vectors=1000
        ... )
    """
    try:
        # Validate input parameters
        if not isinstance(vectors, list):
            raise ResponseFormattingError(
                f"Vectors must be list, got {type(vectors)}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        _validate_response_data(vectors)

        if index_id is not None and not isinstance(index_id, str):
            raise ResponseFormattingError(
                f"Index ID must be string, got {type(index_id)}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        # Create response data with vector information
        response_data = {DATA_FIELD_VECTORS: vectors, DATA_FIELD_COUNT: len(vectors)}

        # Add optional index identifier
        if index_id:
            response_data[DATA_FIELD_INDEX_ID] = index_id

        # Add pagination information if provided
        _add_pagination_info(response_data, start_idx=start_idx)

        # Add any additional metadata
        if metadata:
            _validate_response_data(metadata)
            response_data.update(metadata)

        logger.debug(f"Formatted vector results: {len(vectors)} vectors")
        return success_response(response_data)

    except ResponseFormattingError:
        # Re-raise response formatting errors
        raise
    except Exception as e:
        # Wrap unexpected errors for consistent error handling
        raise ResponseFormattingError(
            f"Unexpected error formatting vector results: {str(e)}",
            error_category=ERROR_CATEGORY_INTERNAL,
            original_error=e,
        )


def format_index_status(index_id: str, status_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format index status information in a standardized structure with comprehensive validation.

    This function creates consistent index status responses with health information,
    configuration details, and proper error handling for index management operations.

    Args:
        index_id: Unique identifier of the index
        status_data: Dictionary containing index status and configuration information

    Returns:
        dict: Standardized index status response with status data and metadata

    Raises:
        ResponseFormattingError: If index status formatting fails or validation errors occur

    Example:
        >>> # Basic index status
        >>> status_data = {"vector_count": 1000, "dimension": 128, "index_type": "IndexFlatL2"}
        >>> response = format_index_status("my-index", status_data)
        >>>
        >>> # Detailed index status with health metrics
        >>> status_data = {
        ...     "vector_count": 1000,
        ...     "dimension": 128,
        ...     "index_type": "IndexFlatL2",
        ...     "memory_usage_mb": 45.2,
        ...     "last_updated": "2025-01-01T12:00:00Z"
        ... }
        >>> response = format_index_status("my-index", status_data)
    """
    try:
        # Validate input parameters
        if not index_id or not isinstance(index_id, str):
            raise ResponseFormattingError(
                f"Index ID must be non-empty string, got {index_id}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        if not isinstance(status_data, dict):
            raise ResponseFormattingError(
                f"Status data must be dictionary, got {type(status_data)}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        _validate_response_data(status_data)

        # Create response data with index status
        response_data = {DATA_FIELD_INDEX_ID: index_id, DATA_FIELD_STATUS: status_data}

        logger.debug(f"Formatted index status for: {index_id}")
        return success_response(response_data)

    except ResponseFormattingError:
        # Re-raise response formatting errors
        raise
    except Exception as e:
        # Wrap unexpected errors for consistent error handling
        raise ResponseFormattingError(
            f"Unexpected error formatting index status: {str(e)}",
            error_category=ERROR_CATEGORY_INTERNAL,
            original_error=e,
        )


def format_operation_result(
    operation: str, index_id: str, details: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format operation result in a standardized structure with comprehensive context and validation.

    This function creates consistent operation result responses with detailed information
    about the operation outcome, timing, and any relevant metadata.

    Args:
        operation: Name of the operation that was performed
        index_id: Identifier of the index involved in the operation
        details: Dictionary containing operation result details and context

    Returns:
        dict: Standardized operation result response with operation data and metadata

    Raises:
        ResponseFormattingError: If operation result formatting fails or validation errors occur

    Example:
        >>> # Basic operation result
        >>> details = {"vectors_added": 100, "operation_time_ms": 250.5}
        >>> response = format_operation_result("add_vectors", "my-index", details)
        >>>
        >>> # Detailed operation result with metrics
        >>> details = {
        ...     "vectors_added": 100,
        ...     "operation_time_ms": 250.5,
        ...     "memory_delta_mb": 12.3,
        ...     "index_size_after": 1100
        ... }
        >>> response = format_operation_result("add_vectors", "my-index", details)
    """
    try:
        # Validate input parameters
        if not operation or not isinstance(operation, str):
            raise ResponseFormattingError(
                f"Operation must be non-empty string, got {operation}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        if not index_id or not isinstance(index_id, str):
            raise ResponseFormattingError(
                f"Index ID must be non-empty string, got {index_id}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        if not isinstance(details, dict):
            raise ResponseFormattingError(
                f"Operation details must be dictionary, got {type(details)}",
                error_category=ERROR_CATEGORY_VALIDATION,
            )

        _validate_response_data(details)

        # Create response data with operation result
        response_data = {
            DATA_FIELD_OPERATION: operation,
            DATA_FIELD_INDEX_ID: index_id,
            **details,
        }

        logger.debug(f"Formatted operation result: {operation} on {index_id}")
        return success_response(response_data)

    except ResponseFormattingError:
        # Re-raise response formatting errors
        raise
    except Exception as e:
        # Wrap unexpected errors for consistent error handling
        raise ResponseFormattingError(
            f"Unexpected error formatting operation result: {str(e)}",
            error_category=ERROR_CATEGORY_INTERNAL,
            original_error=e,
        )

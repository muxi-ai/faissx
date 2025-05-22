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
FAISSx Server Response Module

This module provides standardized response formatting for the FAISSx server API.
It ensures consistent response structures across all API endpoints and
provides utility functions for creating success and error responses.
"""

import time
from typing import Any, Dict, List, Optional


def success_response(data: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a standardized success response.

    Args:
        data (dict, optional): Data to include in the response
        message (str, optional): Optional success message

    Returns:
        dict: Standardized success response
    """
    response = {
        "success": True,
        "timestamp": time.time(),
    }

    if message:
        response["message"] = message

    if data:
        response.update(data)

    return response


def error_response(error: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        error (str): Error message
        code (str, optional): Error code for categorization
        details (dict, optional): Additional error details

    Returns:
        dict: Standardized error response
    """
    response = {
        "success": False,
        "error": error,
        "timestamp": time.time(),
    }

    if code:
        response["error_code"] = code

    if details:
        response["error_details"] = details

    return response


def format_search_results(results: List[Dict[str, List[float]]], index_id: Optional[str] = None, query_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format search results in a standardized structure.

    Args:
        results: List of search results with distances and indices
        index_id: Optional ID of the index searched
        query_info: Optional query parameters info

    Returns:
        dict: Standardized search results response
    """
    response_data = {
        "results": results,
        "query_count": len(results)
    }

    if index_id:
        response_data["index_id"] = index_id

    if query_info:
        response_data.update(query_info)

    return success_response(response_data)


def format_vector_results(vectors: List[List[float]], index_id: Optional[str] = None, start_idx: Optional[int] = None, **metadata) -> Dict[str, Any]:
    """
    Format vector results in a standardized structure.

    Args:
        vectors: List of vectors
        index_id: Optional ID of the index
        start_idx: Optional starting index for paginated results
        **metadata: Additional metadata about the vectors

    Returns:
        dict: Standardized vector results response
    """
    response_data = {
        "vectors": vectors,
        "count": len(vectors)
    }

    if index_id:
        response_data["index_id"] = index_id

    if start_idx is not None:
        response_data["start_idx"] = start_idx

    if metadata:
        response_data.update(metadata)

    return success_response(response_data)


def format_index_status(index_id: str, status_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format index status in a standardized structure.

    Args:
        index_id: ID of the index
        status_data: Status data for the index

    Returns:
        dict: Standardized index status response
    """
    return success_response({
        "index_id": index_id,
        "status": status_data
    })


def format_operation_result(operation: str, index_id: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format operation result in a standardized structure.

    Args:
        operation: Name of the operation performed
        index_id: ID of the index
        details: Details about the operation result

    Returns:
        dict: Standardized operation result response
    """
    return success_response({
        "operation": operation,
        "index_id": index_id,
        **details
    })

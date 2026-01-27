#!/usr/bin/env python3
#
# FAISSx Server Hybrid Search Module
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
Hybrid Search Engine for FAISSx

This module provides comprehensive hybrid search capabilities that combine vector similarity
search with metadata filtering and intelligent result re-ranking. It enables sophisticated
search workflows that go beyond pure vector similarity to incorporate business logic,
metadata attributes, and custom scoring functions.

Key Features:
- Metadata storage and filtering system with complex query support
- Multiple re-ranking algorithms (linear combination, reciprocal rank fusion, custom scoring)
- Flexible filter conditions with AND/OR/NOT logic and comparison operators
- Performance-optimized search with configurable result expansion
- Type-safe interfaces with comprehensive error handling

Hybrid Search Workflow:
1. Vector Similarity: Initial FAISS search to find candidate vectors
2. Metadata Filtering: Apply complex filters to candidate results
3. Result Re-ranking: Apply sophisticated scoring algorithms to final ranking
4. Result Aggregation: Format and return structured results

Use Cases:
- E-commerce search with price/category/availability filters
- Document search with date/author/topic constraints
- Recommendation systems with user preference weighting
- Content discovery with freshness and popularity scoring

Integration:
This module integrates seamlessly with FAISSx server infrastructure to provide
enterprise-grade hybrid search capabilities with monitoring and debugging support.
"""

import ast
import logging  # Structured logging for monitoring and debugging
import math
import operator
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)  # Type hints for better code safety

import numpy as np  # Numerical operations for vector processing and scoring

# Logging configuration for hybrid search operations
logger = logging.getLogger("faissx.server")

# Constants for hybrid search configuration and optimization
# These constants provide sensible defaults and prevent magic numbers throughout the code

# Search expansion parameters
DEFAULT_FILTER_EXPANSION_FACTOR = (
    3  # Multiply k by this factor when filtering is enabled
)
MAX_FILTER_EXPANSION_FACTOR = 10  # Maximum expansion to prevent excessive memory usage
MIN_FILTER_EXPANSION_K = 50  # Minimum expanded k for effective filtering

# Re-ranking algorithm parameters
DEFAULT_VECTOR_WEIGHT = (
    0.7  # Default weight for vector similarity in linear combination
)
DEFAULT_METADATA_WEIGHT = 0.3  # Default weight for metadata score in linear combination
DEFAULT_RRF_K = 60  # Default k parameter for Reciprocal Rank Fusion
MIN_RRF_K = 1  # Minimum RRF k value
MAX_RRF_K = 1000  # Maximum practical RRF k value

# Filter operation constants
FILTER_OPERATORS = {  # Supported filter operators
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "nin",
    "exists",
}
LOGICAL_OPERATORS = {"AND", "OR", "NOT"}  # Supported logical operators

# Metadata field validation
MAX_METADATA_FIELDS = 100  # Maximum number of metadata fields per store
MAX_FIELD_NAME_LENGTH = 100  # Maximum length for metadata field names

# Performance and safety limits
DEFAULT_CUSTOM_SCORE_TIMEOUT = 1.0  # Timeout for custom scoring function evaluation
MAX_METADATA_SIZE_BYTES = 10240  # Maximum size for individual metadata objects


def _create_error_response(error_message: str) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.

    Args:
        error_message: Description of the error

    Returns:
        dict: Standardized error response
    """
    return {"success": False, "error": error_message}


def _create_success_response(**kwargs) -> Dict[str, Any]:
    """
    Create a standardized success response dictionary.

    Args:
        **kwargs: Additional fields to include in the response

    Returns:
        dict: Standardized success response
    """
    response = {"success": True}
    response.update(kwargs)
    return response


def _validate_vector_ids(vector_ids: List[int]) -> Optional[str]:
    """
    Validate vector IDs list for proper format and values.

    Args:
        vector_ids: List of vector IDs to validate

    Returns:
        str: Error message if validation fails, None if valid
    """
    for vid in vector_ids:
        if not isinstance(vid, int) or vid < 0:
            return f"Invalid vector ID: {vid} (must be non-negative integer)"
    return None


class MetadataStore:
    """
    Enterprise-grade metadata storage and filtering system for vector databases.

    This class provides comprehensive metadata management for FAISS indices, supporting
    complex filtering operations, field validation, and efficient storage. It serves as
    the foundation for hybrid search capabilities in the FAISSx system.

    Key Features:
        - Multi-index metadata management with complete isolation
        - Complex filtering with AND/OR/NOT logic and comparison operators
        - Field tracking and validation for schema management
        - Memory-efficient storage with configurable size limits
        - Type-safe operations with comprehensive error handling

    Storage Structure:
        stores[index_id] = {
            "metadata": {vector_id: {field: value, ...}, ...},
            "index_metadata": {global_index_properties},
            "fields": {set_of_all_field_names}
        }

    Thread Safety:
        This class is not thread-safe. External synchronization is required
        for concurrent access in multi-threaded environments.
    """

    def __init__(self) -> None:
        """
        Initialize an empty metadata store.

        Creates the internal storage structure for managing metadata across
        multiple indices with complete isolation between tenants/indices.
        """
        # Storage structure: index_id -> metadata store configuration
        # Each index gets its own isolated metadata namespace
        self.stores: Dict[str, Dict[str, Any]] = {}
        logger.debug("Initialized empty MetadataStore")

    def create_store(self, index_id: str) -> None:
        """
        Create a new metadata store for an index with validation.

        This method initializes a new metadata namespace for the specified index,
        providing complete isolation from other indices. It creates the necessary
        data structures for efficient metadata storage and field tracking.

        Args:
            index_id: Unique identifier for the index (must be non-empty string)

        Raises:
            ValueError: If index_id is invalid or already exists

        Example:
            >>> store = MetadataStore()
            >>> store.create_store("products_index")
        """
        # Validate index_id parameter
        if not isinstance(index_id, str) or not index_id.strip():
            raise ValueError("index_id must be a non-empty string")

        if index_id in self.stores:
            logger.warning(f"Metadata store for index {index_id} already exists")
            return

        # Initialize the metadata store structure
        self.stores[index_id] = {
            "metadata": {},  # Dict of vector_id -> metadata dictionary
            "index_metadata": {},  # Global properties for the entire index
            "fields": set(),  # Set of all metadata field names used
        }

        logger.info(f"Created metadata store for index: {index_id}")

    def add_metadata(
        self, index_id: str, vector_ids: List[int], metadata_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add metadata for vectors in an index with comprehensive validation.

        This method associates metadata dictionaries with specific vector IDs within
        an index. It automatically creates the index store if it doesn't exist and
        performs extensive validation to ensure data integrity.

        Args:
            index_id: Unique identifier for the target index
            vector_ids: List of vector IDs (must be non-negative integers)
            metadata_list: List of metadata dictionaries (one per vector)

        Returns:
            dict: Operation result with success status and statistics
                - success (bool): Whether the operation completed successfully
                - count (int): Number of metadata items added
                - fields (List[str]): Updated list of all metadata fields
                - error (str): Error description if operation failed

        Raises:
            ValueError: If inputs are invalid or constraints are violated

        Example:
            >>> store.add_metadata("idx1", [1, 2], [{"type": "doc"}, {"type": "img"}])
        """
        # Ensure the metadata store exists
        if index_id not in self.stores:
            self.create_store(index_id)

        store = self.stores[index_id]

        # Comprehensive input validation
        if len(vector_ids) != len(metadata_list):
            error_msg = (
                f"Vector IDs count ({len(vector_ids)}) doesn't match "
                f"metadata items count ({len(metadata_list)})"
            )
            logger.error(f"Metadata addition failed for {index_id}: {error_msg}")
            return _create_error_response(error_msg)

        # Validate vector IDs using helper function
        vector_id_error = _validate_vector_ids(vector_ids)
        if vector_id_error:
            return _create_error_response(vector_id_error)

        # Process each metadata item with field validation
        added_fields = set()
        for i, vid in enumerate(vector_ids):
            metadata = metadata_list[i]

            # Validate metadata structure
            if not isinstance(metadata, dict):
                error_msg = f"Metadata at index {i} must be a dictionary"
                return _create_error_response(error_msg)

            # Check metadata size limits
            metadata_str = str(metadata)
            if len(metadata_str.encode("utf-8")) > MAX_METADATA_SIZE_BYTES:
                error_msg = f"Metadata too large for vector {vid}"
                return _create_error_response(error_msg)

            # Store the metadata
            store["metadata"][vid] = metadata

            # Track field names with validation
            for field_name in metadata.keys():
                if len(field_name) > MAX_FIELD_NAME_LENGTH:
                    logger.warning(f"Field name too long: {field_name[:50]}...")
                    continue
                added_fields.add(field_name)
                store["fields"].add(field_name)

        # Check total field count limit
        if len(store["fields"]) > MAX_METADATA_FIELDS:
            logger.warning(
                f"Index {index_id} has {len(store['fields'])} fields "
                f"(exceeds recommended limit of {MAX_METADATA_FIELDS})"
            )

        logger.info(f"Added metadata for {len(vector_ids)} vectors in index {index_id}")
        return {
            "success": True,
            "count": len(vector_ids),
            "fields": list(store["fields"]),
            "new_fields": list(added_fields),
        }

    def get_metadata(self, index_id: str, vector_ids: List[int]) -> Dict[str, Any]:
        """
        Get metadata for vectors in an index.

        Args:
            index_id: ID of the index
            vector_ids: List of vector IDs

        Returns:
            dict: Result containing metadata
        """
        if index_id not in self.stores:
            return {
                "success": False,
                "error": f"No metadata store exists for index {index_id}",
            }

        store = self.stores[index_id]
        result = []

        for vid in vector_ids:
            if vid in store["metadata"]:
                result.append(store["metadata"][vid])
            else:
                result.append({})  # Empty metadata for missing IDs

        return {"success": True, "metadata": result, "count": len(result)}

    def delete_metadata(
        self, index_id: str, vector_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Delete metadata for vectors in an index.

        Args:
            index_id: ID of the index
            vector_ids: List of vector IDs to delete (None to delete all)

        Returns:
            dict: Result information
        """
        if index_id not in self.stores:
            return {
                "success": False,
                "error": f"No metadata store exists for index {index_id}",
            }

        store = self.stores[index_id]

        if vector_ids is None:
            # Delete all metadata
            count = len(store["metadata"])
            store["metadata"] = {}
        else:
            # Delete specified IDs
            count = 0
            for vid in vector_ids:
                if vid in store["metadata"]:
                    del store["metadata"][vid]
                    count += 1

        # Recalculate fields
        if len(store["metadata"]) == 0:
            store["fields"] = set()
        else:
            store["fields"] = set()
            for metadata in store["metadata"].values():
                for field in metadata.keys():
                    store["fields"].add(field)

        return {"success": True, "count": count, "remaining": len(store["metadata"])}

    def delete_store(self, index_id: str) -> Dict[str, Any]:
        """
        Delete an entire metadata store.

        Args:
            index_id: ID of the index

        Returns:
            dict: Result information
        """
        if index_id not in self.stores:
            return {
                "success": False,
                "error": f"No metadata store exists for index {index_id}",
            }

        count = len(self.stores[index_id]["metadata"])
        del self.stores[index_id]

        return {"success": True, "count": count}

    def filter_results(
        self,
        index_id: str,
        indices: List[List[int]],
        distances: List[List[float]],
        filter_conditions: Dict[str, Any],
    ) -> Tuple[List[List[int]], List[List[float]], List[List[Dict[str, Any]]]]:
        """
        Filter search results based on metadata conditions.

        Args:
            index_id: ID of the index
            indices: List of lists of vector indices from search results
            distances: List of lists of distances from search results
            filter_conditions: Dictionary of filter conditions

        Returns:
            tuple: (filtered_indices, filtered_distances, filtered_metadata)
        """
        if index_id not in self.stores:
            # Return empty results if no metadata store exists
            return [], [], []

        store = self.stores[index_id]

        # Prepare the filter function
        filter_fn = self._build_filter_function(filter_conditions)

        filtered_indices = []
        filtered_distances = []
        filtered_metadata = []

        # Apply filter to each query result
        for _q_idx, (query_indices, query_distances) in enumerate(
            zip(indices, distances)
        ):
            f_indices = []
            f_distances = []
            f_metadata = []

            # Filter each result for this query
            for i, idx in enumerate(query_indices):
                if idx == -1:  # Skip -1 indices (no match)
                    continue

                # Get metadata for this vector
                metadata = store["metadata"].get(int(idx), {})

                # Apply filter
                if filter_fn(metadata):
                    f_indices.append(idx)
                    f_distances.append(query_distances[i])
                    f_metadata.append(metadata)

            filtered_indices.append(f_indices)
            filtered_distances.append(f_distances)
            filtered_metadata.append(f_metadata)

        return filtered_indices, filtered_distances, filtered_metadata

    def _build_filter_function(self, filter_conditions: Dict[str, Any]) -> Callable:
        """
        Build a filter function from filter conditions.

        Args:
            filter_conditions: Dictionary of filter conditions

        Returns:
            function: Filter function that takes metadata and returns bool
        """
        if not filter_conditions:
            # No filtering
            return lambda _: True

        # Handle AND condition (default)
        if "AND" in filter_conditions:
            conditions = filter_conditions["AND"]
            sub_filters = [
                self._build_filter_function({k: v}) for k, v in conditions.items()
            ]
            return lambda metadata: all(f(metadata) for f in sub_filters)

        # Handle OR condition
        if "OR" in filter_conditions:
            conditions = filter_conditions["OR"]
            sub_filters = [
                self._build_filter_function({k: v}) for k, v in conditions.items()
            ]
            return lambda metadata: any(f(metadata) for f in sub_filters)

        # Handle NOT condition
        if "NOT" in filter_conditions:
            sub_filter = self._build_filter_function(filter_conditions["NOT"])
            return lambda metadata: not sub_filter(metadata)

        # Handle basic field conditions
        field_filters = []
        for field, condition in filter_conditions.items():
            if field in ("AND", "OR", "NOT"):
                continue

            # Build a filter for this field
            if isinstance(condition, dict):
                # Complex condition (operators)
                for op, value in condition.items():
                    if op == "eq":
                        field_filters.append(
                            lambda md, f=field, v=value: f in md and md[f] == v
                        )
                    elif op == "ne":
                        field_filters.append(
                            lambda md, f=field, v=value: f not in md or md[f] != v
                        )
                    elif op == "gt":
                        field_filters.append(
                            lambda md, f=field, v=value: f in md and md[f] > v
                        )
                    elif op == "gte":
                        field_filters.append(
                            lambda md, f=field, v=value: f in md and md[f] >= v
                        )
                    elif op == "lt":
                        field_filters.append(
                            lambda md, f=field, v=value: f in md and md[f] < v
                        )
                    elif op == "lte":
                        field_filters.append(
                            lambda md, f=field, v=value: f in md and md[f] <= v
                        )
                    elif op == "in":
                        field_filters.append(
                            lambda md, f=field, v=value: f in md and md[f] in v
                        )
                    elif op == "nin":
                        field_filters.append(
                            lambda md, f=field, v=value: f not in md or md[f] not in v
                        )
                    elif op == "exists":
                        field_filters.append(
                            lambda md, f=field, v=value: (f in md) == v
                        )
            else:
                # Simple equality condition
                field_filters.append(
                    lambda md, f=field, v=condition: f in md and md[f] == v
                )

        # Combine all field filters with AND
        return lambda metadata: all(f(metadata) for f in field_filters)


def rerank_results(
    distances: List[List[float]],
    indices: List[List[int]],
    metadata: List[List[Dict[str, Any]]],
    rerank_config: Dict[str, Any],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Re-rank search results based on additional criteria.

    Args:
        distances: List of lists of distances from search results
        indices: List of lists of vector indices from search results
        metadata: List of lists of metadata for search results
        rerank_config: Configuration for re-ranking

    Returns:
        tuple: (reranked_indices, reranked_distances)
    """
    method = rerank_config.get("method", "linear_combination")

    if method == "linear_combination":
        return _rerank_linear_combination(distances, indices, metadata, rerank_config)
    elif method == "reciprocal_rank_fusion":
        return _rerank_reciprocal_rank_fusion(
            distances, indices, metadata, rerank_config
        )
    elif method == "custom_score":
        return _rerank_custom_score(distances, indices, metadata, rerank_config)
    else:
        # Default: return original ranking
        return indices, distances


def _rerank_linear_combination(
    distances: List[List[float]],
    indices: List[List[int]],
    metadata: List[List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Re-rank using linear combination of vector distance and metadata score.

    Args:
        distances: List of lists of distances from search results
        indices: List of lists of vector indices from search results
        metadata: List of lists of metadata for search results
        config: Configuration for re-ranking

    Returns:
        tuple: (reranked_indices, reranked_scores)
    """
    # Get weights and metadata field
    vector_weight = config.get("vector_weight", 0.7)
    metadata_weight = config.get("metadata_weight", 0.3)
    metadata_field = config.get("metadata_field", "score")

    # Normalize to ensure weights sum to 1
    total_weight = vector_weight + metadata_weight
    vector_weight /= total_weight
    metadata_weight /= total_weight

    # Check if we need to invert distances (higher is better)
    invert_distance = config.get("invert_distance", True)

    reranked_indices = []
    reranked_scores = []

    # Use helper function for common loop pattern
    def _process_query_metadata(
        distances: List[List[float]],
        indices: List[List[int]],
        metadata: List[List[Dict[str, Any]]],
    ):
        """Helper to handle the common query processing pattern."""
        return enumerate(zip(distances, indices, metadata))

    for _q_idx, (
        query_distances,
        query_indices,
        query_metadata,
    ) in _process_query_metadata(distances, indices, metadata):
        # Initialize new scores
        new_scores = []

        # Calculate combined scores
        for _i, (dist, _idx, md) in enumerate(
            zip(query_distances, query_indices, query_metadata)
        ):
            # Process vector distance
            if invert_distance:
                # Convert distance to similarity score (higher is better)
                # Use 1/(1+d) to keep the score in [0,1] range
                vector_score = 1.0 / (1.0 + dist)
            else:
                # Use distance directly (lower is better)
                vector_score = -dist

            # Get metadata score
            metadata_score = md.get(metadata_field, 0.0)

            # Combine scores
            combined_score = (vector_weight * vector_score) + (
                metadata_weight * metadata_score
            )
            new_scores.append(combined_score)

        # Sort by combined score (descending)
        sorted_indices = np.argsort(new_scores)[::-1]

        # Rearrange results
        r_indices = [query_indices[i] for i in sorted_indices]
        r_scores = [new_scores[i] for i in sorted_indices]

        reranked_indices.append(r_indices)
        reranked_scores.append(r_scores)

    return reranked_indices, reranked_scores


def _rerank_reciprocal_rank_fusion(
    distances: List[List[float]],
    indices: List[List[int]],
    metadata: List[List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Re-rank using Reciprocal Rank Fusion (RRF).

    Args:
        distances: List of lists of distances from search results
        indices: List of lists of vector indices from search results
        metadata: List of lists of metadata for search results
        config: Configuration for re-ranking

    Returns:
        tuple: (reranked_indices, reranked_scores)
    """
    # Get parameters
    k = config.get("k", DEFAULT_RRF_K)  # Use constant instead of magic number
    metadata_field = config.get("metadata_field", "score")

    reranked_indices = []
    reranked_scores = []

    for _q_idx, (query_distances, query_indices, query_metadata) in enumerate(
        zip(distances, indices, metadata)
    ):
        # Create rankings for vector similarity
        vector_ranks = {}
        for i, (_dist, idx) in enumerate(zip(query_distances, query_indices)):
            vector_ranks[idx] = i + 1  # 1-based rank

        # Create rankings for metadata score
        # Sort metadata by score (descending)
        metadata_items = [
            (i, md.get(metadata_field, 0.0)) for i, md in enumerate(query_metadata)
        ]
        metadata_items.sort(key=lambda x: x[1], reverse=True)

        metadata_ranks = {}
        for rank, (i, _) in enumerate(metadata_items):
            metadata_ranks[query_indices[i]] = rank + 1  # 1-based rank

        # Compute RRF scores
        rrf_scores = {}
        for idx in query_indices:
            v_rank = vector_ranks.get(idx, len(query_indices))
            m_rank = metadata_ranks.get(idx, len(query_indices))

            # RRF formula: 1/(k + rank)
            rrf_score = (1.0 / (k + v_rank)) + (1.0 / (k + m_rank))
            rrf_scores[idx] = rrf_score

        # Sort by RRF score (descending)
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Extract results
        r_indices = [idx for idx, _ in sorted_items]
        r_scores = [score for _, score in sorted_items]

        reranked_indices.append(r_indices)
        reranked_scores.append(r_scores)

    return reranked_indices, reranked_scores


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "log": math.log,
    "sqrt": math.sqrt,
    "exp": math.exp,
}

_MAX_FORMULA_LEN = 512
_MAX_AST_DEPTH = 20


def _safe_eval(expr_str: str, variables: Dict[str, float]) -> float:
    """Evaluate a simple arithmetic expression without using eval().

    Only numeric literals, variable names present in *variables*,
    basic arithmetic operators (+, -, *, /, //, %, **), unary +/-,
    and a small whitelist of math functions are allowed.

    Raises ValueError for anything else (attribute access, calls to
    unknown functions, comprehensions, lambdas, etc.).
    """
    if len(expr_str) > _MAX_FORMULA_LEN:
        raise ValueError("Formula too long")

    tree = ast.parse(expr_str, mode="eval")

    def _eval_node(node: ast.AST, depth: int = 0) -> float:
        if depth > _MAX_AST_DEPTH:
            raise ValueError("Expression too deeply nested")

        if isinstance(node, ast.Expression):
            return _eval_node(node.body, depth + 1)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        if isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            raise ValueError(f"Unknown variable: {node.id}")

        if isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = _eval_node(node.left, depth + 1)
            right = _eval_node(node.right, depth + 1)
            return float(op_fn(left, right))

        if isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return float(op_fn(_eval_node(node.operand, depth + 1)))

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
                args = [_eval_node(a, depth + 1) for a in node.args]
                if node.keywords:
                    raise ValueError("Keyword arguments not allowed")
                return float(_SAFE_FUNCS[node.func.id](*args))
            raise ValueError("Unsupported function call")

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    return _eval_node(tree)


def _rerank_custom_score(
    distances: List[List[float]],
    indices: List[List[int]],
    metadata: List[List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Re-rank using a custom scoring function on metadata fields.

    The formula is evaluated with a safe AST-based parser that only allows
    numeric literals, named variables, basic arithmetic (+, -, *, /, //, %,
    **), and a small set of math functions (abs, min, max, log, sqrt, exp).

    Args:
        distances: List of lists of distances from search results
        indices: List of lists of vector indices from search results
        metadata: List of lists of metadata for search results
        config: Configuration for re-ranking

    Returns:
        tuple: (reranked_indices, reranked_scores)
    """
    formula = config.get("formula", "")
    if not formula:
        return indices, distances

    # Validate formula with sample values before using it
    try:
        test_vars: Dict[str, float] = {"distance": 0.5}
        test_vars.update({f"md_{field}": 1.0 for field in config.get("fields", [])})
        _safe_eval(formula, test_vars)
    except Exception as e:
        logger.error(f"Error validating custom score formula: {str(e)}")
        return indices, distances

    reranked_indices = []
    reranked_scores = []

    for _q_idx, (query_distances, query_indices, query_metadata) in enumerate(
        zip(distances, indices, metadata)
    ):
        custom_scores = []

        for _i, (dist, idx, md) in enumerate(
            zip(query_distances, query_indices, query_metadata)
        ):
            vars_dict: Dict[str, float] = {"distance": dist}
            for field in config.get("fields", []):
                vars_dict[f"md_{field}"] = md.get(field, 0.0)

            try:
                score = _safe_eval(formula, vars_dict)
                custom_scores.append((idx, score))
            except Exception as e:
                logger.error(f"Error evaluating custom score: {str(e)}")
                custom_scores.append((idx, -999999))

        # Sort by custom score (descending)
        custom_scores.sort(key=lambda x: x[1], reverse=True)

        # Extract results
        r_indices = [idx for idx, _ in custom_scores]
        r_scores = [score for _, score in custom_scores]

        reranked_indices.append(r_indices)
        reranked_scores.append(r_scores)

    return reranked_indices, reranked_scores


def hybrid_search(
    index: Any,
    query_vectors: List[List[float]],
    k: int,
    metadata_store: MetadataStore,
    index_id: str,
    filter_conditions: Optional[Dict[str, Any]] = None,
    rerank_config: Optional[Dict[str, Any]] = None,
    search_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform hybrid search combining vector similarity, metadata filtering, and re-ranking.

    Args:
        index: FAISS index to search
        query_vectors: List of query vectors
        k: Number of results to return
        metadata_store: MetadataStore instance
        index_id: ID of the index
        filter_conditions: Metadata filter conditions
        rerank_config: Re-ranking configuration
        search_params: Search parameters for FAISS

    Returns:
        dict: Search results
    """
    # Convert query vectors to numpy array
    query_np = np.array(query_vectors, dtype=np.float32)

    # Apply search parameters if provided
    if search_params:
        for param_name, param_value in search_params.items():
            if hasattr(index, "set_" + param_name):
                getattr(index, "set_" + param_name)(param_value)

    # Get more results than requested if filtering is enabled
    search_k = k
    if filter_conditions:
        # Get more results since some might be filtered out
        # (3x is a heuristic, can be adjusted)
        search_k = min(k * 3, index.ntotal)

    # Perform the search
    distances, indices = index.search(query_np, search_k)

    # Apply metadata filtering if requested
    if filter_conditions:
        indices, distances, metadata = metadata_store.filter_results(
            index_id, indices, distances, filter_conditions
        )

        # Trim to requested k if needed
        for i in range(len(indices)):
            if len(indices[i]) > k:
                indices[i] = indices[i][:k]
                distances[i] = distances[i][:k]
                metadata[i] = metadata[i][:k]
    else:
        # Get metadata for results without filtering
        metadata = []
        for query_indices in indices:
            query_metadata = []
            for idx in query_indices:
                if idx == -1:  # -1 indicates no match
                    query_metadata.append({})
                else:
                    md = (
                        metadata_store.stores.get(index_id, {})
                        .get("metadata", {})
                        .get(int(idx), {})
                    )
                    query_metadata.append(md)
            metadata.append(query_metadata)

    # Apply re-ranking if requested
    if rerank_config:
        indices, distances = rerank_results(distances, indices, metadata, rerank_config)

    # Format the results
    results = []
    for i in range(len(query_vectors)):
        query_results = []
        for j in range(min(k, len(indices[i]))):
            if j < len(indices[i]):
                idx = indices[i][j]
                if idx != -1:  # Skip -1 indices (no match)
                    dist = distances[i][j]
                    md = (
                        metadata[i][j]
                        if i < len(metadata) and j < len(metadata[i])
                        else {}
                    )

                    query_results.append(
                        {"id": int(idx), "distance": float(dist), "metadata": md}
                    )

        results.append(query_results)

    return {
        "success": True,
        "results": results,
        "num_queries": len(query_vectors),
        "k": k,
        "filtered": filter_conditions is not None,
        "reranked": rerank_config is not None,
    }

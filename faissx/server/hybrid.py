#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
FAISSx Server Hybrid Search Module

This module provides functionality for hybrid search operations that combine
vector similarity with metadata filtering and result re-ranking.
"""

import json
import faiss
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

logger = logging.getLogger("faissx.server")


class MetadataStore:
    """
    Storage and filtering system for vector metadata.

    This class manages metadata associated with vectors in FAISS indices
    and provides filtering capabilities for hybrid search operations.
    """

    def __init__(self):
        """Initialize an empty metadata store."""
        self.stores = {}  # Dict of index_id -> metadata

    def create_store(self, index_id: str) -> None:
        """
        Create a new metadata store for an index.

        Args:
            index_id: ID of the index
        """
        if index_id not in self.stores:
            self.stores[index_id] = {
                "metadata": {},  # Dict of vector_id -> metadata
                "index_metadata": {},  # Metadata about the index itself
                "fields": set()  # Set of all metadata fields used
            }

    def add_metadata(
        self,
        index_id: str,
        vector_ids: List[int],
        metadata_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add metadata for vectors in an index.

        Args:
            index_id: ID of the index
            vector_ids: List of vector IDs
            metadata_list: List of metadata dictionaries

        Returns:
            dict: Result information
        """
        if index_id not in self.stores:
            self.create_store(index_id)

        store = self.stores[index_id]

        # Validate inputs
        if len(vector_ids) != len(metadata_list):
            return {
                "success": False,
                "error": f"Number of vector IDs ({len(vector_ids)}) doesn't match number of metadata items ({len(metadata_list)})"
            }

        # Add metadata for each vector
        for i, vid in enumerate(vector_ids):
            metadata = metadata_list[i]
            store["metadata"][vid] = metadata

            # Update fields set
            for field in metadata.keys():
                store["fields"].add(field)

        return {
            "success": True,
            "count": len(vector_ids),
            "fields": list(store["fields"])
        }

    def get_metadata(
        self,
        index_id: str,
        vector_ids: List[int]
    ) -> Dict[str, Any]:
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
                "error": f"No metadata store exists for index {index_id}"
            }

        store = self.stores[index_id]
        result = []

        for vid in vector_ids:
            if vid in store["metadata"]:
                result.append(store["metadata"][vid])
            else:
                result.append({})  # Empty metadata for missing IDs

        return {
            "success": True,
            "metadata": result,
            "count": len(result)
        }

    def delete_metadata(
        self,
        index_id: str,
        vector_ids: Optional[List[int]] = None
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
                "error": f"No metadata store exists for index {index_id}"
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

        return {
            "success": True,
            "count": count,
            "remaining": len(store["metadata"])
        }

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
                "error": f"No metadata store exists for index {index_id}"
            }

        count = len(self.stores[index_id]["metadata"])
        del self.stores[index_id]

        return {
            "success": True,
            "count": count
        }

    def filter_results(
        self,
        index_id: str,
        indices: List[List[int]],
        distances: List[List[float]],
        filter_conditions: Dict[str, Any]
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
        for q_idx, (query_indices, query_distances) in enumerate(zip(indices, distances)):
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
            sub_filters = [self._build_filter_function({k: v}) for k, v in conditions.items()]
            return lambda metadata: all(f(metadata) for f in sub_filters)

        # Handle OR condition
        if "OR" in filter_conditions:
            conditions = filter_conditions["OR"]
            sub_filters = [self._build_filter_function({k: v}) for k, v in conditions.items()]
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
                        field_filters.append(lambda md, f=field, v=value: f in md and md[f] == v)
                    elif op == "ne":
                        field_filters.append(lambda md, f=field, v=value: f not in md or md[f] != v)
                    elif op == "gt":
                        field_filters.append(lambda md, f=field, v=value: f in md and md[f] > v)
                    elif op == "gte":
                        field_filters.append(lambda md, f=field, v=value: f in md and md[f] >= v)
                    elif op == "lt":
                        field_filters.append(lambda md, f=field, v=value: f in md and md[f] < v)
                    elif op == "lte":
                        field_filters.append(lambda md, f=field, v=value: f in md and md[f] <= v)
                    elif op == "in":
                        field_filters.append(lambda md, f=field, v=value: f in md and md[f] in v)
                    elif op == "nin":
                        field_filters.append(lambda md, f=field, v=value: f not in md or md[f] not in v)
                    elif op == "exists":
                        field_filters.append(lambda md, f=field, v=value: (f in md) == v)
            else:
                # Simple equality condition
                field_filters.append(lambda md, f=field, v=condition: f in md and md[f] == v)

        # Combine all field filters with AND
        return lambda metadata: all(f(metadata) for f in field_filters)


def rerank_results(
    distances: List[List[float]],
    indices: List[List[int]],
    metadata: List[List[Dict[str, Any]]],
    rerank_config: Dict[str, Any]
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
        return _rerank_linear_combination(
            distances, indices, metadata, rerank_config
        )
    elif method == "reciprocal_rank_fusion":
        return _rerank_reciprocal_rank_fusion(
            distances, indices, metadata, rerank_config
        )
    elif method == "custom_score":
        return _rerank_custom_score(
            distances, indices, metadata, rerank_config
        )
    else:
        # Default: return original ranking
        return indices, distances


def _rerank_linear_combination(
    distances: List[List[float]],
    indices: List[List[int]],
    metadata: List[List[Dict[str, Any]]],
    config: Dict[str, Any]
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

    for q_idx, (query_distances, query_indices, query_metadata) in enumerate(zip(distances, indices, metadata)):
        # Initialize new scores
        new_scores = []

        # Calculate combined scores
        for i, (dist, idx, md) in enumerate(zip(query_distances, query_indices, query_metadata)):
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
            combined_score = (vector_weight * vector_score) + (metadata_weight * metadata_score)
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
    config: Dict[str, Any]
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
    k = config.get("k", 60)  # RRF constant
    metadata_field = config.get("metadata_field", "score")

    reranked_indices = []
    reranked_scores = []

    for q_idx, (query_distances, query_indices, query_metadata) in enumerate(zip(distances, indices, metadata)):
        # Create rankings for vector similarity
        vector_ranks = {}
        for i, (dist, idx) in enumerate(zip(query_distances, query_indices)):
            vector_ranks[idx] = i + 1  # 1-based rank

        # Create rankings for metadata score
        # Sort metadata by score (descending)
        metadata_items = [(i, md.get(metadata_field, 0.0)) for i, md in enumerate(query_metadata)]
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


def _rerank_custom_score(
    distances: List[List[float]],
    indices: List[List[int]],
    metadata: List[List[Dict[str, Any]]],
    config: Dict[str, Any]
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Re-rank using a custom scoring function on metadata fields.

    Args:
        distances: List of lists of distances from search results
        indices: List of lists of vector indices from search results
        metadata: List of lists of metadata for search results
        config: Configuration for re-ranking

    Returns:
        tuple: (reranked_indices, reranked_scores)
    """
    # Get formula and fields
    formula = config.get("formula", "")
    if not formula:
        # Default to original ranking if no formula provided
        return indices, distances

    # Parse and validate formula (basic validation only)
    try:
        # Simply test the formula with sample values
        test_vars = {"distance": 0.5}
        test_vars.update({f"md_{field}": 1.0 for field in config.get("fields", [])})
        eval(formula, {"__builtins__": None}, test_vars)
    except Exception as e:
        logger.error(f"Error validating custom score formula: {str(e)}")
        return indices, distances

    reranked_indices = []
    reranked_scores = []

    for q_idx, (query_distances, query_indices, query_metadata) in enumerate(zip(distances, indices, metadata)):
        # Calculate custom scores
        custom_scores = []

        for i, (dist, idx, md) in enumerate(zip(query_distances, query_indices, query_metadata)):
            # Prepare variables for formula
            vars_dict = {"distance": dist}

            # Add metadata fields
            for field in config.get("fields", []):
                vars_dict[f"md_{field}"] = md.get(field, 0.0)

            # Evaluate formula
            try:
                score = eval(formula, {"__builtins__": None}, vars_dict)
                custom_scores.append((idx, score))
            except Exception as e:
                logger.error(f"Error evaluating custom score: {str(e)}")
                custom_scores.append((idx, -999999))  # Rank at the end in case of error

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
    search_params: Optional[Dict[str, Any]] = None
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
                    md = metadata_store.stores.get(index_id, {}).get("metadata", {}).get(int(idx), {})
                    query_metadata.append(md)
            metadata.append(query_metadata)

    # Apply re-ranking if requested
    if rerank_config:
        indices, distances = rerank_results(
            distances, indices, metadata, rerank_config
        )

    # Format the results
    results = []
    for i in range(len(query_vectors)):
        query_results = []
        for j in range(min(k, len(indices[i]))):
            if j < len(indices[i]):
                idx = indices[i][j]
                if idx != -1:  # Skip -1 indices (no match)
                    dist = distances[i][j]
                    md = metadata[i][j] if i < len(metadata) and j < len(metadata[i]) else {}

                    query_results.append({
                        "id": int(idx),
                        "distance": float(dist),
                        "metadata": md
                    })

        results.append(query_results)

    return {
        "success": True,
        "results": results,
        "num_queries": len(query_vectors),
        "k": k,
        "filtered": filter_conditions is not None,
        "reranked": rerank_config is not None
    }

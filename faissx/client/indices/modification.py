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
FAISSx index modification implementation.

This module provides functions for modifying indices, including:
- Merging multiple indices into one
- Splitting an index into multiple smaller indices

The implementation allows for both local and remote modes:
- In local mode, it leverages FAISS's native functionality
- In remote mode, it reconstructs and redistributes vectors appropriately
"""

import logging
import numpy as np
import time
from typing import List, Any, Optional, Callable, Dict, TypeVar

try:
    import faiss
except ImportError:
    faiss = None

from ..client import get_client
from .flat import IndexFlatL2
from .ivf_flat import IndexIVFFlat
from .hnsw_flat import IndexHNSWFlat
from .pq import IndexPQ
from .ivf_pq import IndexIVFPQ
from .scalar_quantizer import IndexScalarQuantizer
from .id_map import IndexIDMap, IndexIDMap2
from .factory import index_factory
from .base import FAISSxBaseIndex

logger = logging.getLogger(__name__)

# Define type for any FAISSx index
FAISSxIndexType = TypeVar('FAISSxIndexType', bound=FAISSxBaseIndex)


def _parse_server_response(response: Any, default_value: Any) -> Any:
    """
    Parse server response with consistent error handling.

    Args:
        response: Server response to parse
        default_value: Default value to use if response isn't a dict

    Returns:
        Parsed value from response or default value
    """
    if isinstance(response, dict):
        return response.get("index_id", default_value)
    elif isinstance(response, str):
        # Sometimes the server returns just a string
        return response
    else:
        logger.warning(f"Unexpected server response format: {response}")
        return default_value


def _get_vectors_from_index(index: FAISSxBaseIndex) -> Optional[np.ndarray]:
    """
    Extract vectors from an index with robust error handling.

    Attempts to use cached vectors, reconstruct from the index,
    or fetch from server as appropriate.

    Args:
        index: FAISSx index to extract vectors from

    Returns:
        NumPy array of vectors or None if extraction fails
    """
    # Check for cached vectors first
    if hasattr(index, "_cached_vectors") and index._cached_vectors is not None:
        logger.debug(f"Using cached vectors from index {getattr(index, 'name', 'unknown')}")
        return index._cached_vectors

    # Next try get_vectors if available
    if hasattr(index, "get_vectors"):
        try:
            start_time = time.time()
            vectors = index.get_vectors()
            elapsed = time.time() - start_time
            logger.debug(f"Retrieved vectors using get_vectors in {elapsed:.2f}s")
            return vectors
        except Exception as e:
            logger.warning(f"Failed to get vectors: {e}")

    # Fallback to reconstruct method
    try:
        start_time = time.time()
        vectors: List[np.ndarray] = []

        # Try reconstruct_n first (more efficient)
        if hasattr(index, "reconstruct_n") and index.ntotal > 0:
            try:
                vectors = index.reconstruct_n(0, index.ntotal)
                elapsed = time.time() - start_time
                logger.debug(
                    f"Retrieved {len(vectors)} vectors using reconstruct_n in {elapsed:.2f}s"
                )
                return vectors
            except Exception as e:
                logger.warning(
                    f"Failed to use reconstruct_n: {e}, "
                    f"falling back to individual reconstruction"
                )

        # Fallback to individual reconstruction
        if hasattr(index, "reconstruct"):
            vectors_list: List[np.ndarray] = []
            for i in range(index.ntotal):
                try:
                    vectors_list.append(index.reconstruct(i))
                except Exception:
                    # Skip any vectors that can't be reconstructed
                    continue

            if vectors_list:
                vectors_array = np.vstack(vectors_list)
                elapsed = time.time() - start_time
                logger.debug(
                    f"Retrieved {len(vectors_list)} vectors using individual "
                    f"reconstruct in {elapsed:.2f}s"
                )
                return vectors_array
    except Exception as e:
        logger.warning(f"Failed to reconstruct vectors: {e}")

    logger.warning("Could not extract vectors from index")
    return None


def merge_indices(
    indices: List[FAISSxBaseIndex],
    output_type: Optional[str] = None,
    id_map: bool = False,
    id_map2: bool = False,
    batch_size: int = 10000,
) -> FAISSxBaseIndex:
    """
    Merge multiple indices into a single index.

    Args:
        indices: List of FAISSx indices to merge
        output_type: Optional type of the output index as a FAISS-compatible string
                    description (if None, use the same type as the first index)
        id_map: Whether to wrap the result in an IndexIDMap for custom IDs
        id_map2: Whether to wrap the result in an IndexIDMap2 for updatable vectors
                with custom IDs (takes precedence over id_map if both are True)
        batch_size: Size of vector batches when processing large indices

    Returns:
        A new FAISSx index containing all vectors from the input indices

    Raises:
        ValueError: If indices are incompatible or empty
    """
    # Start timing the operation
    start_time = time.time()

    # Validate input indices
    if not indices:
        raise ValueError("No indices provided for merging")

    # Ensure all indices have the same dimensionality
    d = indices[0].d
    for i, idx in enumerate(indices[1:], 1):
        if idx.d != d:
            raise ValueError(
                f"Dimension mismatch: index 0 has {d} dimensions, "
                f"but index {i} has {idx.d} dimensions"
            )

    # Check if we should use remote mode
    client = get_client()
    use_remote = client is not None and client.mode == "remote"

    # If remote mode, check if server supports merge operation directly
    if use_remote:
        try:
            # Check if all indices are remote and have index_id
            remote_indices = all(hasattr(idx, "index_id") for idx in indices)
            if remote_indices:
                logger.info("Attempting to merge indices directly on server")
                index_ids = [idx.index_id for idx in indices]
                remote_output_type = output_type or _get_index_type_description(indices[0])

                try:
                    response = client.merge_indices(index_ids, remote_output_type)
                    merged_index_id = _parse_server_response(response, None)

                    if merged_index_id:
                        # Server successfully merged indices
                        logger.info(f"Server merged indices into {merged_index_id}")
                        merged_index = index_factory(d, remote_output_type)
                        merged_index.index_id = merged_index_id
                        merged_index.name = merged_index_id

                        # Apply IDMap wrapper if needed
                        if id_map or id_map2:
                            wrapper_class = IndexIDMap2 if id_map2 else IndexIDMap
                            merged_index = wrapper_class(merged_index)

                        elapsed = time.time() - start_time
                        logger.info(f"Remote merge completed in {elapsed:.2f}s")
                        return merged_index
                except Exception as e:
                    logger.warning(
                        f"Server-side merge failed: {e}, falling back to client-side merge"
                    )
        except Exception as e:
            logger.warning(f"Error checking remote merge capability: {e}")

    # Determine output index type and ID mapping settings
    if output_type is None:
        # Use the same type as the first index if not specified
        if isinstance(indices[0], (IndexIDMap, IndexIDMap2)):
            base_index = indices[0].index
            first_type = _get_index_type_description(base_index)
            # Set appropriate id_map/id_map2 flags based on the first index
            id_map = True
            id_map2 = isinstance(indices[0], IndexIDMap2)
        else:
            first_type = _get_index_type_description(indices[0])
        output_type = first_type

    # Determine if we need ID mapping
    id_mappings: List[Dict[int, int]] = []
    needs_id_mapping = id_map or id_map2 or bool(id_mappings)

    # Create the target index
    if needs_id_mapping:
        # Create empty base index first
        base_index = index_factory(d, output_type)
        # Create the appropriate wrapper around the empty base index
        wrapper_class = IndexIDMap2 if id_map2 else IndexIDMap
        merged_index = wrapper_class(base_index)
    else:
        merged_index = index_factory(d, output_type)

    # Prepare to collect all vectors and IDs if needed
    all_vectors: List[np.ndarray] = []
    total_vectors = 0

    # Process each source index
    for i, idx in enumerate(indices):
        logger.debug(f"Processing index {i+1}/{len(indices)}")
        # Handle IDMap/IDMap2 wrappers
        id_map_data: Optional[Dict[int, int]] = None

        if isinstance(idx, (IndexIDMap, IndexIDMap2)):
            # Collect ID mappings with adjusted indices for the merged index
            id_map_data = {}
            for internal_idx, external_id in idx._id_map.items():
                id_map_data[internal_idx + total_vectors] = external_id

        # Extract vectors if the index has any
        # IMPORTANT: Use idx directly, not base_index, because vectors are stored in the wrapper
        if getattr(idx, "ntotal", 0) > 0:
            vectors = _get_vectors_from_index(idx)

            if vectors is not None and len(vectors) > 0:
                all_vectors.append(vectors)
                if id_map_data:
                    id_mappings.append(id_map_data)
                total_vectors += len(vectors)

    # Return empty index if no vectors found
    if not all_vectors:
        logger.warning("No vectors found in the provided indices")
        return merged_index

    # Combine and add vectors to the merged index
    combined_vectors = np.vstack(all_vectors)
    logger.info(f"Merged {total_vectors} vectors from {len(indices)} indices")

    # Train the merged index if needed
    if hasattr(merged_index, "train") and not getattr(merged_index, "is_trained", True):
        logger.info(f"Training merged index with {len(combined_vectors)} vectors")
        merged_index.train(combined_vectors)

    # Add the vectors to the merged index in batches to avoid memory issues
    total_added = 0

    if needs_id_mapping and id_mappings:
        # We have ID mappings, so use add_with_ids
        combined_mappings: Dict[int, int] = {}
        for mapping in id_mappings:
            combined_mappings.update(mapping)

        # Create arrays of vectors and their corresponding IDs
        all_ids = []
        for i in range(len(combined_vectors)):
            if i in combined_mappings:
                all_ids.append(combined_mappings[i])
            else:
                all_ids.append(i)  # Use index as ID for vectors without explicit IDs

        # Add vectors with IDs in batches
        for i in range(0, len(combined_vectors), batch_size):
            batch_end = min(i + batch_size, len(combined_vectors))
            batch_vectors = combined_vectors[i:batch_end]
            batch_ids = np.array(all_ids[i:batch_end])
            merged_index.add_with_ids(batch_vectors, batch_ids)
            total_added += len(batch_vectors)
            logger.debug(
                f"Added batch of {len(batch_vectors)} vectors with IDs "
                f"({total_added}/{len(combined_vectors)})"
            )
    else:
        # No ID mappings, use regular add
        for i in range(0, len(combined_vectors), batch_size):
            batch = combined_vectors[i:i + batch_size]
            merged_index.add(batch)
            total_added += len(batch)
            logger.debug(
                f"Added batch of {len(batch)} vectors ({total_added}/{len(combined_vectors)})"
            )

    elapsed = time.time() - start_time
    logger.info(f"Merge completed in {elapsed:.2f}s")
    return merged_index


def split_index(
    index: FAISSxBaseIndex,
    num_parts: int = 2,
    split_method: str = "sequential",
    custom_split_fn: Optional[Callable[[np.ndarray], List[int]]] = None,
    output_type: Optional[str] = None,
    preserve_ids: bool = True,
    batch_size: int = 10000,
) -> List[FAISSxBaseIndex]:
    """
    Split an index into multiple smaller indices.

    Args:
        index: The FAISSx index to split
        num_parts: Number of parts to split into
        split_method: Method to use for splitting:
            - 'sequential': Simple sequential partitioning
            - 'cluster': Use k-means clustering to group similar vectors
            - 'custom': Use the provided custom_split_fn function
        custom_split_fn: Custom function that takes a matrix of vectors and returns
                        a list of part indices for each vector (0 to num_parts-1)
        output_type: Optional type for the output indices (same as input if None)
        preserve_ids: Whether to preserve custom IDs for IndexIDMap/IndexIDMap2
        batch_size: Size of vector batches when processing large indices

    Returns:
        List of FAISSx indices containing the split vectors

    Raises:
        ValueError: If the index is empty or split_method is invalid
    """
    # Start timing the operation
    start_time = time.time()

    # Validate input index
    if getattr(index, "ntotal", 0) == 0:
        raise ValueError("Cannot split an empty index")

    # Get index properties
    d = index.d
    has_id_map = isinstance(index, (IndexIDMap, IndexIDMap2))
    is_id_map2 = isinstance(index, IndexIDMap2)
    base_index = index.index if has_id_map else index

    # Determine output index type
    if output_type is None:
        output_type = _get_index_type_description(base_index)

    # Check if we should use remote mode
    client = get_client()
    use_remote = client is not None and client.mode == "remote"

    # If remote mode, check if server supports split operation directly
    if use_remote:
        try:
            # Check if index is remote and has index_id
            if hasattr(index, "index_id"):
                logger.info("Attempting to split index directly on server")
                remote_output_type = output_type

                try:
                    response = client.split_index(
                        index.index_id,
                        num_parts,
                        split_method,
                        remote_output_type
                    )

                    if isinstance(response, dict) and "index_ids" in response:
                        # Server successfully split the index
                        split_index_ids = response["index_ids"]
                        logger.info(f"Server split index into {len(split_index_ids)} parts")

                        result_indices = []
                        for split_id in split_index_ids:
                            split_index = index_factory(d, remote_output_type)
                            split_index.index_id = split_id
                            split_index.name = split_id

                            # Apply IDMap wrapper if needed and preserve_ids is True
                            if has_id_map and preserve_ids:
                                wrapper_class = IndexIDMap2 if is_id_map2 else IndexIDMap
                                split_index = wrapper_class(split_index)

                            result_indices.append(split_index)

                        elapsed = time.time() - start_time
                        logger.info(f"Remote split completed in {elapsed:.2f}s")
                        return result_indices
                except Exception as e:
                    logger.warning(
                        f"Server-side split failed: {e}, falling back to client-side split"
                    )
        except Exception as e:
            logger.warning(f"Error checking remote split capability: {e}")

    # Extract vectors and IDs
    vectors = _get_vectors_from_index(index)
    id_mappings: Dict[int, int] = {}

    # If no vectors could be extracted, fail
    if vectors is None or len(vectors) == 0:
        raise ValueError("Could not extract vectors from the index")

    # Extract ID mappings if needed
    if has_id_map and preserve_ids:
        id_mappings = index._id_map

    logger.info(f"Extracted {len(vectors)} vectors for splitting")

    # Determine vector partitioning based on split method
    if split_method == "sequential":
        # Simple sequential partitioning
        part_size = len(vectors) // num_parts
        if part_size == 0:  # Handle case when num_parts > len(vectors)
            part_size = 1
        part_indices = [min(i // part_size, num_parts - 1) for i in range(len(vectors))]

    elif split_method == "cluster":
        # Use k-means clustering to group similar vectors
        try:
            if faiss is None:
                raise ImportError("FAISS is required for clustering split method")

            kmeans = faiss.Kmeans(d, num_parts, niter=20, verbose=False)
            logger.info("Training k-means for clustering-based split")
            kmeans.train(vectors)
            _, part_indices_array = kmeans.index.search(vectors, 1)
            part_indices = part_indices_array.flatten().tolist()
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise ValueError(f"Clustering failed: {e}")

    elif split_method == "custom":
        # Use custom function to determine the split
        if custom_split_fn is None:
            raise ValueError("Custom split method requires a custom_split_fn")

        try:
            logger.info("Using custom split function")
            part_indices = custom_split_fn(vectors)
            if len(part_indices) != len(vectors):
                raise ValueError(
                    f"Custom split function returned {len(part_indices)} indices "
                    f"but expected {len(vectors)}"
                )
            # Ensure all part indices are within range
            for idx in part_indices:
                if idx < 0 or idx >= num_parts:
                    raise ValueError(
                        f"Invalid part index {idx}, must be 0 to {num_parts - 1}"
                    )
        except Exception as e:
            logger.error(f"Error in custom split function: {e}")
            raise ValueError(f"Custom split function failed: {e}")

    else:
        raise ValueError(f"Unsupported split method: {split_method}")

    # Create the output indices
    result_indices: List[FAISSxBaseIndex] = []
    for _ in range(num_parts):
        # Create an index of the specified type
        result_indices.append(index_factory(d, output_type))

    # Group vectors by their part index
    grouped_vectors: List[List[np.ndarray]] = [[] for _ in range(num_parts)]
    grouped_ids: Optional[List[List[int]]] = (
        [[] for _ in range(num_parts)] if preserve_ids and has_id_map else None
    )

    for i, part_idx in enumerate(part_indices):
        grouped_vectors[part_idx].append(vectors[i])
        if preserve_ids and has_id_map and i in id_mappings:
            assert grouped_ids is not None  # For type checking
            grouped_ids[part_idx].append(id_mappings[i])

    # Add the vectors to each part
    for part_idx, part_vectors in enumerate(grouped_vectors):
        if not part_vectors:
            logger.warning(f"Part {part_idx} has no vectors")
            continue

        part_vectors_array = np.vstack(part_vectors)
        part_index = result_indices[part_idx]

        # Train if needed
        if hasattr(part_index, "train") and not getattr(part_index, "is_trained", True):
            logger.info(f"Training part {part_idx} with {len(part_vectors_array)} vectors")
            part_index.train(part_vectors_array)

        # Add the vectors in batches to avoid memory issues
        total_added = 0
        if preserve_ids and has_id_map and grouped_ids and grouped_ids[part_idx]:
            # Create IDMap/IDMap2 wrapper
            wrapper_class = IndexIDMap2 if is_id_map2 else IndexIDMap
            wrapped_index = wrapper_class(part_index)

            # Add with IDs in batches
            part_ids = np.array(grouped_ids[part_idx])
            for i in range(0, len(part_vectors_array), batch_size):
                batch_end = min(i + batch_size, len(part_vectors_array))
                batch = part_vectors_array[i:batch_end]
                batch_ids = part_ids[i:batch_end]
                wrapped_index.add_with_ids(batch, batch_ids)
                total_added += len(batch)
                logger.debug(
                    f"Added batch of {len(batch)} vectors to part {part_idx} "
                    f"({total_added}/{len(part_vectors_array)})"
                )

            result_indices[part_idx] = wrapped_index
        else:
            # Add without IDs in batches
            for i in range(0, len(part_vectors_array), batch_size):
                batch = part_vectors_array[i:i + batch_size]
                part_index.add(batch)
                total_added += len(batch)
                logger.debug(
                    f"Added batch of {len(batch)} vectors to part {part_idx} "
                    f"({total_added}/{len(part_vectors_array)})"
                )

    elapsed = time.time() - start_time
    logger.info(f"Split completed in {elapsed:.2f}s")
    return result_indices


def _get_index_type_description(index: FAISSxBaseIndex) -> str:
    """
    Get a FAISS-compatible string description for an index type.

    This function analyzes the index object and returns a string description that can be used
    with index_factory to create an index of the same type.

    Args:
        index: FAISSx index instance

    Returns:
        String description usable with index_factory
    """
    if isinstance(index, IndexFlatL2):
        return "Flat"
    elif isinstance(index, IndexIVFFlat):
        return f"IVF{index.nlist},Flat"
    elif isinstance(index, IndexHNSWFlat):
        return f"HNSW{index.m}"
    elif isinstance(index, IndexPQ):
        return f"PQ{index.m}x{index.nbits}"
    elif isinstance(index, IndexIVFPQ):
        return f"IVF{index.nlist},PQ{index.m}x{index.nbits}"
    elif isinstance(index, IndexScalarQuantizer):
        return "SQ8"
    else:
        # Default fallback
        return "Flat"

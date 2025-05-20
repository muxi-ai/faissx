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
from typing import List, Any, Optional, Callable

import faiss

from .flat import IndexFlatL2
from .ivf_flat import IndexIVFFlat
from .hnsw_flat import IndexHNSWFlat
from .pq import IndexPQ
from .ivf_pq import IndexIVFPQ
from .scalar_quantizer import IndexScalarQuantizer
from .id_map import IndexIDMap, IndexIDMap2
from .factory import index_factory

logger = logging.getLogger(__name__)


def merge_indices(
    indices: List[Any],
    output_type: Optional[str] = None,
    id_map: bool = False,
    id_map2: bool = False,
) -> Any:
    """
    Merge multiple indices into a single index.

    Args:
        indices: List of FAISSx indices to merge
        output_type: Optional type of the output index as a FAISS-compatible string description
                    (if None, use the same type as the first index)
        id_map: Whether to wrap the result in an IndexIDMap for custom IDs
        id_map2: Whether to wrap the result in an IndexIDMap2 for updatable vectors with custom IDs
                (takes precedence over id_map if both are True)

    Returns:
        A new FAISSx index containing all vectors from the input indices

    Raises:
        ValueError: If indices are incompatible or empty
    """
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

    # Determine the output index type
    if output_type is None:
        # Use the same type as the first index if not specified
        # Check if the first index is an IDMap/IDMap2 wrapper
        if isinstance(indices[0], (IndexIDMap, IndexIDMap2)):
            base_index = indices[0].index
            first_type = _get_index_type_description(base_index)
            # Set appropriate id_map/id_map2 flags based on the first index
            id_map = True
            id_map2 = isinstance(indices[0], IndexIDMap2)
        else:
            first_type = _get_index_type_description(indices[0])
        output_type = first_type

    # Create the target index
    merged_index = index_factory(d, output_type)

    # Prepare to collect all vectors and IDs if needed
    all_vectors = []
    id_mappings = []
    total_vectors = 0

    # Process each source index
    for idx in indices:
        # Handle IDMap/IDMap2 wrappers
        base_index = idx
        id_map_data = None

        if isinstance(idx, (IndexIDMap, IndexIDMap2)):
            base_index = idx.index
            # Collect ID mappings
            id_map_data = {}
            for internal_idx, external_id in idx._id_map.items():
                # Adjust the internal index for the merged index
                id_map_data[internal_idx + total_vectors] = external_id

        # If the index has vectors, extract them
        if getattr(base_index, 'ntotal', 0) > 0:
            # Extract vectors using reconstruct method
            vectors = []
            try:
                for i in range(base_index.ntotal):
                    # Skip any vectors that might have been removed internally
                    try:
                        vectors.append(base_index.reconstruct(i))
                    except Exception:
                        continue
            except AttributeError:
                # If reconstruct isn't available, try to get vectors directly
                raise ValueError(
                    f"Index of type {type(base_index).__name__} doesn't support reconstruction "
                    "and cannot be merged"
                )

            if vectors:
                vectors = np.vstack(vectors)
                all_vectors.append(vectors)
                # Keep track of ID mappings if needed
                if id_map_data:
                    id_mappings.append(id_map_data)
                # Update the total vector count
                total_vectors += len(vectors)

    # If no vectors were found, return an empty index
    if not all_vectors:
        logger.warning("No vectors found in the provided indices")
        return merged_index

    # Combine all vectors
    combined_vectors = np.vstack(all_vectors)

    # Train the merged index if needed
    if hasattr(merged_index, 'train') and not getattr(merged_index, 'is_trained', True):
        merged_index.train(combined_vectors)

    # Add the vectors to the merged index
    merged_index.add(combined_vectors)

    # If we need to create an IDMap wrapper, do it now
    if id_map or id_map2 or id_mappings:
        # Create the appropriate wrapper
        wrapper_class = IndexIDMap2 if id_map2 else IndexIDMap
        wrapped_index = wrapper_class(merged_index)

        # If we have ID mappings from the source indices, restore them
        if id_mappings:
            # Combine all ID mappings
            combined_mappings = {}
            for mapping in id_mappings:
                combined_mappings.update(mapping)

            # Set up the ID mappings in the wrapped index
            wrapped_index._id_map = combined_mappings
            wrapped_index._rev_id_map = {v: k for k, v in combined_mappings.items()}
            wrapped_index.ntotal = merged_index.ntotal

        return wrapped_index

    return merged_index


def split_index(
    index: Any,
    num_parts: int = 2,
    split_method: str = 'sequential',
    custom_split_fn: Optional[Callable[[np.ndarray], List[int]]] = None,
    output_type: Optional[str] = None,
    preserve_ids: bool = True,
) -> List[Any]:
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

    Returns:
        List of FAISSx indices containing the split vectors

    Raises:
        ValueError: If the index is empty or split_method is invalid
    """
    # Check if index is empty
    if getattr(index, 'ntotal', 0) == 0:
        raise ValueError("Cannot split an empty index")

    # Get the dimensionality of the index
    d = index.d

    # Determine if the input index uses custom IDs
    has_id_map = isinstance(index, (IndexIDMap, IndexIDMap2))
    is_id_map2 = isinstance(index, IndexIDMap2)
    base_index = index.index if has_id_map else index

    # Determine the output index type
    if output_type is None:
        output_type = _get_index_type_description(base_index)

    # Extract vectors and IDs
    vectors = []
    id_mappings = {}

    if has_id_map and preserve_ids:
        # For IDMap indices, extract vectors using the ID map
        for i in range(index.ntotal):
            if i in index._id_map:  # Only include vectors that aren't removed
                external_id = index._id_map[i]
                vector = index.reconstruct(external_id)
                vectors.append(vector)
                id_mappings[len(vectors) - 1] = external_id
    else:
        # For regular indices, extract all vectors
        try:
            for i in range(base_index.ntotal):
                try:
                    vectors.append(base_index.reconstruct(i))
                except Exception:
                    # Skip any vectors that can't be reconstructed
                    continue
        except AttributeError:
            raise ValueError(
                f"Index of type {type(base_index).__name__} doesn't support reconstruction "
                "and cannot be split"
            )

    # Convert vectors to a numpy array
    vectors = np.vstack(vectors) if vectors else np.zeros((0, d))

    # Determine which vectors go into which part
    if split_method == 'sequential':
        # Simple sequential partitioning
        part_size = len(vectors) // num_parts
        part_indices = []
        for i in range(len(vectors)):
            part_indices.append(min(i // part_size, num_parts - 1))

    elif split_method == 'cluster':
        # Use k-means clustering to group similar vectors
        try:
            # Create a small flat index for clustering
            kmeans = faiss.Kmeans(d, num_parts, niter=20, verbose=False)
            kmeans.train(vectors)
            _, part_indices = kmeans.index.search(vectors, 1)
            part_indices = part_indices.flatten()
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise ValueError(f"Clustering failed: {e}")

    elif split_method == 'custom':
        # Use custom function to determine the split
        if custom_split_fn is None:
            raise ValueError("Custom split method requires a custom_split_fn")

        try:
            part_indices = custom_split_fn(vectors)
            if len(part_indices) != len(vectors):
                raise ValueError(
                    f"Custom split function returned {len(part_indices)} indices "
                    f"but expected {len(vectors)}"
                )
            # Ensure all part indices are within range
            for idx in part_indices:
                if idx < 0 or idx >= num_parts:
                    raise ValueError(f"Invalid part index {idx}, must be 0 to {num_parts - 1}")
        except Exception as e:
            logger.error(f"Error in custom split function: {e}")
            raise ValueError(f"Custom split function failed: {e}")

    else:
        raise ValueError(f"Unsupported split method: {split_method}")

    # Create the output indices
    result_indices = []
    for _ in range(num_parts):
        # Create an index of the specified type
        result_indices.append(index_factory(d, output_type))

    # Group vectors by their part index
    grouped_vectors = [[] for _ in range(num_parts)]
    grouped_ids = [[] for _ in range(num_parts)] if preserve_ids and has_id_map else None

    for i, part_idx in enumerate(part_indices):
        grouped_vectors[part_idx].append(vectors[i])
        if preserve_ids and has_id_map and i in id_mappings:
            grouped_ids[part_idx].append(id_mappings[i])

    # Add the vectors to each part
    for part_idx, part_vectors in enumerate(grouped_vectors):
        if not part_vectors:
            logger.warning(f"Part {part_idx} has no vectors")
            continue

        part_vectors = np.vstack(part_vectors)
        part_index = result_indices[part_idx]

        # Train if needed
        if hasattr(part_index, 'train') and not getattr(part_index, 'is_trained', True):
            part_index.train(part_vectors)

        # Add the vectors
        if preserve_ids and has_id_map and grouped_ids and grouped_ids[part_idx]:
            # Create IDMap/IDMap2 wrapper
            wrapper_class = IndexIDMap2 if is_id_map2 else IndexIDMap
            wrapped_index = wrapper_class(part_index)
            # Add with IDs
            part_ids = np.array(grouped_ids[part_idx])
            wrapped_index.add_with_ids(part_vectors, part_ids)
            result_indices[part_idx] = wrapped_index
        else:
            # Add without IDs
            part_index.add(part_vectors)

    return result_indices


def _get_index_type_description(index: Any) -> str:
    """
    Get a FAISS-compatible string description for an index type.

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

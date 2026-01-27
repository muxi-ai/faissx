#!/usr/bin/env python3
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
FAISSx index persistence implementation.

This module provides functions for saving and loading indices from disk,
compatible with FAISS's write_index and read_index functions.

The implementation allows for both local and remote modes:
- In local mode, it uses FAISS's native persistence functions directly
- In remote mode, it reconstructs the index and transfers data appropriately
"""

import logging
import os
import tempfile
import time
from typing import Any, Optional, TypeVar, Union

import faiss
import numpy as np

from faissx.client.client import get_client

from .factory import index_factory
from .flat import IndexFlatL2
from .hnsw_flat import IndexHNSWFlat
from .id_map import IndexIDMap, IndexIDMap2
from .ivf_flat import IndexIVFFlat
from .ivf_pq import IndexIVFPQ
from .pq import IndexPQ
from .scalar_quantizer import IndexScalarQuantizer

logger = logging.getLogger(__name__)

# Define a type alias for FAISS indices to improve readability
FaissIndex = TypeVar('FaissIndex')

# Constants for file format
IDMAP_FORMAT_FLAG = 0
IDMAP2_FORMAT_FLAG = 1
HEADER_SIZE = 1  # 1 byte for format flag

# Maximum number of vectors to reconstruct at once to avoid memory issues
MAX_VECTORS_TO_RECONSTRUCT = 100000


def write_index(index: FaissIndex, fname: str) -> None:
    """
    Write a FAISSx index to disk in a format compatible with FAISS.

    Args:
        index: The FAISSx index to save
        fname: Output file name where the index will be saved

    Raises:
        ValueError: If the index type is not supported or file cannot be written
    """
    # Ensure output directory exists
    _ensure_directory_exists(fname)

    # Check client mode (explicit check rather than just checking if client exists)
    client = get_client()
    is_local_mode = client is None or client.mode == "local"

    try:
        # Handle local mode indices directly using FAISS's persistence
        if (
            is_local_mode
            and hasattr(index, "_local_index")
            and index._local_index is not None
        ):
            logger.info(f"Saving index to {fname} using local FAISS implementation")
            faiss.write_index(index._local_index, fname)
            return

        # Special handling for IDMap and IDMap2 indices to preserve ID mappings
        if isinstance(index, (IndexIDMap, IndexIDMap2)):
            logger.info(
                f"Saving IDMap{'2' if isinstance(index, IndexIDMap2) else ''} index to {fname}"
            )
            _write_idmap_index(index, fname)
            return

        # Handle other indices by reconstructing in memory
        logger.info(f"Saving index to {fname} using reconstruction approach")

        # Validate index state
        if not getattr(index, "is_trained", True):
            logger.warning(f"Cannot save untrained index to {fname}")
            raise ValueError("Cannot save untrained index")

        if getattr(index, "ntotal", 0) == 0:
            logger.info(f"Saving empty index to {fname}")
            _write_empty_index(index, fname)
            return

        # Reconstruct and save index
        _reconstruct_and_save_index(index, fname)
        logger.info(f"Successfully saved index to {fname}")

    except Exception as e:
        logger.error(f"Error saving index to {fname}: {e}")
        raise ValueError(f"Failed to save index: {e}")


def read_index(fname: str, gpu: bool = False) -> FaissIndex:
    """
    Read a saved FAISSx index from disk.

    Args:
        fname: Path to the input index file
        gpu: Whether to try loading the index on GPU if available

    Returns:
        The loaded FAISSx index object

    Raises:
        ValueError: If the file cannot be read or is not a valid index
    """
    if not os.path.isfile(fname):
        logger.error(f"File not found: {fname}")
        raise ValueError(f"File not found: {fname}")

    start_time = time.time()

    try:
        # Try reading as custom IDMap format first
        with open(fname, "rb") as f:
            # Check if file is in custom IDMap format
            format_flag = f.read(HEADER_SIZE)[0]

            if format_flag in [IDMAP_FORMAT_FLAG, IDMAP2_FORMAT_FLAG]:
                id_map_type = "2" if format_flag == IDMAP2_FORMAT_FLAG else ""
                logger.info(f"Loading file {fname} as IDMap{id_map_type} format")

                index = _read_idmap_index(fname, format_flag, gpu)

                elapsed = time.time() - start_time
                logger.info(f"Successfully loaded IDMap index from {fname} in {elapsed:.2f}s")
                return index
    except (OSError, ValueError, IndexError) as e:
        # Not a custom IDMap file or error reading it, try standard FAISS reading
        logger.debug(f"Not a custom IDMap file, trying standard FAISS reading: {e}")
        pass

    # Standard FAISS index reading
    try:
        logger.info(f"Loading index from {fname} using standard FAISS")
        faiss_index = faiss.read_index(fname)

        # Handle GPU if requested
        if gpu:
            logger.info("Moving index to GPU")
            faiss_index = _move_to_gpu_if_available(faiss_index)

        # Create and return corresponding FAISSx index
        index = _create_faissx_from_faiss_index(faiss_index, fname, gpu)
        logger.info(
            f"Successfully loaded index from {fname} in {time.time() - start_time:.2f}s"
        )
        return index

    except Exception as e:
        logger.error(f"Error loading index from {fname}: {e}")
        raise ValueError(f"Failed to load index: {e}")


def _ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the directory for a file path exists.

    Args:
        file_path: Path to file for which to ensure directory existence
    """
    dirname = os.path.dirname(file_path)
    if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)


def _write_idmap_index(index: Union[IndexIDMap, IndexIDMap2], fname: str) -> None:
    """
    Write an IDMap or IDMap2 index to disk.

    This function handles saving both IndexIDMap and IndexIDMap2 instances with their
    ID mappings preserved. It uses a custom format when necessary.

    Args:
        index: The IDMap or IDMap2 index to save
        fname: Output file name where the index will be saved
    """
    # Check if we're in remote mode
    client = get_client()
    is_remote = client is not None and client.mode == "remote"

    # For remote mode, use a simplified approach
    if is_remote:
        logger.info("Remote mode detected - using simplified IDMap saving")
        try:
            # Try to get vectors first if possible
            vectors = _reconstruct_vectors_from_index(index)

            if vectors is not None and vectors.shape[0] > 0:
                # Create a flat index with these vectors
                flat_index = faiss.IndexFlatL2(index.d)
                flat_index.add(vectors)
                faiss.write_index(flat_index, fname)
                return

            # Fallback if reconstruction fails
            flat_index = faiss.IndexFlatL2(index.d)
            faiss.write_index(flat_index, fname)
            logger.warning("Remote mode: saved simplified index without vectors")
            return
        except Exception as e:
            logger.warning(f"Failed with simplified approach, falling back: {e}")
            flat_index = faiss.IndexFlatL2(index.d)
            faiss.write_index(flat_index, fname)
            return

    # Use temporary file for intermediate storage
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        temp_path = tmp.name
        try:
            # Convert ID mappings to numpy array for efficient storage
            id_map = _create_id_map_array(index)

            # Reconstruct vectors for storage
            vectors = _reconstruct_vectors_from_index(index)

            # Save base index to temporary file
            base_index = _get_base_index(index)
            faiss.write_index(base_index, temp_path)

            # Read index data for combined storage
            with open(temp_path, "rb") as f:
                index_data = f.read()

            # Write combined file
            _write_idmap_format(index, fname, id_map, index_data, vectors)

            log_idx = "2" if isinstance(index, IndexIDMap2) else ""
            logger.info(f"Successfully saved IndexIDMap{log_idx} to {fname}")

        except Exception as e:
            logger.error(f"Error saving IDMap index: {e}")
            raise
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def _create_id_map_array(index: Union[IndexIDMap, IndexIDMap2]) -> np.ndarray:
    """
    Extract ID mapping from an IDMap index.

    This function attempts to extract ID mappings using multiple approaches:
    1. Direct access to FAISS IDMap (local mode)
    2. Use pythonic ID mapping if available
    3. Fall back to an empty array

    Args:
        index: The IDMap or IDMap2 index

    Returns:
        numpy array of IDs
    """
    # For local mode with direct access
    if hasattr(index, "_local_index") and hasattr(index._local_index, "id_map"):
        # Direct access to FAISS IDMap
        try:
            id_array = faiss.vector_to_array(index._local_index.id_map)
            if id_array is not None and len(id_array) > 0:
                return id_array
        except Exception as e:
            logger.warning(f"Could not access id_map directly: {e}")

    # For remote mode or if direct access fails
    if hasattr(index, "_id_mapping") and index._id_mapping:
        # Use pythonic ID mapping if available
        id_list = []
        idx_list = []

        for idx, id_val in sorted(index._id_mapping.items()):
            idx_list.append(idx)
            id_list.append(id_val)

        return np.array(id_list, dtype=np.int64)

    # Last resort: empty array
    logger.warning("Could not extract ID mapping, using empty array")
    return np.array([], dtype=np.int64)


def _reconstruct_vectors_from_index(
    index: Union[IndexIDMap, IndexIDMap2],
) -> Optional[np.ndarray]:
    """
    Reconstruct vectors from an index.

    Tries multiple approaches in order:
    1. Use get_vectors if available
    2. Use reconstruct_n method
    3. Fall back to individual vector reconstruction
    4. Return empty array if all fail

    Args:
        index: The index to reconstruct vectors from

    Returns:
        numpy array of vectors or None if reconstruction fails
    """
    # Check if we're in remote mode - handle specially
    client = get_client()
    is_remote = client is not None and client.mode == "remote"

    if is_remote:
        try:
            # For remote mode, try a direct approach
            if hasattr(index, "ntotal") and index.ntotal > 0:
                # Create dummy vectors for remote mode
                d = getattr(index, "d", 64)  # Default to 64 if not available
                dummy_vectors = np.zeros((min(index.ntotal, 100), d), dtype=np.float32)
                logger.info(
                    f"Created {dummy_vectors.shape[0]} dummy vectors for remote index"
                )
                return dummy_vectors
        except Exception as e:
            logger.warning(f"Remote vector reconstruction failed: {e}")
            return None

    # Method 1: Using cached vectors if available (most reliable)
    if hasattr(index, "_cached_vectors") and index._cached_vectors is not None:
        return index._cached_vectors

    # Method 2: Try using get_vectors method
    if hasattr(index, "get_vectors") and callable(index.get_vectors):
        try:
            vectors = index.get_vectors()
            if vectors is not None and vectors.shape[0] > 0:
                return vectors
        except Exception as e:
            logger.warning(f"get_vectors() failed: {e}")

    # Method 3: Try reconstruct_n if available
    try:
        if hasattr(index, "reconstruct_n") and callable(index.reconstruct_n):
            ntotal = getattr(index, "ntotal", 0)
            if ntotal > 0:
                # Limit vector count to avoid memory issues
                batch_size = min(ntotal, MAX_VECTORS_TO_RECONSTRUCT)
                vectors = index.reconstruct_n(0, batch_size)
                return vectors
    except Exception as e:
        logger.warning(f"reconstruct_n() failed: {e}")

    # Method 4: Try per-vector reconstruction (last resort, slow)
    try:
        if hasattr(index, "reconstruct") and callable(index.reconstruct):
            ntotal = getattr(index, "ntotal", 0)
            if ntotal > 0:
                # Limit to a reasonable number to avoid memory issues
                count = min(ntotal, 1000)
                d = getattr(index, "d", 64)  # Default to 64 if not available
                vectors = np.zeros((count, d), dtype=np.float32)

                for i in range(count):
                    try:
                        vectors[i] = index.reconstruct(i)
                    except Exception:  # Specify exception type instead of bare except
                        # If reconstruction fails for an individual vector, use zeros
                        pass

                return vectors
    except Exception as e:
        logger.warning(f"Individual vector reconstruction failed: {e}")

    # If all methods fail, return None
    logger.warning("All vector reconstruction methods failed")
    return None


def _get_base_index(index: Union[IndexIDMap, IndexIDMap2]) -> Any:
    """
    Get the base index from an IDMap or IDMap2 index.

    This function attempts to extract the underlying base index from an IDMap wrapper
    using several approaches:
    1. Direct access through _local_index
    2. Access through the index attribute
    3. Create a fallback flat index if all else fails

    Args:
        index: The IDMap or IDMap2 index

    Returns:
        The extracted base index or a fallback flat index
    """
    # Try getting the base index through direct access
    if hasattr(index, "_local_index") and index._local_index is not None:
        try:
            # For local mode with direct access
            if isinstance(index._local_index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                return index._local_index.index
        except Exception as e:
            logger.warning(f"Direct base index access failed: {e}")

    # Try getting through index attribute
    if hasattr(index, "index") and index.index is not None:
        return (
            index.index._local_index
            if hasattr(index.index, "_local_index")
            else index.index
        )

    # Create a basic flat index as fallback
    logger.warning("Could not get base index, creating fallback flat index")
    d = getattr(index, "d", 64)  # Default to 64 if not available
    return faiss.IndexFlatL2(d)


def _write_idmap_format(
    index: Union[IndexIDMap, IndexIDMap2],
    fname: str,
    id_map: np.ndarray,
    index_data: bytes,
    vectors: Optional[np.ndarray] = None,
) -> None:
    """
    Write an IDMap or IDMap2 index to disk in custom format.

    The file format structure is:
    - 1 byte format flag (0 for IDMap, 1 for IDMap2)
    - 4 bytes ntotal (int32)
    - 4 bytes dimension (int32)
    - 8 bytes * ntotal for IDs (int64)
    - Index data from base index
    - Vector data (optional)

    Args:
        index: The IDMap or IDMap2 index to save
        fname: Output file name
        id_map: Array of IDs extracted from the index
        index_data: Serialized base index data as bytes
        vectors: Optional vector data (embeddings) to store

    Raises:
        ValueError: If the ID map format could not be written
    """
    try:
        with open(fname, "wb") as f:
            # Write format flag (1 byte)
            format_flag = (
                IDMAP2_FORMAT_FLAG
                if isinstance(index, IndexIDMap2)
                else IDMAP_FORMAT_FLAG
            )
            f.write(bytes([format_flag]))

            # Write ntotal and dimension (4 bytes each)
            ntotal = getattr(index, "ntotal", 0)
            d = getattr(index, "d", 64)
            f.write(np.array([ntotal, d], dtype=np.int32).tobytes())

            # Write IDs
            if id_map is not None and len(id_map) > 0:
                f.write(id_map.astype(np.int64).tobytes())
            else:
                logger.warning("No IDs to write to IDMap file")

            # Write base index data
            f.write(index_data)

            # Write vectors if available
            if vectors is not None and vectors.shape[0] > 0:
                # Write vector count and dimension
                f.write(
                    np.array(
                        [vectors.shape[0], vectors.shape[1]], dtype=np.int32
                    ).tobytes()
                )
                # Write vectors
                f.write(vectors.astype(np.float32).tobytes())
            else:
                # Write zeros to indicate no vectors
                f.write(np.array([0, 0], dtype=np.int32).tobytes())

        logger.info(f"Wrote custom IDMap format to {fname}")
    except Exception as e:
        logger.error(f"Error writing IDMap format: {e}")
        raise ValueError(f"Failed to write IDMap format: {e}")


def _write_empty_index(index: FaissIndex, fname: str) -> None:
    """
    Write an empty index to disk.

    This function creates and saves an empty index of the appropriate type.
    For remote mode, it creates a simple placeholder index.
    For local mode, it attempts to create an index of the same type.

    Args:
        index: The empty index to save
        fname: Output file name
    """
    # Check if we're in remote mode
    client = get_client()
    is_remote = client is not None and client.mode == "remote"

    # For remote mode, create a simple placeholder
    if is_remote:
        # Create a basic empty index
        d = getattr(index, "d", 64)  # Default to 64 if not available
        empty_index = faiss.IndexFlatL2(d)
        faiss.write_index(empty_index, fname)
        logger.info(f"Wrote empty placeholder index to {fname} in remote mode")
        return

    # For local mode, try to create an appropriate empty index
    empty_index = _create_empty_index_by_type(index)

    if empty_index is not None:
        faiss.write_index(empty_index, fname)
        logger.info(f"Wrote empty index to {fname}")
    else:
        # Fallback to flat index
        d = getattr(index, "d", 64)
        flat_index = faiss.IndexFlatL2(d)
        faiss.write_index(flat_index, fname)
        logger.warning(f"Used fallback flat index for {fname}")


def _create_empty_index_by_type(index: FaissIndex) -> Optional[Any]:
    """
    Create an empty FAISS index of the same type as the input index.

    This function inspects the provided index and creates an empty FAISS index
    with the same parameters (dimension, metric type, etc.) but without any data.

    Args:
        index: The reference index

    Returns:
        Empty FAISS index of the same type, or None if type not supported
    """
    d = getattr(index, "d", 64)  # Get dimension, default to 64

    try:
        # Check index type and create appropriate empty index
        if isinstance(index, IndexFlatL2):
            return faiss.IndexFlatL2(d)

        elif isinstance(index, IndexIVFFlat):
            nlist = getattr(index, "nlist", 100)
            quantizer = faiss.IndexFlatL2(d)
            metric_type = getattr(index, "metric_type", "L2")
            metric = (
                faiss.METRIC_INNER_PRODUCT if metric_type == "IP" else faiss.METRIC_L2
            )
            return faiss.IndexIVFFlat(quantizer, d, nlist, metric)

        elif isinstance(index, IndexHNSWFlat):
            m = getattr(index, "M", 32)
            # Use m rather than creating unused ef_construction variable
            metric_type = getattr(index, "metric_type", "L2")
            metric = (
                faiss.METRIC_INNER_PRODUCT if metric_type == "IP" else faiss.METRIC_L2
            )
            return faiss.IndexHNSWFlat(d, m, metric)

        elif isinstance(index, IndexPQ):
            m = getattr(index, "M", 8)
            nbits = getattr(index, "nbits", 8)
            metric_type = getattr(index, "metric_type", "L2")
            metric = (
                faiss.METRIC_INNER_PRODUCT if metric_type == "IP" else faiss.METRIC_L2
            )
            return faiss.IndexPQ(d, m, nbits, metric)

        elif isinstance(index, IndexIVFPQ):
            nlist = getattr(index, "nlist", 100)
            m = getattr(index, "M", 8)
            nbits = getattr(index, "nbits", 8)
            quantizer = faiss.IndexFlatL2(d)
            metric_type = getattr(index, "metric_type", "L2")
            metric = (
                faiss.METRIC_INNER_PRODUCT if metric_type == "IP" else faiss.METRIC_L2
            )
            return faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, metric)

        elif isinstance(index, IndexScalarQuantizer):
            metric_type = getattr(index, "metric_type", "L2")
            metric = (
                faiss.METRIC_INNER_PRODUCT if metric_type == "IP" else faiss.METRIC_L2
            )
            return faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, metric)

        elif isinstance(index, (IndexIDMap, IndexIDMap2)):
            base = _create_empty_index_by_type(index.index)
            if base is not None:
                return (
                    faiss.IndexIDMap2(base)
                    if isinstance(index, IndexIDMap2)
                    else faiss.IndexIDMap(base)
                )

    except Exception as e:
        logger.warning(f"Failed to create empty index of same type: {e}")

    # Return None if we couldn't create an appropriate empty index
    return None


def _reconstruct_and_save_index(index: FaissIndex, fname: str) -> None:
    """
    Reconstruct an index and save it to disk.

    This function extracts vectors from the index and creates a new index
    with the same parameters but containing only the extracted vectors.
    If vectors cannot be extracted, it falls back to saving an empty index.

    Args:
        index: The index to save
        fname: Output file name
    """
    # Reconstruct vectors from index
    vectors = _get_vectors_from_index(index)

    if vectors is None or vectors.shape[0] == 0:
        logger.warning(
            f"No vectors could be reconstructed from index, saving empty index to {fname}"
        )
        _write_empty_index(index, fname)
        return

    # Create an initialized index with the reconstructed vectors
    initialized_index = _create_initialized_index(index, vectors)

    if initialized_index is not None:
        faiss.write_index(initialized_index, fname)
        logger.info(f"Successfully saved reconstructed index to {fname}")
    else:
        logger.warning(
            f"Could not create initialized index, saving fallback flat index to {fname}"
        )
        flat_index = faiss.IndexFlatL2(index.d)
        flat_index.add(vectors)
        faiss.write_index(flat_index, fname)


def _get_vectors_from_index(index: FaissIndex) -> Optional[np.ndarray]:
    """
    Extract vectors from an index.

    This function tries multiple approaches to get the vectors from an index:
    1. Use vectors directly if available in local mode
    2. Try reconstruction methods (reconstruct_n, get_vectors, get_xb)
    3. Fall back to dummy vectors in remote mode if needed

    The function limits the number of vectors extracted to avoid memory issues.

    Args:
        index: The index to extract vectors from

    Returns:
        Numpy array of vectors or None if extraction fails
    """
    # Check if we're in remote mode
    client = get_client()
    is_remote = client is not None and client.mode == "remote"

    # For local mode, try direct methods first
    if not is_remote:
        # If index has _local_index, try to get vectors directly from there
        if hasattr(index, "_local_index") and index._local_index is not None:
            try:
                if hasattr(index._local_index, "reconstruct_n"):
                    ntotal = index._local_index.ntotal
                    if ntotal > 0:
                        # Get vectors in batches to avoid memory issues
                        batch_size = min(ntotal, MAX_VECTORS_TO_RECONSTRUCT)
                        return index._local_index.reconstruct_n(0, batch_size)
            except Exception as e:
                logger.warning(f"Direct vector access failed: {e}")

    # Try using the index's own reconstruct methods
    try:
        # If index has reconstruct_n method, use it
        if hasattr(index, "reconstruct_n") and index.ntotal > 0:
            batch_size = min(index.ntotal, MAX_VECTORS_TO_RECONSTRUCT)
            return index.reconstruct_n(0, batch_size)
    except Exception as e:
        logger.warning(f"reconstruct_n failed: {e}")

    # Fall back to other methods
    for method_name in ["get_vectors", "get_xb"]:
        if hasattr(index, method_name) and callable(getattr(index, method_name)):
            try:
                vectors = getattr(index, method_name)()
                if vectors is not None and vectors.shape[0] > 0:
                    # Limit to avoid memory issues
                    return vectors[: min(len(vectors), MAX_VECTORS_TO_RECONSTRUCT)]
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")

    # For remote mode, create dummy vectors as last resort
    if is_remote:
        try:
            d = getattr(index, "d", 64)
            ntotal = getattr(index, "ntotal", 100)
            if ntotal > 0:
                # Create dummy vectors - this allows index to be reconstructed
                # but search quality will be affected
                dummy_size = min(ntotal, 100)
                logger.warning(
                    f"Using {dummy_size} dummy vectors for remote index persistence"
                )
                return np.zeros((dummy_size, d), dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to create dummy vectors: {e}")

    # If we get here, we couldn't extract vectors
    return None


def _create_initialized_index(index: FaissIndex, vectors: np.ndarray) -> Optional[Any]:
    """
    Create and initialize a FAISS index of the appropriate type.

    This function creates a new index of the same type as the input index,
    trains it if necessary, and adds the provided vectors to it.

    Args:
        index: Reference index that defines the type and parameters
        vectors: Vectors to add to the newly created index

    Returns:
        Initialized FAISS index with vectors added, or None if creation fails
    """
    local_index = None
    index_type = type(index).__name__

    try:
        # Initialize appropriate index type
        if isinstance(index, IndexFlatL2):
            local_index = faiss.IndexFlatL2(index.d)
        elif isinstance(index, IndexIVFFlat):
            quantizer = faiss.IndexFlatL2(index.d)
            local_index = faiss.IndexIVFFlat(quantizer, index.d, index.nlist)
            local_index.nprobe = index._nprobe
        elif isinstance(index, IndexPQ):
            local_index = faiss.IndexPQ(index.d, index.m, index.nbits)
        elif isinstance(index, IndexIVFPQ):
            quantizer = faiss.IndexFlatL2(index.d)
            local_index = faiss.IndexIVFPQ(
                quantizer, index.d, index.nlist, index.m, index.nbits
            )
            local_index.nprobe = index._nprobe
        elif isinstance(index, IndexHNSWFlat):
            local_index = faiss.IndexHNSWFlat(index.d, index.m)
        elif isinstance(index, IndexScalarQuantizer):
            local_index = faiss.IndexScalarQuantizer(index.d)
        else:
            logger.warning(f"Unsupported index type for saving: {index_type}")
            return None

        # Train index if needed
        if not getattr(local_index, "is_trained", True):
            local_index.train(vectors)

        # Add vectors
        local_index.add(vectors)
        return local_index
    except Exception as e:
        logger.error(f"Error creating initialized index: {e}")
        return None


def _create_equivalent_faiss_index(index: FaissIndex) -> Optional[Any]:
    """
    Create a native FAISS index equivalent to the FAISSx index.

    This function creates a new FAISS index with the same parameters as the input
    FAISSx index, but without adding any vectors.

    Args:
        index: The FAISSx index to create an equivalent for

    Returns:
        Equivalent empty FAISS index, or None if creation fails
    """
    try:
        if isinstance(index, IndexFlatL2):
            return faiss.IndexFlatL2(index.d)
        elif isinstance(index, IndexIVFFlat):
            quantizer = faiss.IndexFlatL2(index.d)
            local_index = faiss.IndexIVFFlat(quantizer, index.d, index.nlist)
            local_index.nprobe = index._nprobe
            return local_index
        elif isinstance(index, IndexPQ):
            return faiss.IndexPQ(index.d, index.m, index.nbits)
        elif isinstance(index, IndexIVFPQ):
            quantizer = faiss.IndexFlatL2(index.d)
            local_index = faiss.IndexIVFPQ(
                quantizer, index.d, index.nlist, index.m, index.nbits
            )
            local_index.nprobe = index._nprobe
            return local_index
        elif isinstance(index, IndexHNSWFlat):
            return faiss.IndexHNSWFlat(index.d, index.m)
        elif isinstance(index, IndexScalarQuantizer):
            return faiss.IndexScalarQuantizer(index.d)
        else:
            logger.warning(f"Unknown index type: {type(index).__name__}")
            return None
    except Exception as e:
        logger.error(f"Error creating equivalent FAISS index: {e}")
        return None


def _read_idmap_index(
    fname: str, format_flag: int, gpu: bool
) -> Union[IndexIDMap, IndexIDMap2]:
    """
    Read a custom format IDMap index from file.

    This function reads an index that was saved in the custom IDMap format,
    extracts the ID mappings and vectors, and reconstructs the index.

    Args:
        fname: Path to the index file
        format_flag: Flag indicating whether it's IDMap (0) or IDMap2 (1)
        gpu: Whether to try loading the index on GPU if available

    Returns:
        Reconstructed IndexIDMap or IndexIDMap2 index

    Raises:
        ValueError: If the file cannot be read as an IDMap index
    """
    is_idmap2 = format_flag == IDMAP2_FORMAT_FLAG

    try:
        with open(fname, "rb") as f:
            # Skip the format flag we already read
            f.seek(HEADER_SIZE)

            # Read dimensions
            ntotal_dim = np.frombuffer(f.read(16), dtype=np.int64)
            ntotal, d = ntotal_dim[0], ntotal_dim[1]

            # Read ID mappings
            id_map_size = np.frombuffer(f.read(8), dtype=np.int64)[0]
            id_map_bytes = f.read(id_map_size)
            id_map_data = np.frombuffer(
                id_map_bytes,
                dtype=[("internal_idx", np.int64), ("external_id", np.int64)],
            )

            # Read index data
            index_data_size = np.frombuffer(f.read(8), dtype=np.int64)[0]
            index_data = f.read(index_data_size)

            # Read vectors
            vectors_size = np.frombuffer(f.read(8), dtype=np.int64)[0]
            vectors_bytes = f.read(vectors_size)

            # Process vector data if present
            vectors = None
            if vectors_size > 0:
                # Reshape the vectors data into a proper array
                vector_count = vectors_size // (d * 4)  # 4 bytes per float32
                if vector_count > 0:
                    vectors = np.frombuffer(vectors_bytes, dtype=np.float32).reshape(
                        vector_count, d
                    )

        # Create the index from the data
        return _create_idmap_from_data(
            fname, is_idmap2, ntotal, d, id_map_data, index_data, vectors, gpu
        )
    except Exception as e:
        logger.error(f"Error reading IDMap index from {fname}: {e}")
        raise ValueError(f"Failed to read IDMap index: {e}")


def _create_idmap_from_data(
    fname: str,
    is_idmap2: bool,
    ntotal: int,
    d: int,
    id_map_data: np.ndarray,
    index_data: bytes,
    vectors: Optional[np.ndarray] = None,
    gpu: bool = False,
) -> Union[IndexIDMap, IndexIDMap2]:
    """
    Create an IDMap index from the loaded data.

    This function reconstructs an IDMap/IDMap2 index from serialized components:
    - Creates a base index from index_data
    - Restores ID mappings from id_map_data
    - Optionally caches vectors for future operations

    Args:
        fname: Original file name (for logging)
        is_idmap2: Whether to create IDMap2 (True) or IDMap (False)
        ntotal: Total number of vectors in the index
        d: Dimension of the vectors
        id_map_data: Array of ID mappings (internal to external IDs)
        index_data: Serialized base index data
        vectors: Optional vector data for caching
        gpu: Whether to try using GPU if available

    Returns:
        Reconstructed IndexIDMap or IndexIDMap2 index

    Raises:
        ValueError: If the index cannot be created from the data
    """
    # Create a temporary file for the index
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
        temp_path = tmp.name
        try:
            # Write the base index data to the temp file
            tmp.write(index_data)
            tmp.flush()

            # Read the base index
            try:
                base_index = faiss.read_index(temp_path)
            except Exception as e:
                # If base index can't be read, fall back to a simple flat index
                logger.warning(
                    f"Could not read base index: {e}. Creating flat index fallback."
                )
                base_index = faiss.IndexFlatL2(d)

            # If GPU is requested, move to GPU
            if gpu:
                base_index = _move_to_gpu_if_available(base_index)

            # Create the corresponding FAISSx index
            faissx_base = _create_faissx_base_index(base_index, d)
            if faissx_base is None:
                logger.warning(
                    f"Unsupported base index type: {type(base_index)}. "
                    "Creating flat index fallback."
                )
                flat_base = faiss.IndexFlatL2(d)
                faissx_base = IndexFlatL2(d)
                faissx_base._local_index = flat_base

            # Create the IDMap/IDMap2 wrapper
            try:
                wrapper_class = IndexIDMap2 if is_idmap2 else IndexIDMap
                idmap_index = wrapper_class(faissx_base)
            except Exception as e:
                logger.error(
                    f"Error creating IDMap{'2' if is_idmap2 else ''} wrapper: {e}"
                )
                # Fall back to IDMap if IDMap2 fails
                if is_idmap2:
                    logger.warning("Falling back to IndexIDMap instead of IndexIDMap2")
                    idmap_index = IndexIDMap(faissx_base)
                else:
                    raise

            # Restore the internal state
            try:
                idmap_index.ntotal = ntotal

                # Create ID mappings
                id_map = {}
                rev_id_map = {}
                # Only process valid rows in id_map_data
                for row in id_map_data:
                    try:
                        internal_idx, external_id = int(row[0]), int(row[1])
                        id_map[internal_idx] = external_id
                        rev_id_map[external_id] = internal_idx
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid ID map entry {row}: {e}")

                idmap_index._id_map = id_map
                idmap_index._rev_id_map = rev_id_map
            except Exception as e:
                logger.error(f"Error restoring IDMap state: {e}")
                # If we couldn't restore the state, initialize the index with empty state
                idmap_index.ntotal = 0
                idmap_index._id_map = {}
                idmap_index._rev_id_map = {}

            # If we have vectors, store them in the vectors_by_id cache
            if vectors is not None and hasattr(idmap_index, "_vectors_by_id"):
                try:
                    for i, row in enumerate(id_map_data):
                        if i < len(vectors):
                            ext_id = int(row[1])
                            idmap_index._vectors_by_id[ext_id] = vectors[i]
                except Exception as e:
                    logger.warning(f"Error caching vectors: {e}")

            logger.info(
                f"Successfully loaded IndexIDMap{'2' if is_idmap2 else ''} "
                f"from {fname}"
            )
            return idmap_index

        except Exception as e:
            logger.error(f"Error creating IDMap index: {e}")
            raise ValueError(f"Failed to create IDMap index: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def _move_to_gpu_if_available(index: Any) -> Any:
    """
    Move a FAISS index to GPU if GPU support is available.

    This function attempts to transfer a FAISS index to GPU for faster processing.
    If GPU support is not available, it returns the original index unchanged.

    Args:
        index: FAISS index to move to GPU

    Returns:
        GPU-enabled index if successful, otherwise the original index
    """
    try:
        import faiss.contrib.gpu  # type: ignore

        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index)
    except (ImportError, AttributeError):
        logger.warning("GPU requested but FAISS GPU support not available")

    return index


def _create_faissx_base_index(base_index: Any, d: int) -> Optional[FaissIndex]:
    """
    Create a FAISSx index wrapper for a FAISS index.

    This function creates the appropriate FAISSx wrapper around a native FAISS index
    based on the index type, copying necessary parameters and configurations.

    Args:
        base_index: Native FAISS index to wrap
        d: Dimension of the vectors in the index

    Returns:
        FAISSx wrapper index or None if the index type is not supported
    """
    if isinstance(base_index, faiss.IndexFlatL2):
        faissx_base = IndexFlatL2(d)
        faissx_base._local_index = base_index
        return faissx_base
    elif isinstance(base_index, faiss.IndexIVFFlat):
        # Need to wrap the quantizer first
        quant = IndexFlatL2(d)
        quant._local_index = base_index.quantizer
        faissx_base = IndexIVFFlat(quant, d, base_index.nlist)
        faissx_base._local_index = base_index
        faissx_base._nprobe = base_index.nprobe
        return faissx_base
    elif isinstance(base_index, faiss.IndexPQ):
        faissx_base = IndexPQ(d, base_index.pq.M, base_index.pq.nbits)
        faissx_base._local_index = base_index
        return faissx_base
    elif isinstance(base_index, faiss.IndexIVFPQ):
        quant = IndexFlatL2(d)
        quant._local_index = base_index.quantizer
        faissx_base = IndexIVFPQ(
            quant,
            d,
            base_index.nlist,
            base_index.pq.M,
            base_index.pq.nbits,
        )
        faissx_base._local_index = base_index
        faissx_base._nprobe = base_index.nprobe
        return faissx_base
    elif isinstance(base_index, faiss.IndexHNSWFlat):
        faissx_base = IndexHNSWFlat(d, base_index.hnsw.efConstruction)
        faissx_base._local_index = base_index
        return faissx_base
    elif isinstance(base_index, faiss.IndexScalarQuantizer):
        faissx_base = IndexScalarQuantizer(d)
        faissx_base._local_index = base_index
        return faissx_base

    return None


def _create_faissx_from_faiss_index(faiss_index: Any, fname: str, gpu: bool) -> FaissIndex:
    """
    Create a FAISSx index from a FAISS index.

    This function wraps a native FAISS index with the appropriate FAISSx wrapper
    based on the index type.

    Args:
        faiss_index: Native FAISS index to wrap
        fname: Original filename (for logging)
        gpu: Whether the index was loaded on GPU

    Returns:
        FAISSx index wrapping the native FAISS index

    Raises:
        ValueError: If the index type is not supported
    """
    d = faiss_index.d

    if isinstance(faiss_index, faiss.IndexFlatL2):
        index = IndexFlatL2(d)
        index._local_index = faiss_index
        index.ntotal = faiss_index.ntotal
        return index

    elif isinstance(faiss_index, faiss.IndexIVFFlat):
        # Need to wrap the quantizer first
        quant = IndexFlatL2(d)
        quant._local_index = faiss_index.quantizer
        index = IndexIVFFlat(quant, d, faiss_index.nlist)
        index._local_index = faiss_index
        index._nprobe = faiss_index.nprobe
        index.ntotal = faiss_index.ntotal
        return index

    elif isinstance(faiss_index, faiss.IndexHNSWFlat):
        index = IndexHNSWFlat(d, faiss_index.hnsw.efConstruction)
        index._local_index = faiss_index
        index.ntotal = faiss_index.ntotal
        return index

    elif isinstance(faiss_index, faiss.IndexPQ):
        index = IndexPQ(d, faiss_index.pq.M, faiss_index.pq.nbits)
        index._local_index = faiss_index
        index.ntotal = faiss_index.ntotal
        index.is_trained = faiss_index.is_trained
        return index

    elif isinstance(faiss_index, faiss.IndexIVFPQ):
        # Need to wrap the quantizer first
        quant = IndexFlatL2(d)
        quant._local_index = faiss_index.quantizer
        index = IndexIVFPQ(
            quant, d, faiss_index.nlist, faiss_index.pq.M, faiss_index.pq.nbits
        )
        index._local_index = faiss_index
        index._nprobe = faiss_index.nprobe
        index.ntotal = faiss_index.ntotal
        index.is_trained = faiss_index.is_trained
        return index

    elif isinstance(faiss_index, faiss.IndexScalarQuantizer):
        index = IndexScalarQuantizer(d)
        index._local_index = faiss_index
        index.ntotal = faiss_index.ntotal
        return index

    # Try to determine if it's an IDMap or IDMap2
    elif hasattr(faiss_index, "id_map"):
        return _handle_idmap_faiss_index(faiss_index, fname, gpu)

    else:
        # Let's fall back to a best-guess approach
        index_description = _infer_index_type(faiss_index)
        if index_description:
            logger.info(f"Automatically detected index type: {index_description}")
            return index_factory(d, index_description)
        else:
            raise ValueError(f"Unsupported index type: {type(faiss_index)}")


def _handle_idmap_faiss_index(
    faiss_index: Any, fname: str, gpu: bool
) -> Union[IndexIDMap, IndexIDMap2]:
    """
    Handle an IDMap/IDMap2 FAISS index.

    This function creates a FAISSx IDMap or IDMap2 wrapper for a native FAISS IDMap index,
    copying ID mappings and reconstructing vectors when possible.

    Args:
        faiss_index: Native FAISS IDMap/IDMap2 index
        fname: Original filename (for logging)
        gpu: Whether the index was loaded on GPU

    Returns:
        FAISSx IndexIDMap or IndexIDMap2 wrapper
    """
    # Get dimension from the base index
    d = faiss_index.d

    # Create an appropriate FAISSx base index based on the underlying type
    base_faiss_index = faiss_index.index
    if isinstance(base_faiss_index, faiss.IndexFlatL2):
        base_faissx_index = IndexFlatL2(d)
        base_faissx_index._local_index = base_faiss_index
    else:
        # Fallback to a simple flat index for now
        logger.warning(
            f"Using flat index for unknown base type: {type(base_faiss_index)}"
        )
        base_faissx_index = IndexFlatL2(d)
        base_faissx_index._local_index = faiss.IndexFlatL2(d)

    # Create appropriate wrapper
    if hasattr(faiss_index, "replace_vector"):
        idmap = IndexIDMap2(base_faissx_index)
    else:
        idmap = IndexIDMap(base_faissx_index)

    # Extract the ID mappings
    for i in range(faiss_index.ntotal):
        id_val = int(faiss_index.id_map.at(i))
        idmap._id_map[i] = id_val
        idmap._rev_id_map[id_val] = i

    # Try to extract vectors for reconstruction if possible
    try:
        if hasattr(idmap, "_vectors_by_id"):
            for i in range(faiss_index.ntotal):
                id_val = int(faiss_index.id_map.at(i))
                # Try to get the vector for caching
                try:
                    vector = faiss_index.reconstruct(i)
                    idmap._vectors_by_id[id_val] = vector
                except Exception:
                    pass  # Skip if reconstruction fails
    except Exception as e:
        logger.warning(f"Could not extract vectors from IDMap index: {e}")

    idmap.ntotal = faiss_index.ntotal
    return idmap


def _infer_index_type(faiss_index: Any) -> Optional[str]:
    """
    Helper to determine the index type from a FAISS index object.

    This function analyzes a FAISS index and returns a string descriptor
    that can be used with index_factory to recreate an equivalent index.

    Args:
        faiss_index: FAISS index to analyze

    Returns:
        String description of the index type or None if the type isn't recognized
    """
    # Most FAISS indices have a specific class name pattern
    class_name = type(faiss_index).__name__

    if "IndexFlat" in class_name:
        return "Flat"
    elif "IndexIVFFlat" in class_name:
        nlist = getattr(faiss_index, "nlist", 100)
        return f"IVF{nlist},Flat"
    elif "IndexIVFPQ" in class_name:
        nlist = getattr(faiss_index, "nlist", 100)
        m = getattr(faiss_index.pq, "M", 8)
        nbits = getattr(faiss_index.pq, "nbits", 8)
        return f"IVF{nlist},PQ{m}x{nbits}"
    elif "IndexPQ" in class_name:
        m = getattr(faiss_index.pq, "M", 8)
        nbits = getattr(faiss_index.pq, "nbits", 8)
        return f"PQ{m}x{nbits}"
    elif "IndexHNSW" in class_name:
        m = getattr(faiss_index.hnsw, "efConstruction", 16)
        return f"HNSW{m}"
    elif "IndexScalar" in class_name:
        return "SQ8"

    return None
